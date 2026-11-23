use super::*;

#[test]
fn copies_are_named_for_weights_and_scales() {
    assert_eq!(
        copy_name("blk.3.attn_qkv.weight").as_deref(),
        Some("blk.3.attn_qkv.fp8.weight")
    );
    assert_eq!(
        copy_name("blk.3.attn_qkv.scale").as_deref(),
        Some("blk.3.attn_qkv.fp8.scale")
    );
    assert_eq!(
        copy_name("blk.47.ffn_down_shexp.scale").as_deref(),
        Some("blk.47.ffn_down_shexp.fp8.scale")
    );
    for name in [
        "blk.3.attn_qkv.input_scale",
        "blk.3.ssm_alpha.weight",
        "blk.3.ffn_up_exps.scale",
        "blk.3.attn_qkv.fp8.weight",
        "output.weight",
    ] {
        assert_eq!(copy_name(name), None, "{name}");
    }
}

#[test]
fn only_small_steps_of_a_file_with_copies_read_them() {
    assert!(takes_step(1) && takes_step(64));
    assert!(!takes_step(65) && !takes_step(4096));
    let on = DenseTwins {
        on: true,
        prompt: false,
        active: Cell::new(false),
    };
    let w = "blk.0.attn_q.weight";
    assert_eq!(on.copy_of(w), None, "outside a graph build");
    {
        let _g = on.step(64);
        assert_eq!(on.copy_of(w).as_deref(), Some("blk.0.attn_q.fp8.weight"));
        {
            let _inner = on.step(4096);
            assert_eq!(on.copy_of(w), None, "a prompt step");
        }
        assert!(on.copy_of(w).is_some(), "the outer build's state is back");
    }
    assert_eq!(on.copy_of(w), None);
    let off = DenseTwins::default();
    let _g = off.step(1);
    assert_eq!(off.copy_of(w), None, "no copies in the file");
}

#[test]
fn prompt_mode_reads_the_copies_in_every_step() {
    let mut on = DenseTwins {
        on: true,
        prompt: false,
        active: Cell::new(false),
    };
    assert!(on.set_prompt(true));
    let w = "blk.0.attn_q.weight";
    assert_eq!(on.copy_of(w), None, "outside a graph build");
    {
        let _g = on.step(4096);
        assert_eq!(on.copy_of(w).as_deref(), Some("blk.0.attn_q.fp8.weight"));
    }
    {
        let _g = on.step(1);
        assert!(on.copy_of(w).is_some());
    }
    assert!(!on.set_prompt(false));
    let _g = on.step(4096);
    assert_eq!(on.copy_of(w), None, "prompt mode off");
    let mut off = DenseTwins::default();
    assert!(!off.set_prompt(true), "no copies in the file");
}

/// Copies in one CPU buffer as the loader stores them: layer 0's query, key and value codes
/// then their scales, layer 0's shared-expert gate and up the same way; layer 1's query run
/// with a filler between key and value codes, and layer 2's without its value.
fn copies() -> Context {
    let ctx = Context::new(32, 0);
    let add = |name: &str, ty, shape: &[i64]| {
        ctx.new_tensor(ty, shape).named(name);
    };
    let codes = |p: &str, l: usize| format!("blk.{l}.{p}.fp8.weight");
    let scale = |p: &str, l: usize| format!("blk.{l}.{p}.fp8.scale");
    for (l, run, rows) in [
        (0, &["attn_q", "attn_k", "attn_v"][..], &[96i64, 32, 32][..]),
        (0, &["ffn_gate_shexp", "ffn_up_shexp"][..], &[40, 40][..]),
        (2, &["attn_q", "attn_k"][..], &[96, 32][..]),
    ] {
        for (p, &m) in run.iter().zip(rows) {
            add(&codes(p, l), types::I8, &[64, m]);
        }
        for (p, &m) in run.iter().zip(rows) {
            add(&scale(p, l), types::F32, &[m]);
        }
    }
    add(&codes("attn_q", 1), types::I8, &[64, 96]);
    add(&codes("attn_k", 1), types::I8, &[64, 32]);
    add("filler", types::I8, &[64]);
    add(&codes("attn_v", 1), types::I8, &[64, 32]);
    for (p, m) in [("attn_q", 96), ("attn_k", 32), ("attn_v", 32)] {
        add(&scale(p, 1), types::F32, &[m]);
    }
    ctx.alloc_tensors(crate::ggml::Backend::cpu().unwrap().buffer_type())
        .unwrap();
    ctx
}

#[test]
fn runs_join_adjacent_copies_only() {
    let w = copies();
    let get = |n: &str| w.tensor(n);
    assert!(Runs::build(3, get, false).is_none(), "host memory");
    let runs = Runs::build(3, get, true).expect("joined runs");
    assert_eq!(runs.len(), 2, "layer 0's two runs");
    let t = |n: &str| w.tensor(n).unwrap();
    let qkv = ["attn_q", "attn_k", "attn_v"];
    let members = |l: usize, ps: &[&str], what: &str| {
        ps.iter()
            .map(|p| t(&format!("blk.{l}.{p}.fp8.{what}")))
            .collect::<Vec<_>>()
    };
    let (c, s) = runs
        .get(&members(0, &qkv, "weight"), &members(0, &qkv, "scale"))
        .expect("layer 0's query, key and value");
    assert_eq!((c.ty(), c.ne()), (types::I8, [64, 160, 1, 1]));
    assert_eq!((s.ty(), s.ne()), (types::F32, [160, 1, 1, 1]));
    assert_eq!(addr(c), addr(t("blk.0.attn_q.fp8.weight")));
    assert_eq!(addr(s), addr(t("blk.0.attn_q.fp8.scale")));
    let sh = ["ffn_gate_shexp", "ffn_up_shexp"];
    let (c, s) = runs
        .get(&members(0, &sh, "weight"), &members(0, &sh, "scale"))
        .expect("layer 0's shared-expert gate and up");
    assert_eq!((c.ne()[1], s.ne()[0]), (80, 80));
    // Not the members it was placed over (two of three), and layer 1's (a gap) is not joined.
    assert!(
        runs.get(
            &members(0, &qkv[..2], "weight"),
            &members(0, &qkv[..2], "scale")
        )
        .is_none()
    );
    assert!(
        runs.get(&members(1, &qkv, "weight"), &members(1, &qkv, "scale"))
            .is_none()
    );
}
