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
        small: Cell::new(false),
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
