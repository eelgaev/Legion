use super::*;
use crate::gguf::{GgufWriter, Value};

fn gguf_with(kv: &[(&str, Value)]) -> (tempfile::TempDir, GgufFile) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.gguf");
    let mut w = GgufWriter::new();
    for (k, v) in kv {
        w.set(k, v).unwrap();
    }
    w.add_tensor("x.weight", types::F32, &[1], &1f32.to_ne_bytes())
        .unwrap();
    w.write(&path).unwrap();
    let g = GgufFile::open(&path).unwrap();
    (dir, g)
}

fn strs(v: &[&str]) -> Value {
    Value::Array(v.iter().map(|s| Value::String(s.to_string())).collect())
}

#[test]
fn reads_prec_a4_list() {
    let (_d, g) = gguf_with(&[
        (KEY_NAMES, strs(&["a.weight", "b.weight"])),
        (
            KEY_PREC_A4,
            Value::Array(vec![Value::Bool(false), Value::Bool(true)]),
        ),
    ]);
    let l = Linear::from_gguf(&g).unwrap();
    assert!(l.keeps_a8("a.weight"));
    assert!(!l.keeps_a8("b.weight"));
    assert!(!l.keeps_a8("c.weight"));
}

#[test]
fn reads_the_fp8_list() {
    let (_d, g) = gguf_with(&[(KEY_FP8_E4M3, strs(&["blk.0.attn_q.weight"]))]);
    let l = Linear::from_gguf(&g).unwrap();
    assert!(l.fp8.contains("blk.0.attn_q.weight"));
}

#[test]
fn all_a8_covers_unlisted_weights() {
    let (_d, g) = gguf_with(&[
        (KEY_NAMES, strs(&["a.weight"])),
        (KEY_PREC_A4, Value::Array(vec![Value::Bool(true)])),
    ]);
    let mut l = Linear::from_gguf(&g).unwrap();
    assert!(!l.keeps_a8("a.weight") && !l.keeps_a8("blk.1.ffn_up_exps.weight"));
    l.set_all_a8(true);
    assert!(l.keeps_a8("a.weight") && l.keeps_a8("blk.1.ffn_up_exps.weight"));
}

#[test]
fn missing_lists_mean_no_hints() {
    let (_d, g) = gguf_with(&[]);
    assert_eq!(Linear::from_gguf(&g).unwrap(), Linear::default());
}

#[test]
fn rejects_mismatched_lists() {
    let (_d, g) = gguf_with(&[
        (KEY_NAMES, strs(&["a.weight", "b.weight"])),
        (KEY_PREC_A4, Value::Array(vec![Value::Bool(false)])),
    ]);
    assert!(matches!(
        Linear::from_gguf(&g),
        Err(ModelError::Metadata(k)) if k == KEY_PREC_A4
    ));
}

#[test]
fn scale_names() {
    assert_eq!(
        scale_name("blk.3.ffn_up.weight").as_deref(),
        Some("blk.3.ffn_up.scale")
    );
    assert_eq!(scale_name("output.weight").as_deref(), Some("output.scale"));
    assert_eq!(scale_name("blk.0.ssm_a"), None);
}

/// A decode step's expert matmul has one token: the per-expert scales still apply (the
/// token count comes from `ne`, which keeps trailing 1s, not `shape`, which drops them).
#[test]
fn per_expert_scales_apply_to_one_token_steps() {
    use crate::ggml::{Backend, ComputeAllocator};
    let (k, m, n_expert) = (4usize, 2usize, 3usize);
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(16, 32);
    // Expert e's weights are all e + 1; x is all ones; expert 2 is used.
    let w = ctx
        .new_tensor(types::F32, &[k as i64, m as i64, n_expert as i64])
        .as_input();
    let scale = ctx.new_tensor(types::F32, &[n_expert as i64]).as_input();
    let x = ctx.new_tensor(types::F32, &[k as i64, 1, 1]).as_input();
    let ids = ctx.new_tensor(types::I32, &[1, 1]).as_input();
    let out = Linear::default()
        .mul_mat_id(&ctx, "blk.0.ffn_up_exps.weight", w, Some(scale), x, ids)
        .unwrap()
        .as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let wv: Vec<f32> = (0..n_expert)
        .flat_map(|e| std::iter::repeat_n(e as f32 + 1.0, k * m))
        .collect();
    w.set_f32(&wv);
    scale.set_f32(&[0.5, 0.25, 0.125]);
    x.set_f32(&[1.0; 4]);
    ids.set_i32(&[2]);
    backend.compute(&graph).unwrap();
    assert_eq!(out.to_f32(), vec![3.0 * 4.0 * 0.125; m]);
}

/// The fused residual adds (#749) are built only for the opt-in prefill batches: `a8-dynamic`
/// with FP8 weights, or dense W4A4 batches; never by default.
#[test]
fn residual_fusion_only_on_opt_in_prefill_batches() {
    let mut l = Linear::default();
    let n = NVFP4_A4_MIN_N;
    assert!(!l.residual_fusion(8192), "defaults");
    l.set_nvfp4_a4(Nvfp4A4::Prefill);
    assert!(l.residual_fusion(n + 1) && !l.residual_fusion(n));
    l.set_nvfp4_a4(Nvfp4A4::Experts);
    assert!(!l.residual_fusion(8192), "routed experts only");
    l.set_nvfp4_a4(Nvfp4A4::Off);
    l.set_fp8_act(Fp8Act::A8Dynamic);
    l.fp8.insert("blk.0.attn_output.weight".into());
    assert!(l.residual_fusion(n + 1) && !l.residual_fusion(n));
    l.set_fp8_act(Fp8Act::A8);
    assert!(!l.residual_fusion(8192));
}

/// Where no kernel fuses it (here: an F32 weight on the CPU), `mul_mat_add` is the matmul
/// followed by ggml's `add`.
#[test]
fn mul_mat_add_falls_back_to_add() {
    use crate::ggml::{Backend, ComputeAllocator};
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(16, 32);
    let w = ctx.new_tensor(types::F32, &[4, 2]).as_input();
    let x = ctx.new_tensor(types::F32, &[4, 3]).as_input();
    let res = ctx.new_tensor(types::F32, &[2, 3]).as_input();
    let mut l = Linear::default();
    l.set_nvfp4_a4(Nvfp4A4::All);
    let out = l
        .mul_mat_add(&ctx, "blk.0.ffn_down.weight", w, None, None, x, res)
        .unwrap()
        .as_output();
    let mut graph = ctx.new_graph(32);
    graph.expand(out);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    w.set_f32(&[1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5]);
    x.set_f32(&[1.0; 12]);
    res.set_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    backend.compute(&graph).unwrap();
    assert_eq!(out.to_f32(), vec![20.0, 22.0, 40.0, 42.0, 60.0, 62.0]);
}

/// `--nvfp4-a4`: which batches take 4-bit activations, before the device check.
#[test]
fn nvfp4_a4_batches() {
    let mut l = Linear::default();
    let n = NVFP4_A4_MIN_N;
    for (a4, want) in [
        (Nvfp4A4::Off, [false, false, false, false]),
        (Nvfp4A4::Experts, [false, false, false, true]),
        (Nvfp4A4::Prefill, [false, true, false, true]),
        (Nvfp4A4::All, [true, true, true, true]),
        (Nvfp4A4::ExpertsDecode, [false, false, true, true]),
    ] {
        l.set_nvfp4_a4(a4);
        let got = [
            l.a4_batch(n, false),
            l.a4_batch(n + 1, false),
            l.a4_batch(n, true),
            l.a4_batch(n + 1, true),
        ];
        assert_eq!(got, want, "{a4:?}");
    }
    // `experts-decode` takes the routed experts' decode steps above ggml's MMVQ range (9+
    // tokens), nothing dense; `experts` none of them.
    l.set_nvfp4_a4(Nvfp4A4::ExpertsDecode);
    let d = NVFP4_GGML_MAX_N;
    assert!(!l.a4_batch(d, true) && l.a4_batch(d + 1, true) && l.a4_batch(16, true));
    assert!(!l.a4_batch(d + 1, false) && !l.a4_batch(4096, false));
    l.set_nvfp4_a4(Nvfp4A4::Experts);
    assert!(!l.a4_batch(16, true));
    // Weights in host memory never take them.
    let backend = crate::ggml::Backend::cpu().unwrap();
    let ctx = Context::new(1, 0);
    let w = ctx.new_tensor(types::NVFP4, &[64, 8]);
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    assert!(!l.w4a4(w, 4096, false));
}

/// `first_rows_scale`: a per-tensor scale as it is, a per-row one cut to the first rows, and the
/// FP8 matmul of the first rows with it gives exactly those rows of the whole product (the
/// channel-wise FP8 output head under `--spec-draft-vocab`). Other layouts: `None`.
#[test]
fn first_rows_scale_cuts_a_scale_per_row() {
    use crate::ggml::{Backend, ComputeAllocator};
    let (k, m, rows, n) = (64usize, 40usize, 24i64, 3usize);
    let mut cpu = Backend::cpu().unwrap();
    let weights = Context::new(4, 0);
    let w = weights.new_tensor(types::I8, &[k as i64, m as i64]);
    let per_row = weights.new_tensor(types::F32, &[m as i64]);
    let one = weights.new_tensor(types::F32, &[1]);
    let other = weights.new_tensor(types::F32, &[m as i64 / 2, 2]);
    weights.alloc_tensors(cpu.buffer_type()).unwrap();
    // E4M3 bytes without NaNs (0x7f / 0xff), scales far apart per row.
    let bytes: Vec<u8> = (0..k * m).map(|i| (i * 37 % 251) as u8 & 0xf7).collect();
    w.set_bytes(&bytes);
    let scales: Vec<f32> = (0..m)
        .map(|j| 2f32.powi(j as i32 % 7 - 3) * (1.0 + (j % 5) as f32 / 4.0))
        .collect();
    per_row.set_f32(&scales);
    one.set_f32(&[0.5]);
    other.set_f32(&scales);

    let ctx = Context::new(16, 32);
    assert!(first_rows_scale(&ctx, other, m as i64, rows).is_none());
    assert_eq!(
        first_rows_scale(&ctx, one, m as i64, rows).unwrap().raw(),
        one.raw()
    );
    let cut = first_rows_scale(&ctx, per_row, m as i64, rows).unwrap();
    assert_eq!(cut.ne(), [rows, 1, 1, 1]);
    let x = ctx.new_tensor(types::F32, &[k as i64, n as i64]).as_input();
    let whole = crate::model::fp8::mul_mat(&ctx, w, x, Some(per_row)).as_output();
    let part = ctx.view_2d(w, [k as i64, rows], w.nb()[1], 0);
    let first = crate::model::fp8::mul_mat(&ctx, part, x, Some(cut)).as_output();
    let mut graph = ctx.new_graph(16);
    graph.expand(whole);
    graph.expand(first);
    ctx.alloc_graph(&graph, ComputeAllocator::new(cpu.buffer_type()))
        .unwrap();
    let xs: Vec<f32> = (0..k * n)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 9.0)
        .collect();
    x.set_f32(&xs);
    cpu.compute(&graph).unwrap();
    let (whole, first) = (whole.to_f32(), first.to_f32());
    for t in 0..n {
        let want = &whole[t * m..t * m + rows as usize];
        let got = &first[t * rows as usize..(t + 1) * rows as usize];
        assert!(
            want.iter()
                .zip(got)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "token {t}"
        );
    }
}

/// `--fp8-act`: the static input scale wins where the weight has one; `a8-dynamic` takes
/// per-token scales only for prefill batches (more than `NVFP4_A4_MIN_N` tokens) of weights
/// without one, where the device allows it; everything else stays W8A16.
#[test]
fn fp8_routes() {
    let mut l = Linear::default();
    let (n, k) = (NVFP4_A4_MIN_N, 128);
    let yes = || true;
    // Default `a8`: static where there is an input scale, W8A16 elsewhere, at every size.
    assert_eq!(l.fp8_route(1, k, true, yes), Fp8Route::Static);
    assert_eq!(l.fp8_route(4096, k, true, yes), Fp8Route::Static);
    assert_eq!(l.fp8_route(4096, k, false, yes), Fp8Route::A16);
    assert_eq!(l.fp8_route(4096, 96, true, yes), Fp8Route::A16);
    l.set_fp8_act(Fp8Act::A16);
    assert_eq!(l.fp8_route(4096, k, true, yes), Fp8Route::A16);
    l.set_fp8_act(Fp8Act::PrefillA8);
    assert_eq!(l.fp8_route(8, k, true, yes), Fp8Route::A16);
    assert_eq!(l.fp8_route(9, k, true, yes), Fp8Route::Static);
    assert_eq!(l.fp8_route(4096, k, false, yes), Fp8Route::A16);

    l.set_fp8_act(Fp8Act::A8Dynamic);
    assert_eq!(l.fp8_route(1, k, true, yes), Fp8Route::Static);
    assert_eq!(l.fp8_route(n + 1, k, true, yes), Fp8Route::Static);
    assert_eq!(l.fp8_route(1, k, false, yes), Fp8Route::A16);
    assert_eq!(l.fp8_route(n, k, false, yes), Fp8Route::A16);
    assert_eq!(l.fp8_route(n + 1, k, false, yes), Fp8Route::Dynamic);
    assert_eq!(l.fp8_route(8192, k, false, yes), Fp8Route::Dynamic);
    assert_eq!(l.fp8_route(8192, 96, false, yes), Fp8Route::A16);
    assert_eq!(l.fp8_route(8192, k, false, || false), Fp8Route::A16);
    // Decode steps never ask the device.
    assert_eq!(
        l.fp8_route(4, k, false, || panic!("asked for a decode step")),
        Fp8Route::A16
    );
}

/// `a8-dynamic` parses from the command line; `prefill-a8` stays internal.
#[test]
fn fp8_act_values() {
    use clap::ValueEnum;
    assert_eq!(
        Fp8Act::from_str("a8-dynamic", false).unwrap(),
        Fp8Act::A8Dynamic
    );
    assert!(Fp8Act::from_str("prefill-a8", false).is_err());
}

/// The fused per-token FP8 ops are built for `a8-dynamic` prefill batches of models that have
/// FP8 weights, nowhere else.
#[test]
fn dynamic_batch_needs_a8_dynamic_prefill_and_fp8_weights() {
    let mut l = Linear::default();
    let n = NVFP4_A4_MIN_N;
    l.set_fp8_act(Fp8Act::A8Dynamic);
    assert!(!l.dynamic_batch(4096), "no FP8 weights");
    l.fp8.insert("blk.0.attn_q.weight".into());
    assert!(l.dynamic_batch(n + 1));
    assert!(l.dynamic_batch(8192));
    assert!(!l.dynamic_batch(n));
    assert!(!l.dynamic_batch(1));
    for a in [Fp8Act::A8, Fp8Act::A16, Fp8Act::PrefillA8] {
        l.set_fp8_act(a);
        assert!(!l.dynamic_batch(8192), "{a:?}");
    }
}

/// The W4A4 down projection takes its residual when the residual has the output's shape (#749:
/// it was compared with the weight's input width, so the fused add never ran on the 27B).
#[test]
fn down_residual_fits_the_output_shape() {
    let ctx = Context::new(8, 0);
    // ffn_down of a 64-wide model with 128 FFN rows: [k = 128, m = 64]; 5 tokens.
    let w = ctx.new_tensor(types::NVFP4, &[128, 64]);
    let act = ctx.new_tensor(types::BF16, &[128, 5]);
    let res = ctx.new_tensor(types::F32, &[64, 5]);
    assert!(down_res_fits(w, act, res));
    // The input width is not the output's; nor are other token counts or types.
    for bad in [
        ctx.new_tensor(types::F32, &[128, 5]),
        ctx.new_tensor(types::F32, &[64, 4]),
        ctx.new_tensor(types::BF16, &[64, 5]),
    ] {
        assert!(!down_res_fits(w, act, bad), "{:?}", bad.ne());
    }
}
