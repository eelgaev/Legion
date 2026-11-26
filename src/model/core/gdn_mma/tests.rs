//! `gdn_mma` against ggml's `gated_delta_net` over q, k and v views of one conv-like row per
//! token, as a DeltaNet layer has them: the op's CPU side (the per-token reference) on the CPU,
//! the chunked kernel (ggml's, ported to src/cuda/gdn_mma.cu) on a GPU (ignored test).

use super::*;
use crate::ggml::{Backend, ComputeAllocator};

/// Held by the tests that change the module's switches ([`ON`], [`STAGED`], [`FORCE`]) or read
/// [`CODES_RAN`], or run the op on a GPU under them: one at a time.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Largest difference relative to the largest magnitude.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let top = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-30);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / top
}

/// A case: key heads `n_k`, value heads `n_v`, tokens `nt`, sequences `ns`.
type Case = (i64, i64, i64, i64);

/// (op, ggml) on `backend`: the tokens' outputs then the final states, and the milliseconds of
/// `reps` more runs of each (0: none).
fn case(
    backend: &mut Backend,
    (n_k, n_v, nt, ns): Case,
    reps: usize,
) -> (Vec<f32>, Vec<f32>, f64, f64) {
    let s = HEAD;
    let row = 2 * s * n_k + s * n_v;
    let ctx = Context::new(32, 32);
    let conv = ctx.new_tensor(types::F32, &[row, nt, ns]).as_input();
    let g = ctx.new_tensor(types::F32, &[1, n_v, nt, ns]).as_input();
    let beta = ctx.new_tensor(types::F32, &[1, n_v, nt, ns]).as_input();
    let state = ctx.new_tensor(types::F32, &[s, s, n_v, ns]).as_input();
    let es = 4usize;
    let nb = [s as usize * es, row as usize * es, (row * nt) as usize * es];
    let q = ctx.view_4d(conv, [s, n_k, nt, ns], nb, 0);
    let k = ctx.view_4d(conv, [s, n_k, nt, ns], nb, (s * n_k) as usize * es);
    let v = ctx.view_4d(conv, [s, n_v, nt, ns], nb, (2 * s * n_k) as usize * es);
    assert!(shapes_fit(q, k, v, g, beta, state), "{:?}", (n_k, n_v, nt));
    let op = gdn_mma(&ctx, q, k, v, g, beta, state).as_output();
    let reference = ctx.gated_delta_net(q, k, v, g, beta, state, 1).as_output();
    assert_eq!(op.ne(), reference.ne(), "the op keeps ggml's shape");
    let mut op_graph = ctx.new_graph(16);
    op_graph.expand(op);
    let mut ref_graph = ctx.new_graph(16);
    ref_graph.expand(reference);
    let mut graph = ctx.new_graph(32);
    graph.expand(op);
    graph.expand(reference);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    // q and k unit rows (as the l2 norms leave them), g < 0 (a decay), beta in (0, 1).
    let mut x = noise((row * nt * ns) as usize, 1);
    for tok in x.chunks_mut(row as usize) {
        for head in tok[..(2 * s * n_k) as usize].chunks_mut(s as usize) {
            let n = head.iter().map(|v| v * v).sum::<f32>().sqrt();
            head.iter_mut().for_each(|v| *v /= n);
        }
    }
    conv.set_f32(&x);
    let gb = (n_v * nt * ns) as usize;
    let map = |seed, f: fn(f32) -> f32| noise(gb, seed).into_iter().map(f).collect::<Vec<_>>();
    g.set_f32(&map(2, |v| 0.5 * (v - 1.0)));
    beta.set_f32(&map(3, |v| 0.5 * (v + 1.0)));
    state.set_f32(
        &noise((s * s * n_v * ns) as usize, 4)
            .iter()
            .map(|v| 0.1 * v)
            .collect::<Vec<_>>(),
    );
    backend.compute(&graph).unwrap();
    let mut time = |graph: &crate::ggml::Graph<'_>| {
        if reps == 0 {
            return 0.0;
        }
        backend.compute(graph).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            backend.compute(graph).unwrap();
        }
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    let (op_ms, ref_ms) = (time(&op_graph), time(&ref_graph));
    (op.to_f32(), reference.to_f32(), op_ms, ref_ms)
}

#[test]
fn op_matches_ggml_on_the_cpu() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(4);
    // A rank's ratio of value to key heads (3), a chunk tail, two sequences.
    for c in [(2, 6, 64, 1), (1, 3, 70, 2)] {
        let (op, reference, _, _) = case(&mut backend, c, 0);
        let r = rel(&op, &reference);
        assert!(r < 1e-5, "{c:?}: {r:e}");
    }
}

#[test]
fn shapes_fit_prompt_groups_of_any_head_counts() {
    let ctx = Context::new(32, 32);
    let s = HEAD;
    let fit = |n_k: i64, n_v: i64, nt: i64| {
        let q = ctx.new_tensor(types::F32, &[s, n_k, nt, 1]);
        let v = ctx.new_tensor(types::F32, &[s, n_v, nt, 1]);
        let g = ctx.new_tensor(types::F32, &[1, n_v, nt, 1]);
        let state = ctx.new_tensor(types::F32, &[s, s, n_v, 1]);
        shapes_fit(q, q, v, g, g, state)
    };
    // A tensor parallel rank of Qwen3.8-Flash-Next (8 of 16 key, 24 of 48 value heads).
    assert!(fit(8, 24, 64));
    assert!(fit(8, 24, 4096));
    // One GPU's layer (ggml-cuda's own chunked kernel's head counts; `takes` decides).
    assert!(fit(16, 48, 4096));
    assert!(fit(16, 32, 4096));
    // Too few tokens (decode, verify), value heads not a multiple of the key heads.
    assert!(!fit(8, 24, 63));
    assert!(!fit(16, 48, 63));
    assert!(!fit(16, 24, 4096));
}

/// The head counts the op takes: a rank's share with `--gdn-mma`, ggml-cuda's own groups with
/// the staged kernel on compute capability 12.0 and 12.1 only. Restores the switches.
#[test]
fn takes_ggmls_groups_staged_on_consumer_blackwell_only() {
    let _lock = lock();
    assert!(staged_gpu(Some((12, 1))));
    assert!(staged_gpu(Some((12, 0))));
    for cc in [
        None,
        Some((8, 6)),
        Some((8, 9)),
        Some((9, 0)),
        Some((10, 0)),
        Some((12, 2)),
    ] {
        assert!(!staged_gpu(cc), "{cc:?}");
    }
    let (on, staged, gpu) = (
        ON.load(Ordering::Relaxed),
        STAGED.load(Ordering::Relaxed),
        STAGED_GPU.load(Ordering::Relaxed),
    );
    assert!(staged, "on by default");
    for (o, st, g) in [
        (false, true, true),
        (true, false, true),
        (true, true, false),
    ] {
        ON.store(o, Ordering::Relaxed);
        STAGED.store(st, Ordering::Relaxed);
        STAGED_GPU.store(g, Ordering::Relaxed);
        assert_eq!(takes(16, 48, None), st && g, "{:?}", (o, st, g));
        assert_eq!(takes(16, 64, None), st && g);
        assert_eq!(takes(8, 24, None), o);
    }
    ON.store(on, Ordering::Relaxed);
    STAGED.store(staged, Ordering::Relaxed);
    STAGED_GPU.store(gpu, Ordering::Relaxed);
}

/// `gdn_staged` copies 16-byte pieces: aligned pointers and strides of whole 4-float groups.
#[test]
fn staged_needs_aligned_rows() {
    let base = std::ptr::without_provenance::<f32>(0x1000);
    let args = |off: usize, sq1: i64| Args {
        q: base,
        k: base.wrapping_add(off),
        v: base.wrapping_add(2 * off),
        g: base,
        beta: base,
        state: base,
        dst: std::ptr::null_mut(),
        state_out: std::ptr::null_mut(),
        h: 48,
        h_k: 16,
        n_tokens: 4096,
        n_seqs: 1,
        rq3: 1,
        sq: [sq1, 10240, 10240 * 4096],
        sv: [128, 10240, 10240 * 4096],
        sb: [1, 48, 48 * 4096],
        scale: 1.0,
        z: base,
        z_ld: 6144,
        norm_w: base,
        gout: std::ptr::null_mut(),
        eps: 1e-6,
        silu: 1,
        head_v: 128,
        codes: std::ptr::null_mut(),
        scales: std::ptr::null_mut(),
        slots: std::ptr::null_mut(),
    };
    assert!(args(2048, 128).staged_aligned());
    assert!(!args(2049, 128).staged_aligned());
    assert!(!args(2048, 130).staged_aligned());
}

#[test]
fn shapes_fit_needs_value_heads_and_sequences() {
    let ctx = Context::new(32, 32);
    let s = HEAD;
    let fit = |n_k: i64, n_v: i64, ns: i64| {
        let q = ctx.new_tensor(types::F32, &[s, n_k, 64, ns.max(1)]);
        let v = ctx.new_tensor(types::F32, &[s, n_v, 64, ns]);
        let g = ctx.new_tensor(types::F32, &[1, n_v, 64, ns]);
        let state = ctx.new_tensor(types::F32, &[s, s, n_v, ns]);
        shapes_fit(q, q, v, g, g, state)
    };
    assert!(fit(8, 24, 2));
    assert!(!fit(8, 0, 1));
    assert!(!fit(8, 24, 0));
}

/// ggml's `select_gdn_mma_path` on NVIDIA GPUs: value slices when whole heads leave half the SMs
/// idle, one CTA an SM when the slices fit one wave.
#[test]
fn schedule_is_ggmls_rule() {
    // A two-Spark rank's 24 value heads on a GB10 (48 SMs): slices at two CTAs an SM.
    assert_eq!(Schedule::pick(24, 1, 48), Schedule::SlicesTwo);
    assert_eq!(Schedule::SlicesTwo.grid(24, 1), 96);
    // One Spark's 48 heads, two sequences of a rank: whole heads.
    assert_eq!(Schedule::pick(48, 1, 48), Schedule::Heads);
    assert_eq!(Schedule::pick(24, 2, 48), Schedule::Heads);
    assert_eq!(Schedule::Heads.grid(24, 2), 48);
    // 25 heads leave fewer than half of 48 SMs idle.
    assert_eq!(Schedule::pick(25, 1, 48), Schedule::Heads);
    // 12 heads: the slices (48 CTAs) fit one wave.
    assert_eq!(Schedule::pick(12, 1, 48), Schedule::Slices);
    assert_eq!(Schedule::Slices.grid(12, 1), 48);
    assert_eq!(Schedule::pick(13, 1, 48), Schedule::SlicesTwo);
}

/// The shared memory sizes `gdn_mma.cu` static_asserts for `shared<16, 32>` and
/// `shared<16, 128>`, and the argument layout it static_asserts.
#[test]
fn launch_sizes_match_the_kernels() {
    assert_eq!(Schedule::Slices.smem(), 38016);
    assert_eq!(Schedule::SlicesTwo.smem(), 38016);
    assert_eq!(Schedule::Heads.smem(), 93312);
    assert_eq!(Schedule::Staged.smem(), 85376);
    assert_eq!(Schedule::Staged.grid(48, 2), 96);
    // gdn_split's split_shared<64> and split_shared<128>; two CTAs a head for the pair.
    assert_eq!(Schedule::Pair.smem(), 48384);
    assert_eq!(Schedule::Wide.smem(), 60672);
    assert_eq!(Schedule::Lean.smem(), 60672);
    assert_eq!(Schedule::WideTwo.smem(), 85376);
    assert_eq!(Schedule::Pair.grid(48, 1), 96);
    assert_eq!(Schedule::Wide.grid(48, 1), 48);
    assert_eq!((Schedule::Wide.warps(), Schedule::Pair.warps()), (16, 8));
    // Two pair CTAs (and the driver's 1 KiB each) fit a GB10 SM's 100 KiB.
    assert!(2 * (Schedule::Pair.smem() + 1024) <= 100 * 1024);
    assert_eq!(std::mem::size_of::<Args>(), 256);
    assert_eq!(std::mem::offset_of!(Args, h), 64);
    assert_eq!(std::mem::offset_of!(Args, sq), 104);
    assert_eq!(std::mem::offset_of!(Args, scale), 176);
    assert_eq!(std::mem::offset_of!(Args, z), 184);
    assert_eq!(std::mem::offset_of!(Args, eps), 216);
    assert_eq!(std::mem::offset_of!(Args, head_v), 224);
    assert_eq!(std::mem::offset_of!(Args, codes), 232);
    assert_eq!(std::mem::offset_of!(Args, slots), 248);
    // The codes kernel: `codes_shared` after the two-buffer kernel's, under the 99 KiB a block
    // may opt in to on compute capability 12.x, one CTA (and the driver's 1 KiB) an SM.
    assert_eq!(CODES_SMEM, 100736);
    assert!(CODES_SMEM <= 99 * 1024 && CODES_SMEM + 1024 <= 100 * 1024);
    // gdn_ahead's `ahead_shared` (one q/k/g/beta buffer, two v and z buffers) and with codes.
    assert_eq!(AHEAD_SMEM, 85248);
    assert_eq!(AHEAD_CODES_SMEM, 100608);
    assert!(AHEAD_CODES_SMEM <= 99 * 1024);
    assert_eq!(
        (AHEAD_GN_KERNEL, AHEAD_CODES_KERNEL),
        (CODES_KERNEL + 1, CODES_KERNEL + 2)
    );
}

/// `gdn_ahead` on by default.
#[test]
fn ahead_is_on_by_default() {
    let _lock = lock();
    assert!(AHEAD.load(Ordering::Relaxed));
}

/// A codes node's packed activations and slots fit in the tokens' F32 rows the norm leaves
/// unwritten, the slots 16-byte aligned for the kernel's copies (and the scales 4-byte).
#[test]
fn codes_fit_in_the_tokens_rows() {
    for (h, nt) in [
        (48, 64),
        (48, 65),
        (48, 12288),
        (24, 777),
        (1, 64),
        (48, 1 << 20),
    ] {
        let (total, at, len) = codes_layout(h, nt);
        let k = (HEAD * h) as usize;
        assert_eq!(total, crate::model::fp8::dynamic_layout(k as i64, nt).1);
        assert_eq!(len, 4 * (h * nt) as usize);
        assert!(total <= at && at % 256 == 0 && at - total < 256, "{h} {nt}");
        assert!(at + len <= 4 * k * nt as usize, "{h} {nt}");
        assert_eq!((k * nt as usize) % 4, 0, "the scales after the codes");
        // A chunk's slots (16 rows of h words) in whole 16-byte pieces.
        assert_eq!((16 * 4 * h as usize) % 16, 0);
    }
    assert!(CODES_HEADS <= 48);
}

#[test]
fn fits_no_host_group() {
    let ctx = Context::new(8, 8);
    let s = HEAD;
    let q = ctx.new_tensor(types::F32, &[s, 8, 64, 1]);
    let v = ctx.new_tensor(types::F32, &[s, 24, 64, 1]);
    let g = ctx.new_tensor(types::F32, &[1, 24, 64, 1]);
    let state = ctx.new_tensor(types::F32, &[s, s, 24, 1]);
    let backend = Backend::cpu().unwrap();
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    assert!(shapes_fit(q, q, v, g, g, state));
    // A state in host memory (the CPU graph) keeps ggml's op whatever the switches.
    assert!(!fits(q, q, v, g, g, state));
}

/// The ported chunked kernel at a rank's shapes agrees with ggml's per-token kernel to rounding,
/// with the time of each at a 2-Spark rank's shape and prompt steps of 2304 and 4608 tokens.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU of compute capability 8.0+"]
fn cuda_gdn_mma_matches_ggml() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let _lock = lock();
    let mut gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    for c in [
        (2, 6, 64, 1),
        (1, 3, 70, 2),
        (8, 24, 4608, 1),
        (8, 24, 2304, 1),
    ] {
        let reps = if c.0 == 8 { 10 } else { 0 };
        let (op, reference, op_ms, ref_ms) = case(&mut gpu, c, reps);
        if reps > 0 {
            println!("gdn_mma {c:?}: {op_ms:.3} ms, ggml gated_delta_net {ref_ms:.3} ms");
        }
        let r = rel(&op, &reference);
        assert!(r < 1e-3, "{c:?}: {r:e}");
    }
}

/// The GPU the codes tests run on, with the switches they need: `--gdn-mma` on and the engine's
/// GPU device 0.
#[cfg(feature = "cuda")]
fn codes_gpu() -> Backend {
    use crate::ggml::{DeviceKind, device_by_kind};
    let gpu = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    set_gpu(crate::hw::per_device::cc(0));
    set_gdn_mma(true);
    gpu
}

/// [`force_schedule`]`(None)` when dropped: the schedule rule again, also after a failed case.
#[cfg(feature = "cuda")]
struct Unforce;

#[cfg(feature = "cuda")]
impl Drop for Unforce {
    fn drop(&mut self) {
        force_schedule(None);
    }
}

/// `--dense-fp8-prompt` (#1183): a [`gdn_mma_gn`] node of `n_k` key and `n_v` value heads and
/// `nt` tokens switched to codes ([`codes_on`]) against one switched to BF16 rows ([`norm_on`])
/// and `fp8_act::plain_bf16` of them, codes and scales bit for bit, on `gpu` under the switches
/// as they are. Returns what the codes node's launch ran ([`CODES_RAN`], a [`Ran`]).
#[cfg(feature = "cuda")]
fn codes_case(gpu: &mut Backend, (n_k, n_v, nt): (i64, i64, i64)) -> i8 {
    use crate::model::fp8_act;
    let s = HEAD;
    let row = 2 * s * n_k + s * n_v;
    let ctx = Context::new(64, 64);
    let conv = ctx.new_tensor(types::F32, &[row, nt, 1]).as_input();
    let g = ctx.new_tensor(types::F32, &[1, n_v, nt, 1]).as_input();
    let beta = ctx.new_tensor(types::F32, &[1, n_v, nt, 1]).as_input();
    let state = ctx.new_tensor(types::F32, &[s, s, n_v, 1]).as_input();
    let z = ctx.new_tensor(types::F32, &[s * n_v, nt]).as_input();
    let norm_w = ctx.new_tensor(types::F32, &[s]).as_input();
    let es = 4usize;
    let nb = [s as usize * es, row as usize * es, (row * nt) as usize * es];
    let q = ctx.view_4d(conv, [s, n_k, nt, 1], nb, 0);
    let k = ctx.view_4d(conv, [s, n_k, nt, 1], nb, (s * n_k) as usize * es);
    let v = ctx.view_4d(conv, [s, n_v, nt, 1], nb, (2 * s * n_k) as usize * es);
    let qkv = [q, k, v, g, beta];
    let (rows_node, ob) = gdn_mma_gn(&ctx, qkv, state, z, norm_w, 1e-6, true);
    norm_on(rows_node);
    let want = fp8_act::plain_bf16(&ctx, ob).as_output();
    let (codes_node, _) = gdn_mma_gn(&ctx, qkv, state, z, norm_w, 1e-6, true);
    let got = codes_on(&ctx, codes_node);
    codes_node.as_output();
    // SAFETY: graph nodes built above.
    unsafe {
        assert!(is_codes(codes_node.raw()) && !is_codes(rows_node.raw()));
    }
    let mut graph = ctx.new_graph(64);
    graph.expand(want);
    graph.expand(got);
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    let mut x = noise((row * nt) as usize, 1);
    for tok in x.chunks_mut(row as usize) {
        for head in tok[..(2 * s * n_k) as usize].chunks_mut(s as usize) {
            let n = head.iter().map(|v| v * v).sum::<f32>().sqrt();
            head.iter_mut().for_each(|v| *v /= n);
        }
    }
    conv.set_f32(&x);
    let gb = (n_v * nt) as usize;
    let map = |seed, f: fn(f32) -> f32| noise(gb, seed).into_iter().map(f).collect::<Vec<_>>();
    g.set_f32(&map(2, |v| 0.5 * (v - 1.0)));
    beta.set_f32(&map(3, |v| 0.5 * (v + 1.0)));
    let s0: Vec<f32> = noise((s * s * n_v) as usize, 4)
        .iter()
        .map(|v| 0.1 * v)
        .collect();
    state.set_f32(&s0);
    // A gate spread over a few heads' worth of scale, so tokens' amaxes come from any head.
    let zs: Vec<f32> = noise((s * n_v * nt) as usize, 5)
        .iter()
        .map(|v| 3.0 * v)
        .collect();
    z.set_f32(&zs);
    norm_w.set_f32(
        &noise(s as usize, 6)
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect::<Vec<_>>(),
    );
    CODES_RAN.store(-1, Ordering::Relaxed);
    gpu.compute(&graph).unwrap();
    let ran = CODES_RAN.load(Ordering::Relaxed);
    assert!(ran >= 0, "the codes node ran: {:?}", (n_k, n_v, nt));
    let (want, got) = (want.bytes(), got.bytes());
    assert_eq!(want.len(), got.len());
    let (kk, n) = ((s * n_v) as usize, nt as usize);
    let i = want.iter().zip(&got).position(|(a, b)| a != b);
    let at = i.map(|i| {
        if i < kk * n {
            format!("code of token {}, column {}", i / kk, i % kk)
        } else {
            format!("scale of token {}", (i - kk * n) / 4)
        }
    });
    assert_eq!(at, None, "{:?} ran {ran}", (n_k, n_v, nt));
    ran
}

/// [`codes_case`] at one GPU's heads of Qwen3.8-Flash-Next at token counts that are not a
/// multiple of the chunk, and fewer heads: [`CODES_KERNEL`] where it runs (whole heads on
/// [`Schedule::WideTwo`], compute capability 12.x with at least as many SMs as heads).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU of compute capability 8.0+"]
fn cuda_codes_match_norm_then_fp8_act() {
    let _lock = lock();
    let mut gpu = codes_gpu();
    for c in [
        (16, 48, 777),
        (16, 48, 4096 + 5),
        (8, 24, 300),
        (16, 48, 64),
    ] {
        codes_case(&mut gpu, c);
    }
}

/// [`codes_case`] down the fallbacks, each checked to be the path taken (#1183, the default on
/// a GPU with fewer SMs than heads): [`GN_KERNEL`] writing the BF16 rows, then `fp8_act`'s
/// kernel over them (more heads than [`CODES_HEADS`] on [`Schedule::WideTwo`], forced so on any
/// GPU); the plain kernel, `gated_norm_bf16_rows` and `fp8_act`'s kernel writing the codes over
/// the F32 rows the norm read ([`Schedule::Heads`] forced); and more heads than
/// [`CODES_HEADS`] under the schedule rule (either of the two). The slots' memset runs in each.
/// The force is reset after, also when a case fails.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU of compute capability 8.0+"]
fn cuda_codes_fallbacks_match_norm_then_fp8_act() {
    let _lock = lock();
    let mut gpu = codes_gpu();
    let _unforce = Unforce;
    for c in [(8, 56, 777), (8, 64, 4096 + 5)] {
        assert_ne!(codes_case(&mut gpu, c), Ran::Codes as i8, "{c:?}");
    }
    force_schedule(Some(Schedule::WideTwo));
    for c in [(8, 56, 777), (8, 64, 4096 + 5), (1, 49, 64)] {
        assert_eq!(codes_case(&mut gpu, c), Ran::Norm as i8, "{c:?}");
    }
    force_schedule(Some(Schedule::Heads));
    for c in [(8, 48, 777), (8, 48, 4096 + 5), (8, 24, 300), (8, 56, 64)] {
        assert_eq!(codes_case(&mut gpu, c), Ran::Plain as i8, "{c:?}");
    }
}

/// [`set_gdn_ahead`] back to what it was when dropped, also after a failed case.
#[cfg(feature = "cuda")]
struct KeepAhead(bool);

#[cfg(feature = "cuda")]
impl Drop for KeepAhead {
    fn drop(&mut self) {
        set_gdn_ahead(self.0);
    }
}

/// One run of [`ahead_case`]: the bytes the node wrote that a reader may read (the final states,
/// then the BF16 rows or the codes and scales; empty when timed), the table index of the kernel
/// that ran ([`LAST_KERNEL`]) and the milliseconds a run took (0 when not timed).
#[cfg(feature = "cuda")]
type AheadRun = (Vec<u8>, i8, f64);

/// A gated-norm node of `n_k` key and `n_v` value heads and `nt` tokens, codes_case's inputs (a
/// non-zero initial state), its norm switched to BF16 rows or (`codes`) FP8 codes, built
/// without the shape gate so any token count runs (also under [`MIN_TOKENS`]), on whole heads
/// ([`Schedule::WideTwo`] forced). Runs it once per entry of `plan` with [`set_gdn_ahead`] set
/// to it; with `reps` the entry's time is the mean of `reps` more runs, else its bytes are read.
/// Before each run that is read, the regions read are poisoned: 0xA5 before the first, the
/// complement of the run before's bytes after, so a store a kernel skips shows as a difference.
#[cfg(feature = "cuda")]
fn ahead_case(
    gpu: &mut Backend,
    (n_k, n_v, nt): (i64, i64, i64),
    codes: bool,
    plan: &[bool],
    reps: usize,
) -> Vec<AheadRun> {
    let s = HEAD;
    let row = 2 * s * n_k + s * n_v;
    let ctx = Context::new(64, 64);
    let conv = ctx.new_tensor(types::F32, &[row, nt, 1]).as_input();
    let g = ctx.new_tensor(types::F32, &[1, n_v, nt, 1]).as_input();
    let beta = ctx.new_tensor(types::F32, &[1, n_v, nt, 1]).as_input();
    let state = ctx.new_tensor(types::F32, &[s, s, n_v, 1]).as_input();
    let z = ctx.new_tensor(types::F32, &[s * n_v, nt]).as_input();
    let norm_w = ctx.new_tensor(types::F32, &[s]).as_input();
    let es = 4usize;
    let nb = [s as usize * es, row as usize * es, (row * nt) as usize * es];
    let q = ctx.view_4d(conv, [s, n_k, nt, 1], nb, 0);
    let k = ctx.view_4d(conv, [s, n_k, nt, 1], nb, (s * n_k) as usize * es);
    let v = ctx.view_4d(conv, [s, n_v, nt, 1], nb, (2 * s * n_k) as usize * es);
    // gdn_mma_gn's node, without its shape gate.
    let (rows, half) = (nt + s, (nt + 1) / 2);
    let node = ctx.external_with(
        &GDN_MMA,
        types::F32,
        [s * n_v, rows + half, 1, 1],
        &[q, k, v, g, beta, state, z, norm_w],
        &[0, 1, 1e-6f32.to_bits() as i32],
    );
    if codes {
        codes_on(&ctx, node);
    } else {
        norm_on(node);
    }
    node.as_output();
    let mut graph = ctx.new_graph(64);
    graph.expand(node);
    ctx.alloc_graph(&graph, ComputeAllocator::new(gpu.buffer_type()))
        .unwrap();
    let mut x = noise((row * nt) as usize, 1);
    for tok in x.chunks_mut(row as usize) {
        for head in tok[..(2 * s * n_k) as usize].chunks_mut(s as usize) {
            let n = head.iter().map(|v| v * v).sum::<f32>().sqrt();
            head.iter_mut().for_each(|v| *v /= n);
        }
    }
    conv.set_f32(&x);
    let gb = (n_v * nt) as usize;
    let map = |seed, f: fn(f32) -> f32| noise(gb, seed).into_iter().map(f).collect::<Vec<_>>();
    g.set_f32(&map(2, |v| 0.5 * (v - 1.0)));
    beta.set_f32(&map(3, |v| 0.5 * (v + 1.0)));
    let s0: Vec<f32> = noise((s * s * n_v) as usize, 4)
        .iter()
        .map(|v| 0.1 * v)
        .collect();
    state.set_f32(&s0);
    let zs: Vec<f32> = noise((s * n_v * nt) as usize, 5)
        .iter()
        .map(|v| 3.0 * v)
        .collect();
    z.set_f32(&zs);
    norm_w.set_f32(
        &noise(s as usize, 6)
            .iter()
            .map(|v| 1.0 + 0.5 * v)
            .collect::<Vec<_>>(),
    );
    force_schedule(Some(Schedule::WideTwo));
    let _unforce = Unforce;
    let width = (s * n_v) as usize * es;
    let states = (nt as usize * width, s as usize * width);
    let read = if codes {
        (0, codes_layout(n_v, nt).0)
    } else {
        (rows as usize * width, nt as usize * width / 2)
    };
    let mut before: Option<Vec<u8>> = None;
    plan.iter()
        .map(|&on| {
            set_gdn_ahead(on);
            if reps == 0 {
                // The final states are the node's rows, not the input state's tensor, which keeps
                // its values.
                let poison: Vec<u8> = match &before {
                    Some(b) => b.iter().map(|x| !x).collect(),
                    None => vec![0xA5; states.1 + read.1],
                };
                node.set_bytes_at(states.0, &poison[..states.1]);
                node.set_bytes_at(read.0, &poison[states.1..]);
            }
            LAST_KERNEL.store(-1, Ordering::Relaxed);
            gpu.compute(&graph).unwrap();
            let kernel = LAST_KERNEL.load(Ordering::Relaxed);
            if reps > 0 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    gpu.compute(&graph).unwrap();
                }
                let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
                return (Vec::new(), kernel, ms);
            }
            let mut bytes = node.bytes_at(states.0, states.1);
            bytes.extend(node.bytes_at(read.0, read.1));
            before = Some(bytes.clone());
            (bytes, kernel, 0.0)
        })
        .collect()
}

/// The table index of the kernel [`ahead_case`] must run: the codes or the rows kernel, as
/// `gdn_ahead` (`on`) or as `gdn_split`.
#[cfg(feature = "cuda")]
fn ahead_kernel(codes: bool, on: bool) -> i8 {
    (match (codes, on) {
        (true, true) => AHEAD_CODES_KERNEL,
        (true, false) => CODES_KERNEL,
        (false, true) => AHEAD_GN_KERNEL,
        (false, false) => GN_KERNEL,
    }) as i8
}

/// #1183: `gdn_ahead` (z rows one chunk ahead, the Gram under the projections) against
/// `gdn_split<128, 16, 1, 1, 2, true, CODES>` bit for bit, the final states and the BF16 rows
/// (or the FP8 codes and scales), at one GPU's heads of Qwen3.8-Flash-Next (16 key, 48 value)
/// and a prompt step's token counts: 12288, a step's tail (8067), eight chunks and a one-token
/// chunk (129) and one token. Needs a GPU with 48 SMs or more for the codes kernels (a GB10).
/// Restores the switch.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU of compute capability 8.0+ with 48 SMs"]
fn cuda_ahead_matches_split_bit_for_bit() {
    let _lock = lock();
    let mut gpu = codes_gpu();
    let _keep = KeepAhead(AHEAD.load(Ordering::Relaxed));
    for nt in [12288, 8067, 129, 1] {
        for codes in [false, true] {
            let runs = ahead_case(&mut gpu, (16, 48, nt), codes, &[false, true], 0);
            let (old, new) = (&runs[0], &runs[1]);
            assert_eq!(
                [old.1, new.1],
                [ahead_kernel(codes, false), ahead_kernel(codes, true)],
                "{nt} tokens, codes {codes}: the kernels"
            );
            assert_eq!(old.0.len(), new.0.len());
            let at = old.0.iter().zip(&new.0).position(|(a, b)| a != b);
            assert_eq!(at, None, "{nt} tokens, codes {codes}: the first byte apart");
        }
    }
}

/// #1183's microbench: `gdn_split` (old) against `gdn_ahead` (new) at 12288 tokens of one GPU's
/// heads of Qwen3.8-Flash-Next, with and without the codes, the order alternating each round
/// (old first on even rounds); prints each one's median and best mean of a round and the
/// speedup of the medians.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU of compute capability 8.0+ with 48 SMs; a timing"]
fn cuda_ahead_bench() {
    let _lock = lock();
    let mut gpu = codes_gpu();
    let _keep = KeepAhead(AHEAD.load(Ordering::Relaxed));
    let (rounds, reps) = (15, 20);
    for codes in [false, true] {
        let plan: Vec<bool> = (0..rounds).flat_map(|r| [r % 2 == 1, r % 2 == 0]).collect();
        let runs = ahead_case(&mut gpu, (16, 48, 12288), codes, &plan, reps);
        // Every timing is of the kernel it is labelled with (no fallback, say from codes to the
        // rows and fp8_act's pack).
        for (r, (&on, run)) in plan.iter().zip(&runs).enumerate() {
            assert_eq!(
                run.1,
                ahead_kernel(codes, on),
                "codes {codes}, ahead {on}, run {r}: the kernel"
            );
        }
        let stats = |on: bool| {
            let mut ms: Vec<f64> = plan
                .iter()
                .zip(&runs)
                .filter(|(p, _)| **p == on)
                .map(|(_, r)| r.2)
                .collect();
            ms.sort_by(f64::total_cmp);
            (ms[ms.len() / 2], ms[0])
        };
        let ((old_med, old_min), (new_med, new_min)) = (stats(false), stats(true));
        println!(
            "gdn 16x48 12288 tokens, codes {codes}: old median {old_med:.3} ms min {old_min:.3} \
             ms, new median {new_med:.3} ms min {new_min:.3} ms, speedup {:.3}x",
            old_med / new_med
        );
    }
}

#[test]
fn each_gpu_of_a_mixed_box_answers_for_itself() {
    use crate::hw::per_device::test_cc;
    // The GB10, the 4090, the A100 and the 5090 have the tensor cores; the V100 and an unknown
    // device do not, whatever the engine's GPU is.
    for engine in [false, true] {
        let picks: Vec<bool> = (0..6)
            .map(|d| gpu_for_by(Some(d), engine, test_cc))
            .collect();
        assert_eq!(
            picks,
            [true, true, false, true, true, false],
            "engine {engine}"
        );
        assert_eq!(gpu_for_by(None, engine, test_cc), engine);
    }
    assert!(gpu_takes(Some(8)) && gpu_takes(Some(12)) && !gpu_takes(Some(7)) && !gpu_takes(None));
}

/// The staged kernel takes ggml-cuda's groups on the GB10 and the 5090 of a mixed box only,
/// whatever the engine's GPU is; the engine's answer where the state names no device.
#[test]
fn staged_answers_per_device_of_a_mixed_box() {
    use crate::hw::per_device::test_cc;
    for engine in [false, true] {
        let picks: Vec<bool> = (0..6)
            .map(|d| staged_for_by(Some(d), engine, test_cc))
            .collect();
        assert_eq!(
            picks,
            [true, false, false, false, true, false],
            "engine {engine}"
        );
        assert_eq!(staged_for_by(None, engine, test_cc), engine);
    }
}
