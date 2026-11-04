use super::*;

fn params(pattern: &str, shared: usize) -> Gemma4Params {
    let swa: Vec<bool> = pattern.chars().map(|c| c == 's').collect();
    let n = swa.len();
    Gemma4Params {
        n_head_kv: swa.iter().map(|&s| if s { 8 } else { 2 }).collect(),
        swa,
        head_dim_full: 512,
        head_dim_swa: 256,
        n_rot_swa: 256,
        rope_base_swa: 10_000.0,
        window: 512,
        n_layer_kv: n - shared,
        n_ff: vec![1; n],
        softcap: 30.0,
        n_embd_per_layer: 0,
        assistant: None,
        compact: Vec::new(),
        compact_rot: Vec::new(),
    }
}

#[test]
fn shared_layers_read_the_last_cached_layer_of_their_kind() {
    // E2B's layout: 35 layers, every fifth global, the last 20 shared.
    let pattern: String = (0..35)
        .map(|il| if il % 5 == 4 { 'g' } else { 's' })
        .collect();
    let g = params(&pattern, 20);
    assert_eq!(g.n_layer_kv, 15);
    for il in 0..15 {
        assert_eq!(g.kv_source(il), il);
    }
    for il in 15..35 {
        let want = if g.swa[il] { 13 } else { 14 };
        assert_eq!(g.kv_source(il), want, "layer {il}");
        // The source has the same head size and kv heads as the layer that reads it.
        assert_eq!(g.kv_dims(il), g.kv_dims(want));
    }
}

#[test]
fn without_shared_kv_every_layer_reads_its_own() {
    let g = params("sssssg", 0);
    assert!((0..6).all(|il| g.kv_source(il) == il));
    assert_eq!(g.kv_dims(0), (256, 8));
    assert_eq!(g.kv_dims(5), (512, 2));
}

/// Gather attention reading the cache cell by cell (tensor parallelism's layout, see
/// `Model::set_cell_gather`; then flash attention) gives the block-by-block gather's logits up
/// to rounding: a prompt, then a few decode steps over several blocks.
#[test]
#[cfg_attr(
    target_arch = "aarch64",
    ignore = "ggml's CPU flash attention (F16) is outside this tolerance on aarch64, see #906"
)]
fn cell_gather_matches_block_gather() {
    use crate::ggml::Backend;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let run = |cells: bool| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let mut m = Model::load(&gguf, |_| b.buffer_type()).unwrap();
        m.set_cell_gather(cells);
        let mut st = m.new_state(b.buffer_type(), 256).unwrap();
        let mut alloc = None;
        let prompt: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 50).collect();
        let mut out = m
            .forward(&mut b, &mut alloc, &mut st, &prompt, false)
            .unwrap();
        for t in 0..30u32 {
            out.extend(
                m.forward(&mut b, &mut alloc, &mut st, &[t % 50], false)
                    .unwrap(),
            );
        }
        out
    };
    let (a, b) = (run(false), run(true));
    assert_eq!(a.len(), b.len());
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    // Flash attention (the cell path) rounds Q·K in F16 on the CPU: close, not equal.
    assert!(
        diff <= 1e-2 * scale.max(1.0),
        "max |diff| {diff} of {scale}"
    );
}

/// Tensor parallelism on the CPU: the tiny model over a meta device of two CPU "devices"
/// (the same one twice), split by the Megatron rule, against the plain CPU. Exercises the
/// split states and the all-reduces of every op in the graph.
#[test]
fn meta_device_matches_one_device() {
    use crate::ggml::{Backend, DeviceKind, device_by_kind};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let prompt: Vec<u32> = (0..40).map(|i| (i * 7 + 3) % 50).collect();
    let host = Backend::cpu().unwrap().buffer_type();
    let run = |mut b: Backend| {
        b.set_threads(2);
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        })
        .unwrap();
        m.set_cell_gather(true);
        let mut st = m.new_state(b.buffer_type(), 256).unwrap();
        let mut alloc = None;
        let mut out = m
            .forward(&mut b, &mut alloc, &mut st, &prompt, false)
            .unwrap();
        for t in 0..20u32 {
            out.extend(
                m.forward(&mut b, &mut alloc, &mut st, &[t % 50], false)
                    .unwrap(),
            );
        }
        out
    };
    let one = run(Backend::cpu().unwrap());
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let meta =
        crate::ggml::meta::meta_device(&[cpu, cpu], crate::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    let two = run(meta.init().unwrap());
    let scale = one.iter().fold(0f32, |m, v| m.max(v.abs()));
    let diff = one
        .iter()
        .zip(&two)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(
        diff <= 1e-2 * scale.max(1.0),
        "max |diff| {diff} of {scale}"
    );
}

/// The executor over a meta device of two CPU "devices" (tensor parallelism's runtime): a
/// prompt, decode steps, a second sequence's prompt, then decode of both, so kept graphs of
/// several shapes alternate (a meta backend drops a graph's per-device views when it computes
/// another). Against the same steps on one CPU.
#[test]
fn meta_executor_alternates_kept_graphs() {
    use crate::ggml::{Backend, DeviceKind, device_by_kind};
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    let run = |mut b: Backend, ring: bool| -> Vec<u32> {
        b.set_threads(2);
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        })
        .unwrap();
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let window = m.hp.arch.swa_window();
        let ring = (ring && window > 0)
            .then(|| crate::model::core::state::KvRing::for_window(window, 64, 2));
        let mut exec =
            Executor::new_staged_ring(m, vec![(b, 0..n_layer)], 16, 3, 0, ring, Default::default())
                .unwrap();
        exec.set_attention(Attention::Gather);
        let blocks: Vec<Vec<u32>> = vec![(0..8).collect(), (8..16).collect()];
        let prompts: Vec<Vec<u32>> = vec![
            (0..37).map(|i| (i * 7 + 3) % 50).collect(),
            (0..21).map(|i| (i * 5 + 1) % 50).collect(),
        ];
        let mut out = Vec::new();
        let mut n_past = [0usize; 2];
        let mut last = [0u32; 2];
        // One step of `seqs` (slot, tokens); returns each one's greedy next token.
        let mut step = |exec: &mut Executor, seqs: &[(usize, Vec<u32>)]| -> Vec<u32> {
            let groups: Vec<Group> = seqs
                .iter()
                .map(|(s, toks)| Group {
                    slot: *s,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: toks,
                        n_past: n_past[*s],
                        blocks: &blocks[*s],
                        logits: Logits::Greedy,
                        ring: *s,
                        media: None,
                    }],
                })
                .collect();
            let r = exec.step(&groups).unwrap();
            for (s, toks) in seqs {
                n_past[*s] += toks.len();
            }
            r.argmax
        };
        let r = step(&mut exec, &[(0, prompts[0].clone())]);
        last[0] = r[0];
        out.extend(r);
        for _ in 0..3 {
            let r = step(&mut exec, &[(0, vec![last[0]])]);
            last[0] = r[0];
            out.extend(r);
        }
        let r = step(&mut exec, &[(1, prompts[1].clone())]);
        last[1] = r[0];
        out.extend(r);
        for _ in 0..3 {
            let r = step(&mut exec, &[(0, vec![last[0]]), (1, vec![last[1]])]);
            last = [r[0], r[1]];
            out.extend(r);
        }
        out.extend(step(&mut exec, &[(0, vec![last[0]])]));
        out
    };
    let one = run(Backend::cpu().unwrap(), false);
    let one_ring = run(Backend::cpu().unwrap(), true);
    assert_eq!(one, one_ring, "rings change the tokens");
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let meta =
        crate::ggml::meta::meta_device(&[cpu, cpu], crate::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    let two = run(meta.init().unwrap(), true);
    assert_eq!(one, two);
}

/// The MTP assistant under tensor parallelism: target steps and assistant drafts alternate on
/// a meta device of two CPU "devices" (the assistant reads the target's split cache), against
/// one CPU: the same target tokens and drafts.
#[test]
fn meta_executor_runs_the_assistant() {
    use crate::ggml::{Backend, DeviceKind, device_by_kind};
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let (path, apath) = (dir.path().join("tiny.gguf"), dir.path().join("asst.gguf"));
    tiny::write(&path, 3).unwrap();
    tiny::write_assistant(&apath, 5).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let agguf = crate::gguf::GgufFile::open(&apath).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    let run = |mut b: Backend| -> Vec<u32> {
        b.set_threads(2);
        let place = |info: &crate::gguf::TensorInfo| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        };
        let mut m = Model::load(&gguf, place).unwrap();
        m.load_mtp(&agguf, place).unwrap();
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let n_embd = m.hp.n_embd as usize;
        let mut exec = Executor::new(m, b, 16, 2, 0).unwrap();
        exec.set_attention(Attention::Gather);
        exec.enable_mtp().unwrap();
        exec.capture_hidden(vec![n_layer - 1]);
        let blocks: Vec<u32> = (0..16).collect();
        fn seq<'a>(tokens: &'a [u32], n_past: usize, blocks: &'a [u32]) -> Group<'a> {
            Group {
                slot: 0,
                verify: false,
                seqs: vec![SeqInput {
                    tokens,
                    n_past,
                    blocks,
                    logits: Logits::Last,
                    ring: 0,
                    media: None,
                }],
            }
        }
        let mut out = Vec::new();
        let prompt: Vec<u32> = (0..30).map(|i| (i * 7 + 3) % 50).collect();
        let mut n_past = 0;
        let mut toks = prompt.clone();
        for _ in 0..4 {
            let (next, h) = {
                let r = exec.step(&[seq(&toks, n_past, &blocks)]).unwrap();
                (
                    r.argmax[0],
                    r.hidden[0][(toks.len() - 1) * n_embd..].to_vec(),
                )
            };
            n_past += toks.len();
            out.push(next);
            let fed = [next];
            let d = exec.mtp_step(&[seq(&fed, n_past, &blocks)], &h).unwrap();
            out.push(d.argmax[0]);
            toks = vec![next];
        }
        out
    };
    let one = run(Backend::cpu().unwrap());
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let meta =
        crate::ggml::meta::meta_device(&[cpu, cpu], crate::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    let two = run(meta.init().unwrap());
    assert_eq!(one, two);
}

/// Tensor parallelism sizes its KV cache next to measured compute buffers
/// (`fit_tp_cache`): the steps a server runs at the end of the measured context (a 4-sequence
/// decode, a prompt chunk, both batched together, the assistant's drafts) must stay within the
/// bytes measured for them, on a meta device of two CPU "devices".
#[test]
fn meta_steps_stay_within_measured_buffers() {
    use crate::ggml::{DeviceKind, device_by_kind};
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput, StepPart};
    let dir = tempfile::tempdir().unwrap();
    let (path, apath) = (dir.path().join("tiny.gguf"), dir.path().join("asst.gguf"));
    tiny::write(&path, 3).unwrap();
    tiny::write_assistant(&apath, 5).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let agguf = crate::gguf::GgufFile::open(&apath).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let meta =
        crate::ggml::meta::meta_device(&[cpu, cpu], crate::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    let mut b = meta.init().unwrap();
    b.set_threads(2);
    let host = crate::ggml::Backend::cpu().unwrap().buffer_type();
    let place = |info: &crate::gguf::TensorInfo| match Model::placement(&info.name, false) {
        crate::model::core::Placement::Host => host,
        crate::model::core::Placement::Device => b.buffer_type(),
    };
    let mut m = Model::load(&gguf, place).unwrap();
    m.load_mtp(&agguf, place).unwrap();
    m.set_cell_gather(true);
    let n_layer = m.hp.n_layer;
    let (ctx, chunk, n_seqs) = (192usize, 48usize, 4usize);
    let mut exec = Executor::new(m, b, 20, n_seqs + 1, 0).unwrap();
    exec.set_attention(Attention::Gather);
    exec.enable_mtp().unwrap();
    exec.capture_hidden(vec![n_layer - 1]);
    exec.set_max_seq_tokens(ctx);
    // What the fit divides the free memory by: the pool's own figure, as the sizing formula.
    assert_eq!(
        exec.kv_block_bytes(),
        crate::model::core::KvPool::block_bytes_paged(&hp, 0..n_layer, false, Default::default())
    );
    let past = ctx - 1;
    let prompt = StepPart {
        n_seqs: 1,
        n_tok: chunk,
        n_past: ctx - chunk,
        verify: false,
    };
    let decode = StepPart {
        n_seqs,
        n_tok: 1,
        n_past: past,
        verify: false,
    };
    let rest = StepPart {
        n_seqs: n_seqs - 1,
        ..decode
    };
    let p = exec.measure_groups(&[prompt]).unwrap()[0];
    let both = exec.measure_groups(&[prompt, rest]).unwrap()[0];
    let d = exec.measure_groups(&[decode]).unwrap()[0];
    let mtp = exec.measure_mtp(n_seqs, 1, past).unwrap();
    let budget = p.max(both) + d + mtp;
    assert!(d > 0 && p > 0 && mtp > 0);

    // The same steps for real (the cache's contents do not matter, only the graphs' sizes).
    let blocks: Vec<Vec<u32>> = (0..n_seqs as u32)
        .map(|s| (0..3).map(|b| s * 3 + b).collect())
        .collect();
    let chunk_toks: Vec<u32> = (0..chunk as u32).map(|i| i % 50).collect();
    let one = [7u32];
    let seq = |s: usize, tokens, n_past| SeqInput {
        tokens,
        n_past,
        blocks: &blocks[s],
        ring: 0,
        logits: Logits::Last,
        media: None,
    };
    let dec = |from: usize| Group {
        slot: from,
        verify: false,
        seqs: (from..n_seqs).map(|s| seq(s, &one[..], past)).collect(),
    };
    let pre = Group {
        slot: 0,
        verify: false,
        seqs: vec![seq(0, &chunk_toks[..], ctx - chunk)],
    };
    let mut peak = 0;
    for groups in [
        vec![dec(0)],
        vec![pre.clone()],
        vec![pre, dec(1)],
        vec![dec(0)],
    ] {
        exec.step(&groups).unwrap();
        peak = peak.max(exec.compute_bytes(0));
    }
    let h = vec![0.0f32; n_seqs * exec.mtp_width()];
    exec.mtp_step(&[dec(0)], &h).unwrap();
    peak = peak.max(exec.compute_bytes(0));
    // The prompt chunk batched with decodes takes more than the chunk alone: measuring only
    // the chunk would have come short.
    assert!(peak > p + d + mtp, "{peak} within {}", p + d + mtp);
    assert!(
        peak <= budget,
        "the steps took {peak} bytes, {budget} measured"
    );
}

/// The gather budget (`Executor::set_gather_budget`) runs a decode batch with one long
/// sequence as several steps, and the assistant's drafts as several passes: the same tokens,
/// residuals and drafts as one batch.
#[test]
fn gather_budget_splits_steps_without_changing_them() {
    use crate::ggml::Backend;
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let (path, apath) = (dir.path().join("tiny.gguf"), dir.path().join("asst.gguf"));
    tiny::write(&path, 3).unwrap();
    tiny::write_assistant(&apath, 5).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let agguf = crate::gguf::GgufFile::open(&apath).unwrap();
    let lens = [300usize, 20, 33];
    let blocks: Vec<Vec<u32>> = vec![(0..6).collect(), vec![6, 7], vec![8, 9]];
    let run = |budget: usize, meta: bool| -> (Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>) {
        let mut b = if meta {
            let cpu = crate::ggml::device_by_kind(crate::ggml::DeviceKind::Cpu).unwrap();
            let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
            crate::ggml::meta::meta_device(
                &[cpu, cpu],
                crate::model::core::tp::megatron_rule(&hp, 2),
            )
            .unwrap()
            .init()
            .unwrap()
        } else {
            Backend::cpu().unwrap()
        };
        b.set_threads(2);
        let host = Backend::cpu().unwrap().buffer_type();
        let place = |info: &crate::gguf::TensorInfo| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        };
        let mut m = Model::load(&gguf, place).unwrap();
        m.load_mtp(&agguf, place).unwrap();
        m.set_cell_gather(meta);
        let n_layer = m.hp.n_layer;
        let mut exec = Executor::new(m, b, 16, 4, 0).unwrap();
        exec.set_attention(Attention::Gather);
        exec.enable_mtp().unwrap();
        exec.capture_hidden(vec![n_layer - 1]);
        let seq = |s: usize, tokens, n_past| SeqInput {
            tokens,
            n_past,
            blocks: &blocks[s],
            ring: 0,
            logits: Logits::Last,
            media: None,
        };
        let prompts: Vec<Vec<u32>> = lens
            .iter()
            .enumerate()
            .map(|(s, &n)| (0..n).map(|i| ((i * 7 + s * 11 + 3) % 50) as u32).collect())
            .collect();
        let mut last = Vec::new();
        for (s, p) in prompts.iter().enumerate() {
            let r = exec
                .step(&[Group {
                    slot: s,
                    verify: false,
                    seqs: vec![seq(s, &p[..], 0)],
                }])
                .unwrap();
            last.push([r.argmax[0]]);
        }
        exec.set_gather_budget(budget);
        let all = Group {
            slot: 0,
            verify: false,
            seqs: (0..3).map(|s| seq(s, &last[s][..], lens[s])).collect(),
        };
        let r = exec.step(std::slice::from_ref(&all)).unwrap();
        let hidden = r.hidden[0].clone();
        let d = exec.mtp_step(std::slice::from_ref(&all), &hidden).unwrap();
        (r.argmax, hidden, d.argmax, d.pmax)
    };
    let whole = run(0, false);
    // One long sequence's span: the long one alone, the two short ones together.
    let budget = Executor::gather_span(lens[0] + 1);
    let split = run(budget, false);
    assert_eq!(whole.0, split.0);
    assert_eq!(whole.2, split.2);
    let close = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        let diff = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(diff <= 1e-4, "max |diff| {diff}");
    };
    close(&whole.1, &split.1);
    close(&whole.3, &split.3);
    // On a meta device (cell gather): split and whole alike.
    let meta_whole = run(0, true);
    let meta_split = run(budget, true);
    assert_eq!(meta_whole.0, meta_split.0);
    assert_eq!(meta_whole.2, meta_split.2);
    // Flash attention over F16 keys sums a shorter span in another order: closely.
    let (a, b) = (&meta_whole.1, &meta_split.1);
    let diff = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let scale = a.iter().map(|x| x.abs()).fold(0.0, f32::max);
    assert!(
        diff <= 1e-2 * scale.max(1.0),
        "max |diff| {diff} of {scale}"
    );
}

/// Gathering cells through rings, the sliding-window layers read only their window
/// (`swa_span`), not the whole span: over a 370-390-token context (a 512-key span, a
/// 256-key window gather) the same tokens and residuals as the cell gather without rings
/// (the whole span, masked). The rings hold 192 positions: decode positions 384-390 read
/// windows across a ring wrap.
#[test]
fn windowed_ring_gather_matches_whole_span() {
    use crate::ggml::Backend;
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let run = |windowed: bool| -> (Vec<u32>, Vec<f32>) {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let host = Backend::cpu().unwrap().buffer_type();
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        })
        .unwrap();
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let ring = windowed.then(|| crate::model::core::state::KvRing::for_window(8, 64, 1));
        let mut exec =
            Executor::new_staged_ring(m, vec![(b, 0..n_layer)], 8, 2, 0, ring, Default::default())
                .unwrap();
        exec.set_attention(Attention::Gather);
        exec.capture_hidden(vec![n_layer - 1]);
        let blocks: Vec<u32> = (0..8).collect();
        let prompt: Vec<u32> = (0..370).map(|i| ((i * 7 + 3) % 50) as u32).collect();
        let mut out = Vec::new();
        let mut hidden = Vec::new();
        let mut n_past = 0;
        let mut last = 0;
        let step = |exec: &mut Executor, toks: &[u32], n_past: usize| {
            let r = exec
                .step(&[Group {
                    slot: 0,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: toks,
                        n_past,
                        blocks: &blocks,
                        ring: 0,
                        logits: Logits::Greedy,
                        media: None,
                    }],
                }])
                .unwrap();
            (r.argmax[0], r.hidden[0].clone())
        };
        for chunk in prompt.chunks(64) {
            (last, _) = step(&mut exec, chunk, n_past);
            n_past += chunk.len();
        }
        for _ in 0..20 {
            let (t, h) = step(&mut exec, &[last], n_past);
            out.push(t);
            hidden.extend(h);
            n_past += 1;
            last = t;
        }
        (out, hidden)
    };
    let (whole, hb) = run(false);
    let (windowed, hw) = run(true);
    assert_eq!(whole, windowed);
    let diff = hb
        .iter()
        .zip(&hw)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let scale = hb.iter().map(|a| a.abs()).fold(0.0, f32::max);
    assert!(
        diff <= 1e-3 * scale.max(1.0),
        "max |diff| {diff} of {scale}"
    );
}

/// A meta backend drops the per-device tensors (inputs included) of kept graphs that did not
/// run lately: each kept graph is made ready before its inputs are set. Three sequences of
/// different spans decode in turn, so three kept decode graphs (and the assistant's)
/// alternate on a meta device of two CPU "devices": the same tokens and drafts as one CPU.
#[test]
fn meta_executor_readies_alternating_graphs() {
    use crate::ggml::{Backend, DeviceKind, device_by_kind};
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let (path, apath) = (dir.path().join("tiny.gguf"), dir.path().join("asst.gguf"));
    tiny::write(&path, 3).unwrap();
    tiny::write_assistant(&apath, 5).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let agguf = crate::gguf::GgufFile::open(&apath).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    // Spans of 256, 512 and 768 keys: three decode shapes.
    let lens = [20usize, 300, 600];
    let blocks: Vec<Vec<u32>> = vec![(0..1).collect(), (1..6).collect(), (6..16).collect()];
    let run = |mut b: Backend| -> Vec<u32> {
        b.set_threads(2);
        let place = |info: &crate::gguf::TensorInfo| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        };
        let mut m = Model::load(&gguf, place).unwrap();
        m.load_mtp(&agguf, place).unwrap();
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let n_embd = m.hp.n_embd as usize;
        let mut exec = Executor::new(m, b, 16, 3, 0).unwrap();
        exec.set_attention(Attention::Gather);
        exec.enable_mtp().unwrap();
        exec.capture_hidden(vec![n_layer - 1]);
        fn seq<'a>(s: usize, tokens: &'a [u32], n_past: usize, blocks: &'a [u32]) -> Group<'a> {
            Group {
                slot: s,
                verify: false,
                seqs: vec![SeqInput {
                    tokens,
                    n_past,
                    blocks,
                    ring: 0,
                    logits: Logits::Last,
                    media: None,
                }],
            }
        }
        let mut n_past = [0usize; 3];
        let mut last = [0u32; 3];
        for s in 0..3 {
            let p: Vec<u32> = (0..lens[s])
                .map(|i| ((i * 7 + s * 5 + 1) % 50) as u32)
                .collect();
            for (c, chunk) in p.chunks(128).enumerate() {
                let r = exec.step(&[seq(s, chunk, c * 128, &blocks[s])]).unwrap();
                last[s] = r.argmax[0];
            }
            n_past[s] = lens[s];
        }
        let mut out = Vec::new();
        for _ in 0..3 {
            for s in [0, 2, 1] {
                let fed = [last[s]];
                let r = exec.step(&[seq(s, &fed, n_past[s], &blocks[s])]).unwrap();
                let h = r.hidden[0][..n_embd].to_vec();
                n_past[s] += 1;
                last[s] = r.argmax[0];
                out.push(last[s]);
                let fed = [last[s]];
                let d = exec
                    .mtp_step(&[seq(s, &fed, n_past[s], &blocks[s])], &h)
                    .unwrap();
                out.push(d.argmax[0]);
            }
        }
        out
    };
    let one = run(Backend::cpu().unwrap());
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let meta =
        crate::ggml::meta::meta_device(&[cpu, cpu], crate::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    let two = run(meta.init().unwrap());
    assert_eq!(one, two);
}

/// `swa_span` sizing: a sliding layer gathering cells reads the window before a step's
/// tokens and the tokens themselves. Prompt chunks of 250 tokens make the window plus the
/// tokens cross the 256-key padding (a 512-key window gather inside a 768-key span), and a
/// group of two sequences in separate rings, at different depths, alternates 3-token and
/// 1-token steps. Against the whole-span cell gather (no rings): the same tokens, residuals
/// within rounding.
#[test]
#[cfg_attr(
    target_arch = "aarch64",
    ignore = "ggml's CPU flash attention (F16) rounds a near tie the other way on aarch64, see #906"
)]
fn windowed_gather_spans_long_chunks_and_two_rings() {
    use crate::ggml::Backend;
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let blocks: Vec<Vec<u32>> = vec![(0..16).collect(), (16..24).collect(), (24..32).collect()];
    let run = |windowed: bool| -> (Vec<u32>, Vec<f32>) {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        let host = Backend::cpu().unwrap().buffer_type();
        let mut m = Model::load(&gguf, |info| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        })
        .unwrap();
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let ring = windowed.then(|| crate::model::core::state::KvRing::for_window(8, 256, 3));
        let mut exec =
            Executor::new_staged_ring(m, vec![(b, 0..n_layer)], 32, 3, 0, ring, Default::default())
                .unwrap();
        exec.set_attention(Attention::Gather);
        exec.capture_hidden(vec![n_layer - 1]);
        let mut tokens = Vec::new();
        let mut hidden = Vec::new();
        let mut keep = |r: crate::model::core::StepOutput| {
            tokens.extend(&r.argmax);
            hidden.extend(&r.hidden[0]);
            r.argmax
        };
        // Sequence 0: an 800-token prompt in 250-token chunks (the last chunk of 50 lands
        // at 750..800: a 768/1024-key span, a 512-key window gather); every row scored.
        let prompt: Vec<u32> = (0..800).map(|i| ((i * 7 + 3) % 50) as u32).collect();
        for (c, chunk) in prompt.chunks(250).enumerate() {
            let r = exec
                .step(&[Group {
                    slot: 0,
                    verify: false,
                    seqs: vec![SeqInput {
                        tokens: chunk,
                        n_past: c * 250,
                        blocks: &blocks[0],
                        ring: 0,
                        logits: Logits::FirstGreedy(chunk.len()),
                        media: None,
                    }],
                }])
                .unwrap();
            keep(r);
        }
        // Sequences 1 and 2 at depths 370 and 130, in rings 1 and 2.
        let mut n_past = [0usize, 0];
        for (i, len) in [370usize, 130].into_iter().enumerate() {
            let p: Vec<u32> = (0..len)
                .map(|j| ((j * 5 + i * 3 + 1) % 50) as u32)
                .collect();
            for chunk in p.chunks(250) {
                let r = exec
                    .step(&[Group {
                        slot: 1 + i,
                        verify: false,
                        seqs: vec![SeqInput {
                            tokens: chunk,
                            n_past: n_past[i],
                            blocks: &blocks[1 + i],
                            ring: 1 + i,
                            logits: Logits::Greedy,
                            media: None,
                        }],
                    }])
                    .unwrap();
                keep(r);
                n_past[i] += chunk.len();
            }
        }
        // Both in one group: 3-token and 1-token steps in turn.
        for step in 0..6 {
            let n = if step % 2 == 0 { 3 } else { 1 };
            let fed: Vec<Vec<u32>> = (0..2)
                .map(|i| {
                    (0..n)
                        .map(|j| ((step * 11 + i * 7 + j) % 50) as u32)
                        .collect()
                })
                .collect();
            let r = exec
                .step(&[Group {
                    slot: 1,
                    verify: false,
                    seqs: (0..2)
                        .map(|i| SeqInput {
                            tokens: &fed[i],
                            n_past: n_past[i],
                            blocks: &blocks[1 + i],
                            ring: 1 + i,
                            logits: Logits::FirstGreedy(n),
                            media: None,
                        })
                        .collect(),
                }])
                .unwrap();
            keep(r);
            for p in &mut n_past {
                *p += n;
            }
        }
        (tokens, hidden)
    };
    let (whole, hw) = run(false);
    let (windowed, hn) = run(true);
    assert_eq!(whole, windowed);
    assert_eq!(hw.len(), hn.len());
    let diff = hw
        .iter()
        .zip(&hn)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    let scale = hw.iter().map(|a| a.abs()).fold(0.0, f32::max);
    // Flash attention sums a window that starts elsewhere in its tiles than the whole span
    // does, so the residuals differ by rounding: measured 0.014 of 4.09 on x86 and 0.043 on
    // aarch64 (other CPU kernels and tiling); aligning the window start to the span's tiles
    // makes both match. A lost key changes the tokens (asserted above) and moves a residual by
    // about its whole scale; the bound sits well under that.
    assert!(
        diff <= 3e-2 * scale.max(1.0),
        "max |diff| {diff} of {scale}"
    );
}

/// On a meta backend a kept graph is initialized again only where another graph's compute
/// may have dropped its per-device tensors. Target decode steps and the assistant's drafts
/// alternate, with separate compute buffers. A draft after a target step redoes nothing. A
/// target step after a draft redoes what lives in the buffers the draft ran over (its cache
/// writes and views of weights the head also reads) and what reads those, never more than the
/// whole graph. A target step right after another redoes nothing. The tokens match one CPU.
#[test]
fn meta_backend_reinitializes_only_what_was_dropped() {
    use crate::ggml::{Backend, DeviceKind, device_by_kind};
    use crate::model::core::{Attention, Executor, Group, Logits, SeqInput};
    let dir = tempfile::tempdir().unwrap();
    let (path, apath) = (dir.path().join("tiny.gguf"), dir.path().join("asst.gguf"));
    tiny::write(&path, 3).unwrap();
    tiny::write_assistant(&apath, 5).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let agguf = crate::gguf::GgufFile::open(&apath).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let host = Backend::cpu().unwrap().buffer_type();
    fn seq<'a>(tokens: &'a [u32], n_past: usize, blocks: &'a [u32]) -> Group<'a> {
        Group {
            slot: 0,
            verify: false,
            seqs: vec![SeqInput {
                tokens,
                n_past,
                blocks,
                logits: Logits::Last,
                ring: 0,
                media: None,
            }],
        }
    }
    // Per cycle after the first: the tensors redone for a draft, for the target step after
    // it, and for a second target step right after that one; and the target step's first
    // (whole) initialization.
    let run = |mut b: Backend| -> (Vec<u32>, Vec<[u64; 3]>, u64) {
        b.set_threads(2);
        let place = |info: &crate::gguf::TensorInfo| match Model::placement(&info.name, false) {
            crate::model::core::Placement::Host => host,
            crate::model::core::Placement::Device => b.buffer_type(),
        };
        let mut m = Model::load(&gguf, place).unwrap();
        m.load_mtp(&agguf, place).unwrap();
        m.set_cell_gather(true);
        let n_layer = m.hp.n_layer;
        let n_embd = m.hp.n_embd as usize;
        let mut exec = Executor::new(m, b, 16, 2, 0).unwrap();
        exec.set_attention(Attention::Gather);
        exec.enable_mtp().unwrap();
        exec.capture_hidden(vec![n_layer - 1]);
        let blocks: Vec<u32> = (0..16).collect();
        let prompt: Vec<u32> = (0..30).map(|i| (i * 7 + 3) % 50).collect();
        let r = exec.step(&[seq(&prompt, 0, &blocks)]).unwrap();
        let (mut tok, mut h) = (r.argmax[0], r.hidden[0][29 * n_embd..].to_vec());
        let mut n_past = 30;
        let (mut out, mut redone, mut whole) = (Vec::new(), Vec::new(), 0);
        let count = |exec: &Executor| exec.reinit_counts().1;
        for cycle in 0..4 {
            let fed = [tok];
            let c0 = count(&exec);
            let d = exec
                .mtp_step(&[seq(&fed, n_past + 1, &blocks)], &h)
                .unwrap();
            let c1 = count(&exec);
            let r = exec.step(&[seq(&fed, n_past, &blocks)]).unwrap();
            let c2 = count(&exec);
            n_past += 1;
            let again = [r.argmax[0]];
            let r2 = exec.step(&[seq(&again, n_past, &blocks)]).unwrap();
            let c3 = count(&exec);
            n_past += 1;
            out.extend([d.argmax[0], r.argmax[0], r2.argmax[0]]);
            if cycle == 0 {
                whole = c2 - c1;
            } else {
                redone.push([c1 - c0, c2 - c1, c3 - c2]);
            }
            tok = r2.argmax[0];
            h = r2.hidden[0].clone();
        }
        (out, redone, whole)
    };
    let (one, ..) = run(Backend::cpu().unwrap());
    let cpu = device_by_kind(DeviceKind::Cpu).unwrap();
    let meta =
        crate::ggml::meta::meta_device(&[cpu, cpu], crate::model::core::tp::megatron_rule(&hp, 2))
            .unwrap();
    let (two, redone, whole) = run(meta.init().unwrap());
    assert_eq!(one, two);
    assert!(whole > 0);
    for [draft, target, again] in redone {
        assert_eq!(
            draft, 0,
            "a draft after a target step redid {draft} tensors"
        );
        assert!(target > 0 && target <= whole, "{target} of {whole}");
        assert_eq!(
            again, 0,
            "a target step right after itself redid {again} tensors"
        );
    }
}

/// A spilled pool allocates the rings the assistant reads in every draft first (in VRAM): the
/// last sliding-window layer leads, the global layer the assistant also reads keeps its place
/// (it has no ring), and every other layer keeps its order.
#[test]
fn assistant_rings_are_allocated_first() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    tiny::write_without_ple(&path, 3).unwrap();
    let gguf = crate::gguf::GgufFile::open(&path).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&gguf).unwrap();
    let n = hp.n_layer;
    let hot = hp.arch.mtp_kv_layers(&hp);
    assert_eq!(hot, vec![hp.kv_source(n - 2), hp.kv_source(n - 1)]);
    let attn = KvPool::attention_layers(&hp, 0..n);
    let order = KvPool::hot_first(&hp, attn.clone(), &hot);
    let ring = hp.kv_source(n - 2);
    assert!(hp.sliding_layer(ring));
    assert_eq!(order[0], ring);
    let rest: Vec<usize> = attn.iter().copied().filter(|&il| il != ring).collect();
    assert_eq!(order[1..], rest[..]);
    // Nothing to put first: the order as it was.
    assert_eq!(KvPool::hot_first(&hp, attn.clone(), &[]), attn);
}
