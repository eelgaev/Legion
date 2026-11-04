//! Cache types: sizes, and a quantized cache written with `set_rows` and read back by the
//! gather path (`KvPool::read`) against ggml's own quantize / dequantize of the same rows.

use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context, types};
use crate::model::core::{Hparams, KV_BLOCK, KvPool};

#[test]
fn sizes_and_fit() {
    assert_eq!(KvType::F16.bytes(128, 256), 512);
    assert_eq!(KvType::Bf16.bytes(128, 256), 512);
    assert_eq!(KvType::Q8_0.bytes(128, 256), 8 * 34);
    assert_eq!(KvType::Q4_0.bytes(128, 256), 8 * 18);
    assert_eq!(KvType::Q5_1.bytes(128, 256), 8 * 24);
    assert_eq!(KvType::Iq4Nl.bytes(128, 256), 8 * 18);
    assert!(KvType::Q8_0.fits(128, 4096));
    assert!(!KvType::Q8_0.fits(48, 48));
    // TurboQuant: per head d × b / 8 bytes of indices and a 2-byte norm, the cell padded to 16.
    assert_eq!(KvType::Tq3.bytes(128, 256), 112);
    assert_eq!(KvType::Tq4.bytes(128, 256), 144);
    assert_eq!(KvType::Tq3.bytes(256, 2048), 8 * 98);
    assert_eq!(KvType::Tq4.bytes(512, 2048), 4 * 258 + 8);
    assert_eq!(KvType::Tq3.row_ne0(128, 256), 112);
    assert_eq!(KvType::Q8_0.row_ne0(128, 256), 256);
    assert!(KvType::Tq3.fits(128, 1024) && KvType::Tq4.fits(512, 2048));
    assert!(!KvType::Tq3.fits(96, 384) && !KvType::Tq4.fits(128, 192));
    let t = KvTypes {
        k: KvType::Q8_0,
        v: KvType::Q4_0,
    };
    assert_eq!(t.row_bytes(128, 256), 8 * 34 + 8 * 18);
    assert!(KvTypes::default().is_f16() && !t.is_f16());
    assert!(!t.has_tq());
    let tq = KvTypes {
        k: KvType::Q8_0,
        v: KvType::Tq3,
    };
    assert!(tq.has_tq() && !tq.is_f16());
    assert_eq!(tq.row_bytes(128, 256), 8 * 34 + 112);
}

/// One attention layer, 64-wide heads, 2 kv heads (rows of 128 values).
fn hparams() -> Hparams {
    Hparams {
        n_layer: 1,
        n_embd: 256,
        n_head: 4,
        n_head_kv: 2,
        head_dim: 64,
        n_rot: 64,
        rope_sections: [0; 4],
        rope_freq_base: 1e4,
        n_ctx_train: 4096,
        rms_eps: 1e-6,
        moe: false,
        n_expert: 0,
        n_expert_used: 0,
        full_attention_interval: 1,
        ssm_d_conv: 4,
        ssm_d_state: 16,
        ssm_n_group: 1,
        ssm_dt_rank: 1,
        ssm_d_inner: 16,
        n_vocab: 16,
        arch: crate::model::core::ArchRef::new(crate::model::qwen35::Qwen35),
    }
}

/// How [`round_trip`] reads the cache back.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Read {
    /// The block gather ([`KvPool::read`] with block indices).
    Blocks,
    /// The cell gather of tensor parallelism ([`KvPool::read`] with one index per cell).
    Cells,
    /// [`KvPool::read_cells`] (flash attention's layout: V not transposed).
    CellsFa,
}

/// Writes `n` positions of noise through `set_rows` into a pool of `types`, gathers them back
/// as the gather path does, and compares with quantize-then-dequantize of the same values.
fn round_trip(backend: &mut Backend, types: KvTypes, read: Read, exact: bool) {
    let hp = hparams();
    let n_blocks = 3;
    let kv = KvPool::new_ring(&hp, backend.buffer_type(), n_blocks, &[0], None, types).unwrap();
    let (d, n_kv_heads) = hp.kv_dims(0);
    let row = (d * n_kv_heads) as usize;
    let n = 2 * KV_BLOCK;
    let ctx = Context::new(64, 64);
    let k = ctx
        .new_tensor(types::F32, &[d, n_kv_heads, n as i64])
        .as_input();
    let v = ctx
        .new_tensor(types::F32, &[d, n_kv_heads, n as i64])
        .as_input();
    let cells = ctx.new_tensor(types::I32, &[n as i64]).as_input();
    let index_len = if read == Read::Blocks { 2 } else { n as i64 };
    let blocks = ctx.new_tensor(types::I32, &[index_len]).as_input();
    let mut graph = ctx.new_graph(64);
    kv.write(&ctx, &mut graph, 0, k, v, cells);
    let (kr, vt) = match read {
        Read::CellsFa => kv.read_cells(&ctx, 0, blocks, n as i64, 1),
        _ => kv.read(&ctx, 0, blocks, n as i64, 1),
    };
    let (kr, vt) = (
        ctx.cast(kr, types::F32).as_output(),
        ctx.cast(vt, types::F32).as_output(),
    );
    graph.expand(kr);
    graph.expand(vt);
    ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
        .unwrap();
    let noise = |seed: u64| -> Vec<f32> {
        let mut x = seed | 1;
        (0..n * row)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    };
    let (kd, vd) = (noise(3), noise(5));
    k.set_f32(&kd);
    v.set_f32(&vd);
    // Positions 0..n into blocks 2 then 0 (out of order).
    let table = [2usize, 0];
    let c: Vec<i32> = (0..n)
        .map(|p| (table[p / KV_BLOCK] * KV_BLOCK + p % KV_BLOCK) as i32)
        .collect();
    cells.set_i32(&c);
    if read == Read::Blocks {
        blocks.set_i32(&[2, 0]);
    } else {
        blocks.set_i32(&c);
    }
    backend.compute(&graph).unwrap();

    // The reference: ggml's quantize and dequantize of each row (TurboQuant: the Rust
    // reference's), then F16 as the gather does.
    let reference = |t: KvType, x: &[f32]| -> Vec<f32> {
        let mut out = match t.tq_bits() {
            Some(bits) => {
                let c = crate::model::core::turboquant::Layout {
                    d: d as usize,
                    n_kv: n_kv_heads as usize,
                    bits,
                };
                let mut cell = vec![0u8; c.bytes()];
                x.chunks(row)
                    .flat_map(|r| {
                        c.encode(0, r, &mut cell);
                        c.decode(0, &cell, false)
                    })
                    .collect()
            }
            None => {
                let q = crate::ggml::quantize(t.ggml(), x, row);
                crate::ggml::dequantize(t.ggml(), &q, x.len())
            }
        };
        let mut h = vec![0u16; out.len()];
        // SAFETY: equal lengths.
        unsafe {
            crate::ggml::ffi::ggml_fp32_to_fp16_row(out.as_ptr(), h.as_mut_ptr(), out.len() as i64);
            crate::ggml::ffi::ggml_fp16_to_fp32_row(h.as_ptr(), out.as_mut_ptr(), out.len() as i64);
        }
        out
    };
    // set_rows quantizes IQ4_NL with ggml's fast row quantizer, quantize() with an importance
    // search, so their blocks differ: IQ4_NL is checked against the values themselves, within
    // its step on [-1, 1) noise; every other type equals ggml's round trip exactly.
    // ggml's CUDA set_rows rounds a few values of the 4- and 5-bit types to the other side
    // of a step than its CPU quantizer (`exact` false): within one step there.
    // One quantization step of a block of [-1, 1) noise (its largest |value| over the levels
    // between the block's scale and zero), times 1.01 for the scale's own rounding.
    let step = |levels: f32| 1.01 / levels;
    let tol = |t: KvType| match t {
        KvType::Iq4Nl => 0.3,
        _ if exact => 0.0,
        KvType::F16 | KvType::Bf16 => 0.0,
        KvType::Q8_0 => step(127.0),
        KvType::Q4_0 => step(8.0),
        KvType::Q5_0 => step(16.0),
        // Unsigned 4 and 5 bits over the block's span (2 for [-1, 1)).
        KvType::Q4_1 => 2.0 * step(15.0),
        KvType::Q5_1 => 2.0 * step(31.0),
        // The CUDA kernels' norm and transform sum in another order: an index can land on the
        // other side of a boundary (a centroid gap of up to ~0.9 × n / sqrt(d), spread back
        // over the head by the inverse transform).
        KvType::Tq3 | KvType::Tq4 => 0.15,
    };
    let want = |t: KvType, x: &[f32]| {
        if t == KvType::Iq4Nl {
            x.to_vec()
        } else {
            reference(t, x)
        }
    };
    let (kw, vw) = (want(types.k, &kd), want(types.v, &vd));
    let kg = kr.to_f32();
    for (i, (a, b)) in kg.iter().zip(&kw).enumerate() {
        assert!(
            (a - b).abs() <= tol(types.k),
            "{types:?} {read:?}: K[{i}] {a} vs {b}"
        );
    }
    // V comes back transposed, [n_kv, d, heads], except from read_cells.
    let vg = vt.to_f32();
    for p in 0..n {
        for i in 0..row {
            let got = if read == Read::CellsFa {
                vg[p * row + i]
            } else {
                vg[i * n + p]
            };
            let (a, b) = (got, vw[p * row + i]);
            assert!(
                (a - b).abs() <= tol(types.v),
                "{types:?} {read:?}: V at {p}, {i}: {a} vs {b}"
            );
        }
    }
}

fn round_trips(backend: &mut Backend, exact: bool) {
    for (k, v) in [
        (KvType::F16, KvType::F16),
        (KvType::Q8_0, KvType::Q8_0),
        (KvType::Q8_0, KvType::Q4_0),
        (KvType::Q4_0, KvType::Q4_0),
        (KvType::Q4_1, KvType::Q5_0),
        (KvType::Q5_1, KvType::Iq4Nl),
        (KvType::Bf16, KvType::Bf16),
        (KvType::Tq3, KvType::Tq4),
        (KvType::Q8_0, KvType::Tq3),
    ] {
        for read in [Read::Blocks, Read::Cells, Read::CellsFa] {
            // ggml's CUDA get_rows reads IQ4_NL rows in 256-value super-blocks: a cell's row
            // here is 128 values (models' are whole heads, 256 or more per GPU).
            let iq4 = k == KvType::Iq4Nl || v == KvType::Iq4Nl;
            if !exact && iq4 && read != Read::Blocks {
                continue;
            }
            round_trip(backend, KvTypes { k, v }, read, exact);
        }
    }
}

#[test]
fn quantized_cache_round_trips_like_ggml() {
    let mut backend = Backend::cpu().unwrap();
    backend.set_threads(2);
    round_trips(&mut backend, true);
}

/// The same on CUDA: ggml's set_rows quantizes and get_rows dequantizes there (the gather
/// path of every GPU, and tensor parallelism's cell gather).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn cuda_quantized_cache_round_trips_like_ggml() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let mut backend = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU device")
        .init()
        .unwrap();
    round_trips(&mut backend, false);
}
