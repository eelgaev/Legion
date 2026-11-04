//! BF16 copies of the MoE routers whose F32 values are all exact in BF16 (#836), read by the
//! decode-sized router matmuls instead of the F32 weights: half the bytes, the same bits.
//!
//! The safetensors loader widens a BF16 checkpoint's router (`mlp.gate.weight`) to F32, the
//! type the graphs' router ops take (nvidia/Qwen3.8-Flash-Next-NVFP4: 49 routers of 512 x 2560,
//! 5.24 MB each as F32). A decode or verify step reads the whole router once per MoE layer:
//! ~250 MB per step of the 48 layers, ~1.4 ms at the ~180 GB/s those reads ran at on a GB10
//! (one sequence with MTP: a verify step per ~78 ms round). The ops that take
//! the copy widen each value back to its F32 bits (`bf16 << 16`) before they use it, so they
//! compute on the very words they would read from the F32 weight:
//! [`super::router_mv`] (1-3 tokens, FMAs on the F32 values) and [`crate::model::dense_mv`]'s
//! [`crate::model::dense_mv::mul_mat_wide`] (4-16 tokens, ggml's `mul_mat_f<float>` TF32
//! products, which take a BF16-exact value as it is). Prompt steps keep the F32 weight
//! (cuBLAS).
//!
//! A router gets a copy only if every one of its values is exact in BF16 (the copy is checked
//! value by value when it is made): a GGUF's F32 router that is not keeps the F32 reads. The
//! copies live next to their routers (same buffer type), made at load before the caches are
//! sized (or else the first time a step graph asks for one). On by default where an op takes them (CUDA, compute capability 9.0+);
//! `legion serve --no-router-bf16` keeps the F32 reads.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::ggml::{BufferType, Context, Tensor, ffi, types};

#[cfg(test)]
mod tests;

/// The suffix of a layer's router weight (`blk.{il}.ffn_gate_inp.weight`).
pub(super) const ROUTER: &str = ".ffn_gate_inp.weight";

/// Whether graphs built from now on read the routers' BF16 copies ([`set_router_bf16`]).
static ON: AtomicBool = AtomicBool::new(true);

/// Whether graphs built from now on read the MoE routers' BF16 copies where their values are
/// all exact in BF16 (on by default); off keeps the F32 reads (`legion serve --no-router-bf16`,
/// for comparisons).
pub fn set_router_bf16(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

/// See [`set_router_bf16`].
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// The BF16 bytes (little endian) of the F32 values `f32_bytes` (little endian), or `None` if a
/// value is not exact in BF16 (its low 16 bits are not all zero) or the bytes are not whole
/// values.
pub fn bf16_copy(f32_bytes: &[u8]) -> Option<Vec<u8>> {
    let (values, rest) = f32_bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(f32_bytes.len() / 2);
    for v in values {
        let bits = u32::from_le_bytes(*v);
        if bits & 0xffff != 0 {
            return None;
        }
        out.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
    }
    Some(out)
}

/// Whether router weight `w` may get a copy: F32, 2-D and contiguous, in a GPU's memory.
pub(super) fn candidate(w: Tensor<'_>) -> bool {
    w.ty() == types::F32
        && w.ne()[2] == 1
        && w.ne()[3] == 1
        && w.is_contiguous()
        && !w.is_host()
        && !w.is_meta()
}

/// The address of `t`'s bytes.
fn addr(t: Tensor<'_>) -> usize {
    // SAFETY: a live tensor; reads a field.
    unsafe { (*t.raw()).data as usize }
}

/// The routers' copies: one context per buffer type, and per copy the address of the F32
/// router it was made from.
pub(super) struct RouterBf16 {
    ctxs: Vec<Context>,
    src: HashMap<String, usize>,
}

impl RouterBf16 {
    /// Copies of the `routers` (name and F32 weight, see [`candidate`]) whose values are all
    /// exact in BF16; `None` if none. A router whose buffer type cannot be read, or whose copy
    /// cannot be allocated, keeps its F32 reads.
    pub(super) fn build<'w>(routers: impl Iterator<Item = (String, Tensor<'w>)>) -> Option<Self> {
        // Per buffer type: the routers' names, shapes and BF16 bytes.
        type Copies<'t> = Vec<(String, Tensor<'t>, Vec<u8>)>;
        let mut by_buft: Vec<(BufferType, Copies<'w>)> = Vec::new();
        for (name, w) in routers.filter(|(_, w)| candidate(*w)) {
            let Some(bytes) = bf16_copy(&w.bytes()) else {
                tracing::debug!(tensor = %name, "MoE router: not exact in BF16, read as F32");
                continue;
            };
            // SAFETY: an allocated weight's buffer; reads its type.
            let buft = unsafe {
                std::ptr::NonNull::new(ffi::ggml_backend_buffer_get_type((*w.raw()).buffer))
                    .map(BufferType)
            };
            let Some(buft) = buft else { continue };
            match by_buft.iter_mut().find(|(b, _)| b.0 == buft.0) {
                Some((_, v)) => v.push((name, w, bytes)),
                None => by_buft.push((buft, vec![(name, w, bytes)])),
            }
        }
        let mut ctxs = Vec::new();
        let mut src = HashMap::new();
        for (buft, routers) in by_buft {
            let ctx = Context::new(routers.len(), 0);
            for (name, w, _) in &routers {
                ctx.new_tensor(types::BF16, &[w.ne()[0], w.ne()[1]])
                    .named(name);
            }
            if let Err(e) = ctx.alloc_tensors(buft) {
                tracing::warn!("MoE routers' BF16 copies: {e}; read as F32");
                continue;
            }
            for (name, w, bytes) in &routers {
                ctx.tensor(name).expect("created above").set_bytes(bytes);
                src.insert(name.clone(), addr(*w));
            }
            // Graph builds look a copy up per layer.
            ctx.index_names();
            ctxs.push(ctx);
        }
        let n = src.len();
        if n == 0 {
            return None;
        }
        tracing::info!("MoE routers read as BF16 (exact): {n}");
        Some(Self { ctxs, src })
    }

    /// The copy of router `w` (named `name`), if it was made from that very tensor (a router
    /// replaced since keeps its own reads).
    pub(super) fn get(&self, name: &str, w: Tensor<'_>) -> Option<Tensor<'_>> {
        if self.src.get(name) != Some(&addr(w)) {
            return None;
        }
        self.ctxs.iter().find_map(|c| c.tensor(name))
    }

    /// The copies (for counting what graphs read of the weights).
    pub(super) fn tensors(&self) -> impl Iterator<Item = Tensor<'_>> + '_ {
        self.src
            .keys()
            .filter_map(|n| self.ctxs.iter().find_map(|c| c.tensor(n)))
    }
}
