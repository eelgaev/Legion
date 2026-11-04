//! The target's output head as a DFlash drafter borrows it, cut down for drafting:
//! `--spec-draft-vocab N` keeps its first `N` rows, `--spec-draft-lm-head q4` a 4-bit copy of
//! them.
//!
//! A draft only changes what the target verifies, never what it emits, so the head the drafter
//! scores its block with can be cheaper than the target's own. The Qwen vocabularies are
//! roughly ordered by frequency, so the first rows are a shortlist: a token outside it is never
//! drafted (MTP's `--spec-draft-vocab` relies on the same order). On a GB10 the head is the
//! biggest matrix of a DFlash 2 pass (248,320 rows): NInfer's DFlash 2 drafts from a
//! 131,072-row 4-bit shortlist head (neroued/ninfer).

use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::ggml::{BufferType, Context, types};
use crate::gguf::{FileRange, GgufFile, Recipe};
use crate::spec::DraftLmHead;
use crate::spec::q4_head::to_q4;

use super::model::floats;
use super::weights::Store;

/// The head's name in the drafter's store.
pub const HEAD: &str = "target.output.weight";
/// Its scale's (an NVFP4 head's per-tensor scale, an FP8 head's one or per row).
pub const SCALE: &str = "target.output.scale";

/// Loads the first `rows` rows (0: all) of the output head of the target GGUF at `path` (its
/// `output.weight`, or the token embeddings of a model with tied embeddings) as [`HEAD`], with
/// its scale's matching values as [`SCALE`]; with [`DraftLmHead::Q4`] as a 4-bit copy that has
/// the scale folded in.
pub fn load(
    path: &Path,
    rows: usize,
    kind: DraftLmHead,
    buft: BufferType,
) -> Result<Store, String> {
    let gguf = GgufFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let src = if gguf.tensor("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let info = gguf
        .tensor(src)
        .ok_or("the target model has no output head")?;
    let (k, m) = match info.shape[..] {
        [k, m] => (k as usize, m as usize),
        _ => {
            return Err(format!(
                "the target's `{src}` has shape {:?}, expected 2-D",
                info.shape
            ));
        }
    };
    let rows = if rows == 0 { m } else { rows.min(m) };
    let ty = info.ggml_type;
    let w = prefix(&gguf, src, crate::ggml::row_size(ty, k) * rows)?;
    let scale = match gguf.tensor("output.scale") {
        Some(s) if src == "output.weight" => {
            if s.ggml_type != types::F32 {
                return Err(format!(
                    "the target's `output.scale` is {}, expected F32",
                    crate::ggml::type_name(s.ggml_type)
                ));
            }
            let n = s.shape.iter().product::<u64>() as usize;
            let keep = match n {
                1 => 1,
                _ if n == m => rows,
                _ => {
                    return Err(format!(
                        "the target's `output.scale` has {n} values for a head of {m} rows"
                    ));
                }
            };
            Some(floats(
                types::F32,
                &prefix(&gguf, "output.scale", 4 * keep)?,
                keep,
            ))
        }
        _ => None,
    };
    let (ty, w, scale) = match kind {
        DraftLmHead::Model | DraftLmHead::Auto => (ty, w, scale),
        DraftLmHead::Q4 => {
            let (q4, bytes) = to_q4(ty, &w, scale.as_deref(), k)?;
            (q4, bytes, None)
        }
    };
    let ctx = Context::new(2, 0);
    let head = ctx.new_tensor(ty, &[k as i64, rows as i64]).named(HEAD);
    let scale_t = scale
        .as_ref()
        .map(|s| ctx.new_tensor(types::F32, &[s.len() as i64]).named(SCALE));
    ctx.alloc_tensors(buft).map_err(|e| e.to_string())?;
    head.set_bytes(&w);
    if let (Some(t), Some(s)) = (scale_t, &scale) {
        t.set_f32(s);
    }
    Ok(Store::from_context(ctx))
}

/// The first `n` bytes of the tensor `name` of `gguf`.
fn prefix(gguf: &GgufFile, name: &str, n: usize) -> Result<Vec<u8>, String> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| format!("the target has no `{name}`"))?;
    // A model synthesized from a Hugging Face checkpoint makes some tensors (repacked NVFP4,
    // widened scales, gathered rows) instead of pointing at a file range.
    let parts = match gguf.recipe(name) {
        Some(Recipe::Computed(f)) => {
            let mut bytes = f().map_err(|e| format!("making `{name}`: {e}"))?;
            if bytes.len() < n {
                return Err(format!("`{name}`: {} bytes made, {n} needed", bytes.len()));
            }
            bytes.truncate(n);
            return Ok(bytes);
        }
        Some(Recipe::Gather(parts)) => parts.clone(),
        None => vec![FileRange {
            file: info.file_index,
            offset: info.offset,
            len: info.size,
        }],
    };
    let shards: Vec<_> = gguf.shard_paths().collect();
    let mut out = vec![0u8; n];
    let mut done = 0;
    for p in parts {
        if done == n {
            break;
        }
        let len = (p.len as usize).min(n - done);
        let file = std::fs::File::open(shards[p.file])
            .map_err(|e| format!("{}: {e}", shards[p.file].display()))?;
        file.read_exact_at(&mut out[done..done + len], p.offset)
            .map_err(|e| format!("reading `{name}`: {e}"))?;
        done += len;
    }
    if done < n {
        return Err(format!("`{name}` has {done} bytes, {n} needed"));
    }
    Ok(out)
}
