//! Synthetic GGUF models: metadata built in memory, tensor data read from other files.
//!
//! A Hugging Face safetensors checkpoint (see [`crate::hf`]) opens as a [`GgufFile`] whose
//! metadata is written into an empty `gguf_context` the way llama.cpp's converter would write
//! it, and whose tensors point into the safetensors shards. Most tensors are a plain byte range of
//! one shard ([`TensorInfo::file_index`] / [`TensorInfo::offset`] are real); the rest carry a
//! [`Recipe`] that [`crate::weights::Weights`] runs instead of reading the range.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::{GgufError, GgufFile, GgufWriter, TensorInfo};
use crate::safetensors::Checkpoint;

/// One contiguous byte range of one backing file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileRange {
    /// Index into [`GgufFile::shard_paths`].
    pub file: usize,
    pub offset: u64,
    pub len: u64,
}

/// Produces a tensor's bytes in memory, at load time.
pub type Compute = Arc<dyn Fn() -> Result<Vec<u8>, String> + Send + Sync>;

/// How a synthetic tensor's bytes are made when they are not one plain file range.
#[derive(Clone)]
pub enum Recipe {
    /// The tensor is these ranges back to back (row reorders, stacked MoE experts). Streams
    /// straight into the destination like a plain range.
    Gather(Vec<FileRange>),
    /// The tensor is computed in memory (dtype widening, `-exp(A_log)`, a column permutation).
    Computed(Compute),
}

impl std::fmt::Debug for Recipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Recipe::Gather(parts) => write!(f, "Gather({} ranges)", parts.len()),
            Recipe::Computed(_) => f.write_str("Computed"),
        }
    }
}

/// One ModelOpt NVFP4 expert matrix as the checkpoint ships it: its packed codes
/// (`rows` x `k/2` bytes), its E4M3 sub-block scales (`rows` x `k/16`) and its per-tensor
/// `weight_scale_2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nvfp4Source {
    pub weight: FileRange,
    pub scale: FileRange,
    pub scale2: f32,
}

/// A stacked NVFP4 expert tensor's sources, expert by expert (a recipe repacks them all at
/// load; the SSD expert tier, [`crate::residency::ssd_experts`], reads them one at a time).
#[derive(Debug, Clone, PartialEq)]
pub struct Nvfp4Stack {
    pub rows: usize,
    pub k: usize,
    pub experts: Vec<Nvfp4Source>,
}

/// The tensor side of a synthetic model.
pub(super) struct Synthetic {
    pub(super) files: Vec<PathBuf>,
    pub(super) tensors: Vec<TensorInfo>,
    pub(super) by_name: HashMap<String, usize>,
    pub(super) recipes: HashMap<String, Recipe>,
    /// Stacked NVFP4 expert tensors by name ([`GgufFile::nvfp4_stack`]).
    pub(super) stacks: HashMap<String, Arc<Nvfp4Stack>>,
    /// The safetensors checkpoint whose mappings the recipes read through, if any
    /// ([`GgufFile::release_page_cache`] unmaps the pages they touched).
    pub(super) checkpoint: Option<Arc<Checkpoint>>,
}

impl GgufFile {
    /// A model whose metadata is `meta` (a writer holding key/values only) and whose tensors are
    /// `tensors`, read from `files`. `path` names the model (e.g. the checkpoint directory).
    /// Tensors with a recipe ignore their `file_index`/`offset`.
    pub fn synthetic(
        meta: GgufWriter,
        path: PathBuf,
        files: Vec<PathBuf>,
        tensors: Vec<TensorInfo>,
        recipes: HashMap<String, Recipe>,
    ) -> Result<Self, GgufError> {
        let by_name: HashMap<String, usize> = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        if by_name.len() != tensors.len() {
            return Err(GgufError::Unsupported(
                path.display().to_string(),
                "duplicate tensor names",
            ));
        }
        for name in recipes.keys() {
            if !by_name.contains_key(name) {
                return Err(GgufError::Unsupported(
                    name.clone(),
                    "recipe for an unknown tensor",
                ));
            }
        }
        Ok(Self {
            ctx: meta.into_metadata(),
            path,
            siblings: Vec::new(),
            synth: Some(Box::new(Synthetic {
                files,
                tensors,
                by_name,
                recipes,
                stacks: HashMap::new(),
                checkpoint: None,
            })),
        })
    }

    /// Records the safetensors checkpoint this synthetic model's recipes read through.
    pub fn set_checkpoint(&mut self, checkpoint: Option<Arc<Checkpoint>>) {
        if let Some(s) = self.synth.as_mut() {
            s.checkpoint = checkpoint;
        }
    }

    /// The safetensors checkpoint behind this synthetic model, if any.
    pub fn checkpoint(&self) -> Option<Arc<Checkpoint>> {
        self.synth.as_ref()?.checkpoint.clone()
    }

    /// Drops this model's file pages from the page cache: unmaps the pages the checkpoint's
    /// mappings touched, then `POSIX_FADV_DONTNEED` on every backing file. Pages still mapped
    /// elsewhere (host tensors pointed into a file mapping) stay; anything read later comes
    /// from the file again, so the bytes are the same.
    pub fn release_page_cache(&self) {
        use std::os::fd::AsRawFd;
        if let Some(c) = self.synth.as_ref().and_then(|s| s.checkpoint.as_ref()) {
            c.release_pages();
        }
        for p in self.shard_paths() {
            if let Ok(f) = std::fs::File::open(p) {
                let _ = nix::fcntl::posix_fadvise(
                    f.as_raw_fd(),
                    0,
                    0,
                    nix::fcntl::PosixFadviseAdvice::POSIX_FADV_DONTNEED,
                );
            }
        }
    }

    /// How `name`'s bytes are made, when they are not the plain range in its [`TensorInfo`].
    pub fn recipe(&self, name: &str) -> Option<&Recipe> {
        self.synth.as_ref()?.recipes.get(name)
    }

    /// Records where each expert of stacked NVFP4 tensors comes from (by tensor name).
    pub fn set_nvfp4_stacks(&mut self, stacks: HashMap<String, Arc<Nvfp4Stack>>) {
        if let Some(s) = self.synth.as_mut() {
            s.stacks = stacks;
        }
    }

    /// Where the experts of the stacked NVFP4 tensor `name` come from, for a checkpoint that
    /// ships them one by one.
    pub fn nvfp4_stack(&self, name: &str) -> Option<Arc<Nvfp4Stack>> {
        self.synth.as_ref()?.stacks.get(name).cloned()
    }

    /// Whether this model was synthesized (e.g. from a safetensors checkpoint).
    pub fn is_synthetic(&self) -> bool {
        self.synth.is_some()
    }
}
