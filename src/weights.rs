//! Model weights: every tensor of a GGUF file in ggml backend memory.
//!
//! A placement function picks the buffer type for each tensor (device VRAM, pinned host memory
//! on a given NUMA node, plain host memory). Tensors are grouped per buffer type into one
//! [`Context`] each, allocated in one buffer, then filled from the file.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ggml::{BufferType, Context, GgmlError, Tensor, TensorSlot};
use crate::gguf::{GgufError, GgufFile, Recipe, TensorInfo};
use crate::load_io::{Dest, DeviceSink, Job, Options, read_all};

pub mod fp16;
pub use fp16::{Fp16Load, WeightsFp16};

/// A device tensor as a [`DeviceSink`] for the reader's worker threads; the job's bytes land
/// at `.1` + the job offset (non-zero for the parts of a gathered tensor).
struct DeviceTensor<'a>(Tensor<'a>, usize);

impl DeviceSink for DeviceTensor<'_> {
    fn write_at(&self, offset: usize, bytes: &[u8]) {
        self.0.set_bytes_at(self.1 + offset, bytes);
    }
}

// SAFETY: workers only call `ggml_backend_tensor_set` on this tensor (reading its metadata, never
// touching the Context's allocator state), for disjoint byte ranges, while the loader holds the
// owning Context borrowed for the whole scoped read.
unsafe impl Send for DeviceTensor<'_> {}
unsafe impl Sync for DeviceTensor<'_> {}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Ggml(#[from] GgmlError),
    #[error("tensor `{tensor}`: {reason}")]
    Compute { tensor: String, reason: String },
    #[error("{path}: reading tensor `{tensor}`: {source}")]
    Read {
        path: PathBuf,
        tensor: String,
        source: std::io::Error,
    },
}

/// One placement group: a context whose tensors all live in one buffer of `buft`.
struct Group {
    buft: BufferType,
    ctx: Context,
    bytes: u64,
}

/// Tensor `source` of `gguf` as a tensor named `name` (same type, shape and file range), when
/// `name` is not in the file, `source` is a plain range of it (not synthesized) and its matmuls
/// take no hint by name (scales, FP8 or activation-precision lists), which the copy would lose.
pub fn alias_info(gguf: &GgufFile, name: &str, source: &str) -> Option<TensorInfo> {
    if gguf.tensor(name).is_some()
        || gguf.recipe(source).is_some()
        || crate::model::core::has_precision_hint(gguf, source)
    {
        return None;
    }
    let mut info = gguf.tensor(source)?;
    info.name = name.to_string();
    Some(info)
}

/// How the tensors of a load reached memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadStats {
    /// Tensors pointed straight into the mapped file (no read, no copy).
    pub mapped: usize,
    /// Plain-CPU tensors that could not be mapped (misaligned) and were read instead.
    pub mmap_fallback: usize,
    /// Every other tensor, read into its buffer.
    pub copied: usize,
}

/// Threads computing tensors of a load at once (see `read_data`).
const COMPUTE_THREADS: usize = 6;

pub struct Weights {
    groups: Vec<Group>,
    stats: LoadStats,
    /// Tensor name -> index into `groups` and the tensor's place in that group's context. Graph
    /// builds look weights up by name thousands of times, and `Context::tensor` compares the
    /// name with every tensor of the context in turn (a few thousand on a big MoE): the slot
    /// skips that search. The contexts never change after the load.
    by_name: HashMap<String, (usize, TensorSlot)>,
}

impl Weights {
    /// Loads every tensor of `gguf`, placing each in the buffer type `place` returns.
    /// Reads with `--load-threads` workers (default 4, at most 8).
    pub fn load(
        gguf: &GgufFile,
        place: impl Fn(&TensorInfo) -> BufferType,
    ) -> Result<Self, LoadError> {
        Self::load_with(gguf, place, &Options::configured())
    }

    /// Like [`Weights::load`], with explicit reader options (thread count, NUMA topology).
    pub fn load_with(
        gguf: &GgufFile,
        place: impl Fn(&TensorInfo) -> BufferType,
        opts: &Options<'_>,
    ) -> Result<Self, LoadError> {
        Self::load_converted(gguf, place, opts, Fp16Load::default())
    }

    /// Like [`Weights::load_with`], with the tensors `fp16` picks converted to F16 as they load
    /// (see [`fp16`]).
    pub fn load_converted(
        gguf: &GgufFile,
        place: impl Fn(&TensorInfo) -> BufferType,
        opts: &Options<'_>,
        fp16: Fp16Load,
    ) -> Result<Self, LoadError> {
        let mut infos = gguf.tensors();
        if !opts.skip.is_empty() {
            infos.retain(|i| !opts.skip.contains(&i.name));
        }
        for &(name, source) in opts.aliases {
            if let Some(info) = alias_info(gguf, name, source) {
                infos.push(info);
            }
        }
        // Extra tensors with a recipe (column shards) join after the F16 conversion, which
        // reads plain ranges.
        let made = |name: &str| opts.extra_recipes.is_some_and(|r| r.contains_key(name));
        infos.extend(opts.extra.iter().filter(|i| !made(&i.name)).cloned());
        let converted = fp16::convert(gguf, &mut infos, fp16);
        infos.extend(opts.extra.iter().filter(|i| made(&i.name)).cloned());
        let recipe = |name: &str| {
            opts.extra_recipes
                .and_then(|r| r.get(name))
                .or_else(|| gguf.recipe(name))
        };
        let converted = converted.map_err(|source| LoadError::Read {
            path: gguf
                .shard_paths()
                .next()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
            tensor: "(weights-fp16)".into(),
            source,
        })?;

        // Group by buffer type, keeping file order inside each group.
        let mut order: Vec<BufferType> = Vec::new();
        let mut members: Vec<Vec<&TensorInfo>> = Vec::new();
        for info in &infos {
            let buft = place(info);
            let g = match order.iter().position(|b| *b == buft) {
                Some(g) => g,
                None => {
                    order.push(buft);
                    members.push(Vec::new());
                    order.len() - 1
                }
            };
            members[g].push(info);
        }

        let paths: Vec<PathBuf> = gguf.shard_paths().map(Path::to_path_buf).collect();
        let mut stats = LoadStats::default();
        let mut mapped_names = std::collections::HashSet::new();
        let mut groups = Vec::with_capacity(order.len());
        let mut by_name = HashMap::with_capacity(infos.len());
        for (g, (buft, infos)) in order.into_iter().zip(&members).enumerate() {
            let ctx = Context::new(infos.len(), 0);
            for info in infos {
                let shape: Vec<i64> = info.shape.iter().map(|&d| d as i64).collect();
                let t = ctx.new_tensor(info.ggml_type, &shape).named(&info.name);
                // A repeated name keeps the last group's first tensor (what a search found).
                if by_name.get(&info.name).is_none_or(|&(at, _)| at != g) {
                    by_name.insert(info.name.clone(), (g, t.slot()));
                }
            }
            // Plain CPU memory: map each shard file that holds some of these tensors and point
            // the tensors into it, no read. Whatever a mapping cannot take (misaligned offsets)
            // is allocated and read below.
            if opts.mmap && buft.is_plain_cpu() {
                let mut mapped_here = 0;
                for (file_index, path) in paths.iter().enumerate() {
                    // Recipe tensors (gathered or computed) are never one mappable range, nor
                    // are converted ones.
                    let in_file: Vec<&TensorInfo> = infos
                        .iter()
                        .copied()
                        .filter(|i| {
                            i.file_index == file_index
                                && recipe(&i.name).is_none()
                                && !converted.contains_key(&i.name)
                        })
                        .collect();
                    if in_file.is_empty() {
                        continue;
                    }
                    let file = File::open(path).map_err(|source| LoadError::Read {
                        path: path.clone(),
                        tensor: String::new(),
                        source,
                    })?;
                    let places: Vec<(&str, u64)> = in_file
                        .iter()
                        .map(|i| (i.name.as_str(), i.offset))
                        .collect();
                    // WILLNEED over the data region starts readahead before the first touch.
                    let lo = in_file.iter().map(|i| i.offset).min().unwrap_or(0);
                    let hi = in_file.iter().map(|i| i.offset + i.size).max().unwrap_or(0);
                    let _ = nix::fcntl::posix_fadvise(
                        std::os::fd::AsRawFd::as_raw_fd(&file),
                        lo as i64,
                        (hi - lo) as i64,
                        nix::fcntl::PosixFadviseAdvice::POSIX_FADV_WILLNEED,
                    );
                    let done = ctx.map_tensors_from_file(&file, &places)?;
                    mapped_here += done.len();
                    mapped_names.extend(done.iter().map(|&i| in_file[i].name.clone()));
                }
                stats.mapped += mapped_here;
                stats.mmap_fallback += infos.len() - mapped_here;
            }
            ctx.alloc_tensors(buft)?;
            let bytes = infos.iter().map(|i| i.size).sum();
            groups.push(Group { buft, ctx, bytes });
        }

        stats.copied = infos.len() - stats.mapped - stats.mmap_fallback;
        let weights = Self {
            groups,
            stats,
            by_name,
        };
        let to_read: Vec<TensorInfo> = infos
            .iter()
            .filter(|i| !mapped_names.contains(&i.name))
            .cloned()
            .collect();
        weights.read_data(gguf, &to_read, opts, converted)?;
        Ok(weights)
    }

    /// Fills every tensor from the file with the parallel reader in [`crate::load_io`]: host
    /// buffers are read into directly, device buffers through staging chunks. Jobs run in file
    /// order. A NUMA topology in `opts` only matters once placement can name a node per tensor.
    fn read_data(
        &self,
        gguf: &GgufFile,
        infos: &[TensorInfo],
        opts: &Options<'_>,
        mut converted: HashMap<String, Vec<u8>>,
    ) -> Result<(), LoadError> {
        let paths: Vec<&Path> = gguf.shard_paths().collect();
        let files = paths
            .iter()
            .map(|path| {
                File::open(path).map_err(|source| LoadError::Read {
                    path: path.to_path_buf(),
                    tensor: String::new(),
                    source,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut order: Vec<&TensorInfo> = infos.iter().collect();
        order.sort_by_key(|i| (i.file_index, i.offset));
        // Each job remembers the tensor it fills, for error messages.
        let mut owner: Vec<usize> = Vec::new();
        let mut jobs: Vec<Job<'_>> = Vec::new();
        let mut computed: Vec<(&TensorInfo, Tensor<'_>, crate::gguf::Compute)> = Vec::new();
        for (k, info) in order.iter().enumerate() {
            let mut t = self
                .get(&info.name)
                .expect("every tensor was created above");
            // A host arena on a NUMA node is filled by that node's CPUs.
            let node = crate::residency::arena_node(self.groups[self.by_name[&info.name].0].buft);
            assert_eq!(
                info.size as usize,
                t.nbytes(),
                "tensor `{}` size",
                info.name
            );
            if let Some(bytes) = converted.remove(&info.name) {
                computed.push((info, t, std::sync::Arc::new(move || Ok(bytes.clone()))));
                continue;
            }
            let recipe = opts
                .extra_recipes
                .and_then(|r| r.get(&info.name))
                .or_else(|| gguf.recipe(&info.name));
            let parts = match recipe {
                Some(Recipe::Computed(f)) => {
                    computed.push((info, t, f.clone()));
                    continue;
                }
                Some(Recipe::Gather(parts)) => parts.clone(),
                None => vec![crate::gguf::FileRange {
                    file: info.file_index,
                    offset: info.offset,
                    len: info.size,
                }],
            };
            assert_eq!(
                parts.iter().map(|p| p.len).sum::<u64>(),
                info.size,
                "tensor `{}`: gathered ranges must cover it exactly",
                info.name
            );
            if t.is_meta() {
                // A meta buffer takes a split tensor in whole slices only (a row, or the whole
                // matrix for one split by rows): read it whole, then set it in one call.
                let paths: Vec<std::path::PathBuf> =
                    paths.iter().map(|p| p.to_path_buf()).collect();
                let read = move || -> Result<Vec<u8>, String> {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut out = Vec::new();
                    for p in &parts {
                        let mut f = File::open(&paths[p.file]).map_err(|e| e.to_string())?;
                        f.seek(SeekFrom::Start(p.offset))
                            .map_err(|e| e.to_string())?;
                        let at = out.len();
                        out.resize(at + p.len as usize, 0);
                        f.read_exact(&mut out[at..]).map_err(|e| e.to_string())?;
                    }
                    Ok(out)
                };
                computed.push((info, t, std::sync::Arc::new(read)));
                continue;
            }
            // SAFETY: `self` is not shared yet (we are inside `load`) and every tensor
            // appears in exactly one job (or one set of disjoint part jobs), so each slice
            // has a single writer.
            match unsafe { t.host_bytes_mut() } {
                Some(mut bytes) => {
                    // An arena whose expert tensors span nodes (GLM's parallel decode): each
                    // node's bytes filled by its own CPUs (#720).
                    let pieces = crate::residency::arena_pieces(
                        self.groups[self.by_name[&info.name].0].buft,
                        t,
                    )
                    .filter(|p| p.len() > 1)
                    .unwrap_or_default();
                    let mut at = 0usize;
                    for p in &parts {
                        let len = p.len as usize;
                        for (r, node) in
                            crate::residency::split::cut_by_node(at..at + len, &pieces, node)
                        {
                            let (head, rest) = bytes.split_at_mut(r.len());
                            bytes = rest;
                            owner.push(k);
                            jobs.push(Job {
                                file: p.file,
                                offset: p.offset + (r.start - at) as u64,
                                len: r.len() as u64,
                                dest: Dest::Host(head),
                                node,
                            });
                        }
                        at += len;
                    }
                }
                None => {
                    let mut at = 0usize;
                    for p in &parts {
                        owner.push(k);
                        jobs.push(Job {
                            file: p.file,
                            offset: p.offset,
                            len: p.len,
                            dest: Dest::Device(Arc::new(DeviceTensor(t, at))),
                            node,
                        });
                        at += p.len as usize;
                    }
                }
            }
        }
        read_all(&files, jobs, opts).map_err(|e| {
            let info = order[owner[e.job]];
            LoadError::Read {
                path: paths[info.file_index].to_path_buf(),
                tensor: info.name.clone(),
                source: e.source,
            }
        })?;
        // Computed tensors (an HF checkpoint's repacked experts, say) on a few threads: each
        // is read and converted on its own, then set here (a bounded queue keeps a few
        // tensors' bytes in memory at once).
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, COMPUTE_THREADS);
        let fs: Vec<crate::gguf::Compute> = computed.iter().map(|c| c.2.clone()).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Result<Vec<u8>, String>)>(workers);
        std::thread::scope(|scope| -> Result<(), LoadError> {
            for _ in 0..workers.min(fs.len()) {
                let (tx, fs, next) = (tx.clone(), &fs, &next);
                scope.spawn(move || {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= fs.len() || tx.send((i, fs[i]())).is_err() {
                            return;
                        }
                    }
                });
            }
            drop(tx);
            for (i, bytes) in rx {
                let (info, t, _) = &mut computed[i];
                let bytes = bytes.map_err(|reason| LoadError::Compute {
                    tensor: info.name.clone(),
                    reason,
                })?;
                if bytes.len() != t.nbytes() {
                    return Err(LoadError::Compute {
                        tensor: info.name.clone(),
                        reason: format!(
                            "computed {} bytes, the tensor has {}",
                            bytes.len(),
                            t.nbytes()
                        ),
                    });
                }
                // SAFETY: as above; computed tensors are in no read job.
                match unsafe { t.host_bytes_mut() } {
                    Some(dst) => dst.copy_from_slice(&bytes),
                    None => t.set_bytes_at(0, &bytes),
                }
            }
            Ok(())
        })?;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Tensor<'_>> {
        self.by_name.get(name).map(|&g| self.at(g))
    }

    /// The tensor at `(group, slot)` of `by_name`.
    fn at(&self, (g, slot): (usize, TensorSlot)) -> Tensor<'_> {
        // SAFETY: the slot was taken from a tensor of group `g`'s context at the load, and that
        // context lives as long as self.
        unsafe { self.groups[g].ctx.tensor_at(slot) }
    }

    /// How many tensors were mapped, read after a failed mapping, or read.
    pub fn stats(&self) -> LoadStats {
        self.stats
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Every tensor.
    pub fn tensors(&self) -> impl Iterator<Item = Tensor<'_>> + '_ {
        self.by_name.values().map(|&g| self.at(g))
    }

    /// Every tensor's name and size in bytes, as loaded (a converted tensor at its new size).
    pub fn sizes(&self) -> impl Iterator<Item = (&str, u64)> + '_ {
        self.by_name
            .iter()
            .map(|(name, &g)| (name.as_str(), self.at(g).nbytes() as u64))
    }

    /// Bytes placed in each buffer type.
    pub fn placement(&self) -> Vec<(BufferType, u64)> {
        self.groups.iter().map(|g| (g.buft, g.bytes)).collect()
    }
}
