//! Decode state: the attention layers' F16 KV cache, paged, and the DeltaNet layers' recurrent
//! state (conv history + delta-rule state, F32), in slots.
//!
//! - [`KvPool`] holds K and V of every attention layer in fixed-size blocks of [`KV_BLOCK`]
//!   positions. Which blocks a sequence owns is bookkeeping outside the graph
//!   (`crate::kv::BlockTable`, shared prefixes through `crate::kv::radix`); the graph gets block
//!   ids and cell ids as data: [`KvPool::write`] scatters this step's K/V to their cells with
//!   `set_rows`, [`KvPool::read`] gathers a sequence's blocks into a contiguous span with
//!   `get_rows` (bit-exact: the F16 rows are copied as I32) for the gather attention path;
//!   paged attention (`super::paged`) reads the blocks in place through [`KvPool::pools`].
//! - [`StatePool`] holds the recurrent state in slots of fixed size (no paging: it does not grow
//!   with the sequence). A batch of `n` sequences uses `n` consecutive slots, so the DeltaNet ops
//!   read and write them as one view with no gather; the scheduler keeps decoding sequences in
//!   slots `0..n` by moving a state when a sequence joins or leaves ([`StatePool::copy_slots`]).
//! - [`SeqState`] is one sequence with blocks `0..` and slot 0, for single-sequence use
//!   ([`super::Model::forward`]).

use crate::ggml::{
    Backend, BufferType, ComputeAllocator, Context, GgmlError, Graph, Tensor, types,
};

use super::kv_type::{KvType, KvTypes};
use super::{ATTN_KV_PAD, Hparams};

/// Positions per KV block. Attention spans are padded to [`ATTN_KV_PAD`], a multiple of it.
pub const KV_BLOCK: usize = 64;

/// The sliding-window layers' caches as rings (#297): such a layer only ever reads the last
/// `window` positions, so instead of every position of every sequence it keeps, per sequence,
/// `blocks` blocks reused cyclically, position `p` in ring entry `(p / KV_BLOCK) % blocks`.
/// A sequence owns ring `r` (`0..sets`): blocks `r × blocks ..` of those layers' pools.
///
/// A step writes its tokens' K/V before its attention reads, so a ring must hold the window
/// before the step's first token plus the step's own tokens, across block boundaries:
/// [`Self::for_window`]. The positions it overwrites are then older than any window still
/// read, and so are rejected speculative rows' (a step is at most `step` tokens).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvRing {
    /// Blocks per ring.
    pub blocks: usize,
    /// Rings (sequences at once).
    pub sets: usize,
}

impl KvRing {
    /// Rings of `sets` sequences for a sliding window of `window` positions and steps of at
    /// most `step` tokens per sequence.
    pub fn for_window(window: usize, step: usize, sets: usize) -> Self {
        Self {
            blocks: (window + step).div_ceil(KV_BLOCK) + 1,
            sets: sets.max(1),
        }
    }

    /// Ring entry `i`'s block of ring `r`.
    pub fn block(&self, r: usize, i: usize) -> usize {
        r * self.blocks + i % self.blocks
    }

    /// The cell of position `p` in ring `r`.
    pub fn cell(&self, r: usize, p: usize) -> usize {
        self.block(r, p / KV_BLOCK) * KV_BLOCK + p % KV_BLOCK
    }

    /// Positions per ring.
    pub fn positions(&self) -> usize {
        self.blocks * KV_BLOCK
    }
}

/// K and V of every attention layer, `n_blocks × KV_BLOCK` cells.
///
/// Both are stored per position, `[head_dim × n_head_kv, cells]` (cell = block × KV_BLOCK +
/// offset), so one block is one contiguous run of `KV_BLOCK` positions. The row width is per
/// layer ([`Hparams::kv_dims`]: Gemma 4's sliding-window and global layers differ).
///
/// Gemma 4's shared-KV layers have no cache of their own: they read the cache of an earlier
/// layer ([`Hparams::kv_source`]). That mapping is fixed when the pool is made, the source's
/// tensors are owned by the pool, and every view a step takes of them ([`Self::pools`],
/// [`Self::read`]) is made in that step's graph context and borrows the pool: a view never
/// outlives the graph it belongs to, and the memory behind it lives as long as the pool. The
/// source layer's writes come first in the graph (layers are built in order), so a shared
/// layer reads this step's K/V too. A pool must hold the sources of the layers it serves
/// (checked by [`Self::check_sources`]).
///
/// A layer whose attention reads no V ([`Hparams::kv_has_v`]: DeepSeek-V4.1's latents are K
/// and V at once) keeps K only: its `cache_v` is an alias of `cache_k` (in `views`), which
/// nothing reads or writes but code listing both (rings, slot views) still finds.
///
/// qwen4exp also caches, per attention layer and cell, the sparse-attention indexer's raw key
/// and (in the last cell of each pool) the pool's key, `[2 × indexer dim, cells + QSA_SCRATCH]`
/// F16 as in llama.cpp's indexer cache. The scratch rows past the cells take the padding
/// entries of a step's pool updates.
pub struct KvPool {
    /// The tiered caches again, in the alias meta buffer over the same memory
    /// ([`KvTier::alias`]); the MTP head's slot views read them. Dropped before `tiers`.
    alias: Option<Context>,
    /// The host tier's tensors (see [`KvTier`]): one context per K or V cache, each over a
    /// tiered range whose host part is a piece of `arena`. Dropped before the arena.
    tiers: Vec<Context>,
    /// The host tier's memory: pinned on the GPU's NUMA node. Only held (the tiered ranges
    /// map it).
    #[allow(dead_code)]
    arena: Option<Context>,
    ctx: Context,
    /// Blocks `0..hbm_blocks` are in VRAM; the rest in the host tier.
    hbm_blocks: usize,
    /// The host tier's staging slots (`stage_k` / `stage_v` in `ctx`), in blocks.
    stage_blocks: usize,
    n_blocks: usize,
    /// The sliding-window layers' rings, if they have them.
    ring: Option<KvRing>,
    /// Per layer index: whether its own cache is a ring.
    ring_layers: Vec<bool>,
    /// The K and V caches' element types.
    types: KvTypes,
    /// Per layer index: `(head_dim, head_dim × n_head_kv)` of the layers with a cache here.
    dims: Vec<Option<(i64, i64)>>,
    /// Per layer index: the layer whose cache it reads.
    source: Vec<usize>,
    host: bool,
    /// Indexer key size, 0 without sparse attention.
    idx_dim: i64,
    /// Blocks per sequence slot when each sequence's blocks are one contiguous range (see
    /// [`Self::slot_span`]); 0 otherwise.
    slot_blocks: usize,
    /// Per layer index: the rotated key dimensions of a compact cache (#551), 0 for a plain one.
    /// A compact layer's `cache_k` holds per cell and kv head one granule of `head_dim + r`
    /// values, `[K rotated (r) | V not rotated | V rotated (r)]`, in the V type, and one pad
    /// cell past the last; its `cache_v` is an alias of the same memory `r` values in (in
    /// `views`), so K reads `[K rotated | V not rotated]` and V reads `[V not rotated | V
    /// rotated]` with a head stride of `head_dim + r` (see [`crate::model::gemma4`]).
    compact: Vec<i64>,
    /// Per layer index: whether its own cache keeps K only (see [`KvPool`]).
    k_only: Vec<bool>,
    /// The compact layers' V aliases over `ctx` or `tiers`, for the paged kernels (none over a
    /// meta buffer: see [`Self::compact_views`]), and the K-only layers' V aliases.
    views: Context,
}

/// A KV pool's host tier (#352): blocks from `hbm_blocks` on live in pinned host memory on NUMA
/// node `node` (the one closest to CUDA device `device`), read over the link at its full rate
/// (68.5 GB/s per V100 on an AC922).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvTier {
    pub device: usize,
    pub node: u32,
    pub hbm_blocks: usize,
    /// VRAM staging slots, in blocks of the widest tiered layer's K and V, that prefill copies
    /// host blocks into before reading them (a piece of the step's host blocks at a time; see
    /// [`super::paged::Tier`]). 0: none, prefill reads the host blocks in place.
    pub stage_blocks: usize,
    /// A tensor-parallel cache's tier (#317): the meta buffer type whose per-GPU slices keep
    /// their first `hbm_blocks` blocks in VRAM and the rest on each GPU's host node
    /// ([`crate::residency::cuda::tp_tier_devices`]). `None`: one GPU's cache, tiered through
    /// a host arena.
    pub split: Option<SplitTier>,
    /// With `split` and sequence slots: an alias of its buffer type, whose buffers are the
    /// same memory (`crate::residency::cuda::tp_tier_devices`), for the MTP head's views.
    pub alias: Option<SplitTier>,
}

/// The buffer type of a tensor-parallel cache's host tier (see [`KvTier::split`]).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct SplitTier(pub BufferType);

impl std::fmt::Debug for SplitTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SplitTier({})", self.0.name())
    }
}

/// Views per cache a step may take of a tensor-parallel host tier's buffer (see
/// `KvPool::split_tier`): a multiple of 16.
const SPLIT_TIER_VIEWS: usize = 256;

/// Scratch rows of the indexer cache (see [`KvPool`]).
pub const QSA_SCRATCH: usize = 256;

/// A tiered cache tensor: name, type, values per cell and pad cells past the pool's.
type CacheTensor = (String, crate::ggml::ffi::ggml_type, i64, i64);

/// Cells past a compact cache's last (`row` values per cell, `r` rotated values per head) for
/// its V alias ([`KvPool`]): the alias ends `r` values past the cells, and CUDA zeroes a
/// placed quantized tensor's row padding after its end (under 512 values, ggml-cuda's
/// `MATRIX_ROW_PADDING`).
pub(crate) fn compact_pad_cells(row: i64, r: i64) -> i64 {
    ((r + 512 + row - 1) / row).max(1)
}

/// The offset to give a view `[d, h, n_kv]` (strides a granule and a cell) of a compact
/// cache's granules `pool` (`h` kv heads per cell) for V `rb` bytes into each granule, of the
/// cells from byte `first` (whole cells) on ([`KvPool::slot_span`]).
///
/// That is `first + rb`, except in a meta buffer split by heads (tensor parallelism):
/// `ggml_backend_meta_buffer_init_tensor_impl` (ggml-backend-meta.cpp) gives each device's
/// slice of a view the view's offset times the device's share of the view's split axis (the
/// heads, 1/n), unless the offset lies within one element of the source's split axis or within
/// a view dimension narrower than the split dimension's stride (here the first `d` values of
/// a granule, where they are narrower than it), where it keeps it as is. Each slice must start
/// `rb` past its first cell's granules (`first / n` into the slice): `first + n × rb` scaled,
/// or `rb` kept for the first cell (`first` = 0) where that is within the `d` values.
pub(crate) fn slot_v_offset(pool: Tensor<'_>, first: usize, rb: usize, d: i64, h: i64) -> usize {
    use crate::ggml::meta::Split;
    let Some((n, Split::Axis { axis: 0, .. })) = crate::ggml::meta::tensor_split(pool) else {
        return first + rb;
    };
    assert!(
        h % n as i64 == 0,
        "{h} kv heads do not split evenly over {n} devices"
    );
    // SAFETY: a pure lookup in ggml's type table.
    let nb0 = unsafe { crate::ggml::ffi::ggml_type_size(pool.ty()) };
    let granule = pool.nb()[1] / h as usize;
    let dim0 = d as usize * nb0;
    let kept = |x: usize| x <= nb0 || (dim0 < granule && x <= dim0);
    let scaled = first + n * rb;
    if !kept(scaled) {
        scaled
    } else {
        assert!(
            first == 0 && kept(rb),
            "no view offset reaches {rb} bytes into the granules from {first}"
        );
        rb
    }
}

impl KvPool {
    pub fn new(hp: &Hparams, buft: BufferType, n_blocks: usize) -> Result<Self, GgmlError> {
        let layers: Vec<usize> = (0..hp.n_layer).filter(|&il| hp.owns_kv(il)).collect();
        Self::new_layers(hp, buft, n_blocks, &layers)
    }

    /// A pool for the attention layers `layers` only (an MTP head's own layer).
    pub fn new_layers(
        hp: &Hparams,
        buft: BufferType,
        n_blocks: usize,
        layers: &[usize],
    ) -> Result<Self, GgmlError> {
        Self::new_ring(hp, buft, n_blocks, layers, None, KvTypes::default())
    }

    /// [`Self::new_layers`], the sliding-window layers among `layers` in rings `ring`.
    pub fn new_ring(
        hp: &Hparams,
        buft: BufferType,
        n_blocks: usize,
        layers: &[usize],
        ring: Option<KvRing>,
        types: KvTypes,
    ) -> Result<Self, GgmlError> {
        Self::new_tiered(hp, buft, n_blocks, layers, ring, types, None)
    }

    /// [`Self::new_ring`] with a host tier: blocks `tier.hbm_blocks..n_blocks` of every
    /// full-attention layer's K and V live in pinned host memory on the GPU's node, mapped right
    /// after the VRAM blocks in one virtual range, so every kernel and ggml op reads a block
    /// through the same pointer wherever it lives ([`crate::residency::cuda::tiered_rows`]).
    /// Rings and the indexer keys stay in VRAM. CUDA only.
    pub fn new_tiered(
        hp: &Hparams,
        buft: BufferType,
        n_blocks: usize,
        layers: &[usize],
        ring: Option<KvRing>,
        types: KvTypes,
        tier: Option<KvTier>,
    ) -> Result<Self, GgmlError> {
        let host_blocks = tier.map_or(0, |t| n_blocks.saturating_sub(t.hbm_blocks));
        // An alias needs the split buffer even with every block in VRAM.
        let tier = tier.filter(|t| host_blocks > 0 || t.alias.is_some());
        let ctx = Context::new(3 * layers.len().max(1), 0);
        let mut host_tensors: Vec<CacheTensor> = Vec::new();
        let cells = (n_blocks.max(1) * KV_BLOCK) as i64;
        let idx_dim = hp.arch.index_dim();
        let top = layers.iter().copied().max().unwrap_or(0).max(hp.n_layer) + 1;
        let mut dims = vec![None; top];
        let mut ring_layers = vec![false; top];
        let mut compact = vec![0; top];
        let mut k_only = vec![false; top];
        for &il in layers {
            let (d, n_kv) = hp.kv_dims(il);
            let r = hp.compact_kv(il);
            compact[il] = r;
            k_only[il] = r == 0 && !hp.kv_has_v(il);
            let row = (d + r) * n_kv;
            dims[il] = Some((d, row));
            let cells = match ring {
                Some(r) if hp.sliding_layer(il) => {
                    ring_layers[il] = true;
                    (r.sets * r.positions()) as i64
                }
                _ => cells,
            };
            let (du, ru) = (d as usize, row as usize);
            assert!(
                // A compact layer has V's type only.
                (r > 0 || types.k.fits(du, ru)) && types.v.fits(du, ru),
                "layer {il}: {row}-wide cache rows do not fit {types:?}"
            );
            assert!(
                r == 0
                    || (types.v.tq_bits().is_none()
                        && types.v.fits(du, r as usize)
                        && !ring_layers[il]),
                "layer {il}: a compact cache needs {r} rotated values to fit {types:?} (not TurboQuant), no ring"
            );
            // A compact layer: one granule tensor in the V type, pad cells for its V alias.
            let caches: &[(&str, KvType, i64)] = if r > 0 {
                &[("k", types.v, compact_pad_cells(row, r))]
            } else if k_only[il] {
                // No V: an alias of K below.
                &[("k", types.k, 0)]
            } else {
                &[("k", types.k, 0), ("v", types.v, 0)]
            };
            for &(kind, ty, pad) in caches {
                // Elements per cell, or a TurboQuant cell's bytes (its I8 tensor).
                let ne0 = ty.row_ne0(du, ru) as i64;
                let name = format!("cache_{kind}-{il}");
                if tier.is_some() && !ring_layers[il] {
                    // In the tiered ranges below (a sparse-attention layer's K and V too: its
                    // kernels read cells through the same pointers; its indexer keys stay in
                    // VRAM).
                    host_tensors.push((name, ty.ggml(), ne0, pad));
                } else {
                    ctx.new_tensor(ty.ggml(), &[ne0, cells + pad]).named(&name);
                    if r > 0 {
                        // A meta buffer keeps a graph's views of its tensors in a pool of 16
                        // per tensor (see `Self::split_tier`): sequence slots view a compact
                        // layer's one tensor twice (K and V), as often as a plain layer's two.
                        ctx.new_tensor(types::I8, &[0])
                            .named(&format!("view_room-{il}"));
                    }
                }
            }
            if hp.arch.indexed(il) {
                ctx.new_tensor(types::F16, &[2 * idx_dim, cells + QSA_SCRATCH as i64])
                    .named(&format!("cache_idx-{il}"));
            }
        }
        // The staging slots, shared by the tiered layers (they run one after another).
        // None for TurboQuant caches: their attention reads the host blocks in place.
        let stage_blocks = tier
            .filter(|_| !types.has_tq())
            .map_or(0, |t| t.stage_blocks.min(host_blocks));
        let widest = host_tensors.iter().map(|x| x.2).max().unwrap_or(0);
        if stage_blocks > 0 && widest > 0 {
            let cells = (stage_blocks * KV_BLOCK) as i64;
            if compact.iter().any(|&r| r > 0) {
                // Bytes: a compact layer's granules are in the V type (a staged compact layer
                // reads both from `stage_k`, `src/cuda/paged_attn/`).
                let bytes = Self::stage_cell_bytes(&dims, &ring_layers, types) as i64;
                for name in ["stage_k", "stage_v"] {
                    ctx.new_tensor(types::I8, &[bytes, cells]).named(name);
                }
            } else {
                ctx.new_tensor(types.k.ggml(), &[widest, cells])
                    .named("stage_k");
                ctx.new_tensor(types.v.ggml(), &[widest, cells])
                    .named("stage_v");
            }
        }
        // Nothing to allocate (every cache tiered): 0 bytes, no buffer.
        ctx.alloc_tensors(buft)?;
        let (tiers, arena) = match tier {
            Some(KvTier {
                split: Some(split), ..
            }) => (vec![Self::split_tier(split, &host_tensors, cells)?], None),
            Some(t) => Self::host_tier(t, host_blocks, &host_tensors)?,
            None => (Vec::new(), None),
        };
        let alias = match tier {
            Some(KvTier {
                split: Some(_),
                alias: Some(alias),
                ..
            }) => Some(Self::split_tier(alias, &host_tensors, cells)?),
            _ => None,
        };
        // The compact layers' V aliases (in their own context: allocating `ctx` or a tier's
        // would give them memory of their own).
        let find = |name: &str| {
            ctx.tensor(name)
                .or_else(|| tiers.iter().find_map(|c| c.tensor(name)))
        };
        let views = Self::compact_views(&compact, &k_only, &dims, cells, types, find);
        // Padding positions are masked, but still multiplied (by an exact 0 weight): every cell
        // must hold a finite number, never uninitialized NaNs.
        ctx.clear_buffers(0);
        for c in &tiers {
            c.clear_buffers(0);
        }
        // Graph builds look the caches up by name, per layer and sequence.
        for c in std::iter::once(&ctx)
            .chain(&tiers)
            .chain([&views])
            .chain(&alias)
        {
            c.index_names();
        }
        Ok(Self {
            alias,
            tiers,
            arena,
            ctx,
            hbm_blocks: tier.map_or(n_blocks.max(1), |t| t.hbm_blocks),
            stage_blocks: if widest > 0 { stage_blocks } else { 0 },
            n_blocks: n_blocks.max(1),
            ring,
            ring_layers,
            types,
            dims,
            source: (0..top).map(|il| hp.kv_source(il)).collect(),
            host: buft.is_host(),
            idx_dim,
            slot_blocks: 0,
            compact,
            k_only,
            views,
        })
    }

    /// The V aliases of the compact layers among `dims` (`compact` > 0): `cache_v-{il}` over
    /// `cache_k-{il}` (found by `find`) from its `r`-th value on, `cells` cells. And the
    /// K-only layers' (`k_only`): `cache_v-{il}` over the whole `cache_k-{il}`, as it is.
    fn compact_views<'a>(
        compact: &[i64],
        k_only: &[bool],
        dims: &[Option<(i64, i64)>],
        cells: i64,
        types: KvTypes,
        find: impl Fn(&str) -> Option<Tensor<'a>>,
    ) -> Context {
        let n_k_only = k_only.iter().filter(|&&k| k).count();
        let n = compact.iter().filter(|&&r| r > 0).count() + n_k_only;
        let views = Context::new(n.max(1), 0);
        for il in (0..k_only.len()).filter(|&il| k_only[il]) {
            let base = find(&format!("cache_k-{il}")).expect("a K-only layer's cache");
            assert!(
                !base.is_meta(),
                "layer {il}: a K-only cache over a meta buffer (tensor parallelism)"
            );
            let v = views
                .new_tensor(base.ty(), &base.shape())
                .named(&format!("cache_v-{il}"));
            // SAFETY: nothing reads or writes a K-only layer's V ([`Hparams::kv_has_v`]), and
            // the ring copies leave it out ([`Self::ring_tensors`]).
            unsafe { v.place_in(base, 0) };
        }
        for (il, &r) in compact.iter().enumerate().filter(|x| *x.1 > 0) {
            let Some(Some((_, row))) = dims.get(il) else {
                continue;
            };
            let base = find(&format!("cache_k-{il}")).expect("a compact layer's cache");
            if base.is_meta() {
                // Tensor parallelism: a meta buffer's tensors have no address of their own
                // (each device's slice has its own), and it keeps the slices of a tensor placed
                // in it with a graph's views, dropped when the next graph runs. Its readers
                // (sequence slots, gathers) view the granules per graph instead
                // ([`Self::slot_span`]).
                continue;
            }
            let v = views
                .new_tensor(types.v.ggml(), &[*row, cells])
                .named(&format!("cache_v-{il}"));
            // SAFETY: V reads the granules' last `head_dim` values in place; nothing writes
            // through the alias (`Self::write` writes whole granules into `cache_k`), and the
            // pad cells ([`compact_pad_cells`]) keep the last cell's values inside `base`, and
            // the quantized row padding CUDA zeroes past a placed tensor.
            unsafe { v.place_in(base, types.v.bytes(r as usize, r as usize)) };
        }
        views
    }

    /// Bytes per cell of a staging slot (the widest non-ring layer's K or V cell, a compact
    /// layer's granules in the V type).
    fn stage_cell_bytes(dims: &[Option<(i64, i64)>], ring: &[bool], types: KvTypes) -> usize {
        (dims.iter().enumerate())
            .filter(|&(il, _)| !ring[il])
            .filter_map(|(_, d)| d.map(|(d, row)| (d as usize, row as usize)))
            .map(|(d, row)| types.k.bytes(d, row).max(types.v.bytes(d, row)))
            .max()
            .unwrap_or(0)
    }

    /// Blocks per sequence slot (see [`Self::slot_span`]), 0 when sequences' blocks are
    /// anywhere.
    pub fn slot_blocks(&self) -> usize {
        self.slot_blocks
    }

    /// Makes the pool's blocks slots of `n` blocks each (0: none): sequence slot `s` owns
    /// blocks `s × n ..` in order ([`crate::kv::BlockPool::slotted`]), which attention reads in
    /// place ([`Self::slot_span`]).
    pub fn set_slot_blocks(&mut self, n: usize) {
        self.slot_blocks = n;
    }

    /// The first `n_kv` positions of sequence slot `slot` ([`Self::set_slot_blocks`]) in the
    /// cache layer `il` reads, in place: K and V views `[head_dim, n_head_kv, n_kv]` of its
    /// pools. A cache split by heads over a meta device splits them by heads too, so tensor
    /// parallelism reads a sequence's keys where they are (VRAM or host tier) with no gather.
    /// `alias`: through the alias buffer's tensors where there are some ([`KvTier::alias`]).
    pub fn slot_span<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        slot: usize,
        n_kv: i64,
        alias: bool,
    ) -> (Tensor<'a>, Tensor<'a>) {
        assert!(
            !self.types.has_tq(),
            "sequence slots over TurboQuant caches"
        );
        let (d, _) = self.dims(il);
        let h = self.n_head_kv(il);
        let src = self.source(il);
        let first = slot * self.slot_blocks * KV_BLOCK;
        let pool = |kind: &str| {
            let name = format!("cache_{kind}-{src}");
            match &self.alias {
                Some(a) if alias => a.tensor(&name),
                _ => None,
            }
            .unwrap_or_else(|| self.tensor(kind, src))
        };
        let view = |pool: Tensor<'a>, offset: usize| {
            let cell = pool.nb()[1];
            ctx.view_3d(pool, [d, h, n_kv], [cell / h as usize, cell], offset)
        };
        let r = self.compact(il);
        if r > 0 {
            // A compact cache (see `KvPool`): K and V both view its granules, V `r` values in.
            let g = pool("k");
            let first = first * g.nb()[1];
            let v = slot_v_offset(g, first, self.types.v.bytes(r as usize, r as usize), d, h);
            return (view(g, first), view(g, v));
        }
        let (k, v) = (pool("k"), pool("v"));
        (view(k, first * k.nb()[1]), view(v, first * v.nb()[1]))
    }

    /// Kv heads of the cache layer `il` reads.
    fn n_head_kv(&self, il: usize) -> i64 {
        let (d, row) = self.dims(il);
        row / (d + self.compact(il))
    }

    /// The rotated key dimensions of the cache layer `il` reads if it is compact (#551, see
    /// [`KvPool`]), else 0.
    pub fn compact(&self, il: usize) -> i64 {
        self.compact.get(self.source(il)).copied().unwrap_or(0)
    }

    /// Bytes per block over all layers (K and V, and the indexer keys).
    pub fn block_bytes(hp: &Hparams) -> usize {
        Self::block_bytes_layers(hp, 0..hp.n_layer)
    }

    /// The attention layers among `layers` that have a cache of their own.
    pub fn attention_layers(hp: &Hparams, layers: std::ops::Range<usize>) -> Vec<usize> {
        layers
            .filter(|&il| il < hp.n_layer && hp.owns_kv(il))
            .collect()
    }

    /// `layers` with the sliding-window layers among `hot` moved to the front, every other
    /// order kept. [`Self::new_tiered`] allocates the caches in this order, and a spilled pool's
    /// buffer is VRAM first, host memory past it: the hot rings (the ones an MTP head reads in
    /// every draft pass, [`super::Arch::mtp_kv_layers`]) then stay in VRAM. Only placement
    /// changes, never what a step computes.
    pub fn hot_first(hp: &Hparams, mut layers: Vec<usize>, hot: &[usize]) -> Vec<usize> {
        layers.sort_by_key(|il| !(hot.contains(il) && hp.sliding_layer(*il)));
        layers
    }

    /// Checks that every attention layer among `layers` reads a cache this pool holds (a
    /// shared-KV layer and its source must be in one pool, i.e. one pipeline stage).
    pub fn check_sources(
        &self,
        hp: &Hparams,
        layers: std::ops::Range<usize>,
    ) -> Result<(), String> {
        for il in layers.filter(|&il| il < hp.n_layer && hp.is_attention(il)) {
            let src = self.source(il);
            if self.dims.get(src).copied().flatten().is_none() {
                return Err(format!(
                    "layer {il} reads the KV cache of layer {src}, which is in another stage"
                ));
            }
        }
        Ok(())
    }

    /// The layer whose cache layer `il` reads.
    pub fn source(&self, il: usize) -> usize {
        self.source.get(il).copied().unwrap_or(il)
    }

    /// `(head_dim, head_dim × n_head_kv)` of the cache layer `il` reads.
    fn dims(&self, il: usize) -> (i64, i64) {
        let src = self.source(il);
        self.dims
            .get(src)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("layer {il} has no KV cache"))
    }

    /// The rings of the sliding-window layers, if they have them.
    pub fn ring(&self) -> Option<KvRing> {
        self.ring
    }

    /// The rings' layers' own K and V (K only for a K-only layer; empty without rings).
    fn ring_tensors(&self) -> Vec<Tensor<'_>> {
        (self.ring_layers.iter().enumerate())
            .filter(|&(il, &ring)| ring && self.source(il) == il)
            .flat_map(|(il, _)| {
                let v = (!self.k_only[il]).then(|| self.tensor("v", il));
                std::iter::once(self.tensor("k", il)).chain(v)
            })
            .collect()
    }

    /// `slots` copies of a ring of each ring tensor in `buft` (host memory), for
    /// [`Self::copy_ring`], and their bytes; `None` without rings (or slots).
    pub fn ring_copies(
        &self,
        buft: BufferType,
        slots: usize,
    ) -> Result<Option<(Context, usize)>, GgmlError> {
        let (Some(ring), tensors) = (self.ring, self.ring_tensors()) else {
            return Ok(None);
        };
        if tensors.is_empty() || slots == 0 {
            return Ok(None);
        }
        let ctx = Context::new(tensors.len(), 0);
        let mut bytes = 0;
        for t in &tensors {
            let shape = [t.shape()[0], (slots * ring.positions()) as i64];
            let c = ctx.new_tensor(t.ty(), &shape).named(&t.name());
            bytes += c.nbytes();
        }
        ctx.alloc_tensors(buft)?;
        Ok(Some((ctx, bytes)))
    }

    /// Bytes of one ring of every ring tensor (a slot of [`Self::ring_copies`]).
    pub fn ring_copy_bytes(&self) -> usize {
        let Some(ring) = self.ring else {
            return 0;
        };
        (self.ring_tensors().iter())
            .map(|t| t.nb()[1] * ring.positions())
            .sum()
    }

    /// Copies ring `r`'s positions `0..n` (all of them past the ring's length) to copy `slot`
    /// of `copies` ([`Self::ring_copies`], `save`), or back. Synchronous; the caller orders it
    /// against the steps that use the ring.
    pub fn copy_ring(&self, copies: &Context, r: usize, slot: usize, n: usize, save: bool) {
        let Some(ring) = self.ring else {
            return;
        };
        let positions = ring.positions();
        let cells = n.next_multiple_of(KV_BLOCK).min(positions);
        for t in self.ring_tensors() {
            let c = copies.tensor(&t.name()).expect("a copy per ring tensor");
            let row = t.nb()[1];
            let (at, to, len) = (r * positions * row, slot * positions * row, cells * row);
            if save {
                t.copy_bytes_to(at, c, to, len);
            } else {
                c.copy_bytes_to(to, t, at, len);
            }
        }
    }

    /// Whether the cache layer `il` reads is a ring.
    pub fn is_ring(&self, il: usize) -> bool {
        self.ring_layers
            .get(self.source(il))
            .copied()
            .unwrap_or(false)
    }

    /// The K and V caches' element types.
    pub fn types(&self) -> KvTypes {
        self.types
    }

    /// Bytes per block of this pool's layers that are not rings (K, V and indexer keys): what
    /// another block would take.
    pub fn bytes_per_block(&self) -> usize {
        self.dims
            .iter()
            .enumerate()
            .filter(|&(il, _)| !self.ring_layers[il])
            .filter_map(|(il, d)| d.map(|(_, row)| (il, row as usize)))
            .map(|(il, row)| {
                let idx = self
                    .ctx
                    .tensor(&format!("cache_idx-{il}"))
                    .map_or(0, |_| 2 * self.idx_dim as usize * 2);
                (self.cell_bytes(il, row) + idx) * KV_BLOCK
            })
            .sum()
    }

    /// Bytes per cell of each tensor [`Self::bytes_per_block`] counts (every non-ring layer's
    /// K, V and indexer keys), over all the devices a tensor-parallel cache splits it on.
    pub fn cell_tensor_bytes(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for (il, d) in self.dims.iter().enumerate() {
            let Some((_, row)) = d.filter(|_| !self.ring_layers[il]) else {
                continue;
            };
            let (dim, row) = (self.dims(il).0 as usize, row as usize);
            match self.compact[il] {
                0 if self.k_only[il] => out.push(self.types.k.bytes(dim, row)),
                0 => {
                    out.push(self.types.k.bytes(dim, row));
                    out.push(self.types.v.bytes(dim, row));
                }
                r => out.push(self.types.v.bytes(dim + r as usize, row)),
            }
            if self.ctx.tensor(&format!("cache_idx-{il}")).is_some() {
                out.push(2 * self.idx_dim as usize * 2);
            }
        }
        out
    }

    /// Bytes per cell of layer `il`'s own K and V (`row` values each, K only for a K-only
    /// layer, or a compact layer's granules).
    fn cell_bytes(&self, il: usize, row: usize) -> usize {
        let (d, _) = self.dims(il);
        match self.compact[il] {
            0 if self.k_only[il] => self.types.k.bytes(d as usize, row),
            0 => self.types.row_bytes(d as usize, row),
            r => self.types.v.bytes((d + r) as usize, row),
        }
    }

    /// Bytes per block of a pool over layer `il` alone, without rings, K and V in `types` (the
    /// indexer keys F16): [`Self::bytes_per_block`] of the MTP head's own pool (`il` =
    /// `n_layer`), which the counts over the model's layers leave out.
    pub fn layer_block_bytes(hp: &Hparams, il: usize, types: KvTypes) -> usize {
        let idx = if hp.arch.indexed(il) {
            2 * hp.arch.index_dim() as usize * 2
        } else {
            0
        };
        (hp.kv_cell_bytes(il, types) + idx) * KV_BLOCK
    }

    /// Bytes per position of the sliding-window layers among `layers` (what a ring of them
    /// takes per position), K and V in `types` (K only for a K-only layer).
    pub fn sliding_token_bytes(
        hp: &Hparams,
        layers: std::ops::Range<usize>,
        types: KvTypes,
    ) -> usize {
        Self::attention_layers(hp, layers)
            .iter()
            .filter(|&&il| hp.sliding_layer(il))
            .map(|&il| {
                let (d, n_kv) = hp.kv_dims(il);
                let (d, row) = (d as usize, (d * n_kv) as usize);
                if hp.kv_has_v(il) {
                    types.row_bytes(d, row)
                } else {
                    types.k.bytes(d, row)
                }
            })
            .sum()
    }

    /// Bytes per block over the layers `layers` that are not rings (all of them with
    /// `ring` false), K and V in `types` (the indexer keys F16).
    pub fn block_bytes_paged(
        hp: &Hparams,
        layers: std::ops::Range<usize>,
        ring: bool,
        types: KvTypes,
    ) -> usize {
        let f16 = KvTypes::default();
        let attn = Self::attention_layers(hp, layers.clone());
        let row_bytes = |t: KvTypes| -> usize {
            attn.iter()
                .filter(|&&il| !(ring && hp.sliding_layer(il)))
                .map(|&il| hp.kv_cell_bytes(il, t))
                .sum()
        };
        // The F16 total minus the F16 K/V rows, plus the rows in their types.
        let all = Self::block_bytes_layers(hp, layers.clone())
            - Self::sliding_token_bytes(hp, layers, f16) * KV_BLOCK * usize::from(ring);
        all - row_bytes(f16) * KV_BLOCK + row_bytes(types) * KV_BLOCK
    }

    /// Bytes per block over the layers `layers`.
    pub fn block_bytes_layers(hp: &Hparams, layers: std::ops::Range<usize>) -> usize {
        let attn = Self::attention_layers(hp, layers);
        let f16 = KvTypes::default();
        let cells: usize = attn.iter().map(|&il| hp.kv_cell_bytes(il, f16)).sum();
        let n_idx = attn.iter().filter(|&&il| hp.arch.indexed(il)).count();
        let idx_row = 2 * hp.arch.index_dim() as usize;
        (cells + n_idx * idx_row * 2) * KV_BLOCK
    }

    /// The indexer cache row of scratch entry `i` (see [`KvPool`]).
    pub fn scratch_cell(&self, i: usize) -> usize {
        assert!(i < QSA_SCRATCH);
        self.n_blocks * KV_BLOCK + i
    }

    /// Layer `il`'s indexer cache as (raw keys, pool keys), both `[indexer dim, rows]` views.
    pub fn idx_rows<'a>(&'a self, ctx: &'a Context, il: usize) -> (Tensor<'a>, Tensor<'a>) {
        let t = self.tensor("idx", il);
        let [w, rows, ..] = t.ne();
        let d = w / 2;
        let nb1 = t.nb()[1];
        let half = d as usize * t.element_size();
        (
            ctx.view_2d(t, [d, rows], nb1, 0),
            ctx.view_2d(t, [d, rows], nb1, half),
        )
    }

    /// Layer `il`'s own K and V caches (`[head_dim × n_head_kv, cells]`) for a kernel that
    /// writes them directly: `None` in host memory, with a host tier (its tiered ranges), or
    /// for a layer that reads another's cache.
    pub fn direct_caches(&self, il: usize) -> Option<(Tensor<'_>, Tensor<'_>)> {
        let plain =
            !self.host && self.tiers.is_empty() && self.source(il) == il && self.compact(il) == 0;
        plain.then(|| (self.tensor("k", il), self.tensor("v", il)))
    }

    /// Layer `il`'s indexer cache (`[2 × indexer dim, rows]` F16) for a kernel that writes it
    /// directly: `None` in host memory or with a host tier.
    pub fn direct_idx(&self, il: usize) -> Option<Tensor<'_>> {
        (!self.host && self.tiers.is_empty()).then(|| self.tensor("idx", il))
    }

    /// Writes `rows` (F32 `[2 × indexer dim, n]`: raw key | zeros) to cells `cells` of layer
    /// `il`'s indexer cache.
    pub fn idx_write<'a>(
        &'a self,
        ctx: &'a Context,
        graph: &mut Graph<'a>,
        il: usize,
        rows: Tensor<'a>,
        cells: Tensor<'a>,
    ) {
        debug_assert_eq!(rows.ne()[0], 2 * self.idx_dim);
        graph.expand(ctx.set_rows(self.tensor("idx", il), rows, cells));
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Whether the pool lives in host memory.
    pub fn is_host(&self) -> bool {
        self.host
    }

    /// K and V that layer `il` reads (its own, or its source's), the whole pool: F16
    /// `[head_dim × n_head_kv, cells]`.
    pub fn pools(&self, il: usize) -> (Tensor<'_>, Tensor<'_>) {
        let src = self.source(il);
        (self.tensor("k", src), self.tensor("v", src))
    }

    /// Layer `il`'s indexer cache (`[2 × index_dim, cells + scratch]`).
    pub fn idx_cache(&self, il: usize) -> Tensor<'_> {
        self.tensor("idx", il)
    }

    /// The tensors of this pool's own memory whose rows are its cells, block by block, from
    /// row 0 (pad or scratch rows past the cells): every non-ring layer's K, V and indexer
    /// cache (not an alias), none with a host tier. A `--kv-vmm` lend (#814) gives back the
    /// memory of their rows past a block.
    pub fn block_tensors(&self) -> Vec<Tensor<'_>> {
        if !self.tiers.is_empty() {
            return Vec::new();
        }
        let own =
            (0..self.dims.len()).filter(|&il| self.dims[il].is_some() && !self.ring_layers[il]);
        own.flat_map(|il| {
            ["k", "v", "idx"]
                .into_iter()
                .filter_map(move |kind| self.ctx.tensor(&format!("cache_{kind}-{il}")))
        })
        .collect()
    }

    fn tensor(&self, kind: &str, il: usize) -> Tensor<'_> {
        let name = format!("cache_{kind}-{il}");
        self.ctx
            .tensor(&name)
            .or_else(|| self.tiers.iter().find_map(|c| c.tensor(&name)))
            .or_else(|| self.views.tensor(&name))
            .unwrap_or_else(|| panic!("layer {il} has no KV cache"))
    }

    /// A tensor-parallel cache's tiered tensors, whole (`cells` rows), in the tier's meta
    /// buffer: each GPU's slice maps its own VRAM and host rows.
    ///
    /// The meta buffer keeps the per-device copies of a graph's views of its tensors in a pool
    /// sized by its tensor count (16 views per tensor). Sequence slots view each cache once per
    /// sequence (and permute those views), so four sequences overflow it: empty tensors after
    /// the caches raise it to [`SPLIT_TIER_VIEWS`] views per cache.
    fn split_tier(
        split: SplitTier,
        caches: &[CacheTensor],
        cells: i64,
    ) -> Result<Context, GgmlError> {
        let pad = caches.len() * (SPLIT_TIER_VIEWS / 16 - 1);
        let ctx = Context::new(caches.len().max(1) + pad, 0);
        for (name, ty, row, extra) in caches {
            ctx.new_tensor(*ty, &[*row, cells + extra]).named(name);
        }
        for i in 0..pad {
            ctx.new_tensor(crate::ggml::ffi::ggml_type_GGML_TYPE_I8, &[0])
                .named(&format!("view_room-{i}"));
        }
        ctx.alloc_tensors(split.0)?;
        Ok(ctx)
    }

    /// The host tier's tensors: per cache, `host_blocks` blocks of rows in a pinned arena on
    /// the GPU's node, then a tiered range of the VRAM blocks before them.
    #[cfg(feature = "cuda")]
    #[allow(clippy::type_complexity)]
    fn host_tier(
        t: KvTier,
        host_blocks: usize,
        caches: &[CacheTensor],
    ) -> Result<(Vec<Context>, Option<Context>), GgmlError> {
        use crate::residency::cuda;
        let alloc = |msg: String| GgmlError::Alloc {
            bytes: 0,
            buffer: format!("KV host tier on node {}: {msg}", t.node),
        };
        let arena_buft = cuda::host_arena(t.node, &[t.device]).map_err(alloc)?;
        let arena = Context::new(caches.len().max(1), 0);
        let host_cells = (host_blocks * KV_BLOCK) as i64;
        for (name, ty, row, extra) in caches {
            arena
                .new_tensor(*ty, &[*row, host_cells + extra])
                .named(&format!("host_{name}"));
        }
        arena.alloc_tensors(arena_buft)?;
        let mut tiers = Vec::new();
        for (name, ..) in caches {
            let host = arena.tensor(&format!("host_{name}")).expect("just made");
            tiers.push(cuda::tiered_rows(
                t.device,
                host,
                t.hbm_blocks * KV_BLOCK,
                name,
            )?);
        }
        Ok((tiers, Some(arena)))
    }

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::type_complexity)]
    fn host_tier(
        _: KvTier,
        _: usize,
        _: &[CacheTensor],
    ) -> Result<(Vec<Context>, Option<Context>), GgmlError> {
        Err(GgmlError::Alloc {
            bytes: 0,
            buffer: "a KV host tier needs CUDA".into(),
        })
    }

    /// Blocks in VRAM (the rest are in the host tier), or every block without one.
    pub fn hbm_blocks(&self) -> usize {
        self.hbm_blocks
    }

    /// The host tier layer `il` reads through, for paged attention: `None` without one, or for
    /// a ring, or without staging slots.
    pub fn paged_tier(&self, il: usize) -> Option<super::paged::Tier<'_>> {
        if self.stage_blocks == 0 || self.is_ring(il) {
            return None;
        }
        Some(super::paged::Tier {
            hbm_blocks: self.hbm_blocks,
            stage_blocks: self.stage_blocks,
            k_stage: self.ctx.tensor("stage_k")?,
            v_stage: self.ctx.tensor("stage_v")?,
        })
    }

    /// Staging slots in blocks (0 without a host tier or staging).
    pub fn stage_blocks(&self) -> usize {
        self.stage_blocks
    }

    /// Bytes `stage_blocks` staging slots take in this pool's device memory: blocks of its
    /// widest full-attention (not ring) layer's K and V.
    pub fn stage_bytes(&self, stage_blocks: usize) -> usize {
        if self.compact.iter().any(|&r| r > 0) {
            let cell = Self::stage_cell_bytes(&self.dims, &self.ring_layers, self.types);
            return 2 * cell * KV_BLOCK * stage_blocks;
        }
        self.dims
            .iter()
            .enumerate()
            .filter(|&(il, _)| !self.ring_layers[il])
            .filter_map(|(_, d)| d.map(|(d, row)| (d as usize, row as usize)))
            .map(|(d, row)| self.types.row_bytes(d, row))
            .max()
            .unwrap_or(0)
            * KV_BLOCK
            * stage_blocks
    }

    /// Adds to `graph` the writes of `k` and `v` (F32 `[head_dim, n_head_kv, n]`) to cells
    /// `cells` (I32 `[n]`) of layer `il`. They must be expanded before the attention that reads
    /// them, which is why this takes the graph.
    pub fn write<'a>(
        &'a self,
        ctx: &'a Context,
        graph: &mut Graph<'a>,
        il: usize,
        k: Tensor<'a>,
        v: Tensor<'a>,
        cells: Tensor<'a>,
    ) {
        assert_eq!(
            self.source(il),
            il,
            "layer {il} reads another layer's cache"
        );
        let n = k.ne()[2];
        let (_, row) = self.dims(il);
        for (kind, x, ty) in [("k", k, self.types.k), ("v", v, self.types.v)] {
            let x = ctx.reshape_2d(x, row, n);
            match self.tq_cache(il, ty) {
                // TurboQuant: Legion's kernel, in place of set_rows.
                Some(c) => super::turboquant::write(ctx, graph, c, self.tensor(kind, il), x, cells),
                None => graph.expand(ctx.set_rows(self.tensor(kind, il), x, cells)),
            }
        }
    }

    /// [`Self::write`] for a compact layer (see [`KvPool`]): `granules` F32
    /// `[head_dim + r, n_head_kv, n]`, contiguous.
    pub fn write_granules<'a>(
        &'a self,
        ctx: &'a Context,
        graph: &mut Graph<'a>,
        il: usize,
        granules: Tensor<'a>,
        cells: Tensor<'a>,
    ) {
        assert!(
            self.source(il) == il && self.compact(il) > 0,
            "layer {il} has no compact cache of its own"
        );
        let (_, row) = self.dims(il);
        let x = ctx.reshape_2d(granules, row, granules.ne()[2]);
        graph.expand(ctx.set_rows(self.tensor("k", il), x, cells));
    }

    /// A compact layer's gathered granules `g` (F16 `[row, n_kv, n_seqs]`, contiguous) as K
    /// `[head_dim, n_head_kv, n_kv, n_seqs]` and V `[head_dim, n_head_kv, n_kv, n_seqs]`,
    /// both views (strided over the granules).
    fn granule_views<'a>(
        &self,
        ctx: &'a Context,
        il: usize,
        g: Tensor<'a>,
        n_kv: i64,
        n_seqs: i64,
    ) -> (Tensor<'a>, Tensor<'a>) {
        let (d, row) = self.dims(il);
        let (r, h) = (self.compact(il), self.n_head_kv(il));
        let es = g.element_size();
        let nb = [
            (d + r) as usize * es,
            row as usize * es,
            (row * n_kv) as usize * es,
        ];
        let view = |off: i64| ctx.view_4d(g, [d, h, n_kv, n_seqs], nb, off as usize * es);
        (view(0), view(r))
    }

    /// The TurboQuant cache of type `ty` layer `il` reads, if `ty` is one: its layout, and its
    /// rotation's seed (the cache's layer).
    fn tq_cache(&self, il: usize, ty: KvType) -> Option<super::turboquant::Cache> {
        let bits = ty.tq_bits()?;
        let (d, row) = self.dims(il);
        Some(super::turboquant::Cache {
            layout: super::turboquant::Layout {
                d: d as usize,
                n_kv: (row / d) as usize,
                bits,
            },
            seed: self.source(il) as u32,
        })
    }

    /// Layer `il`'s K and V TurboQuant caches (`None` for a ggml type), or `None` if neither
    /// is one: such a layer's paged attention is [`super::paged::tq`]'s.
    pub fn tq(&self, il: usize) -> Option<super::paged::tq::TqCaches> {
        let k = self.tq_cache(il, self.types.k);
        let v = self.tq_cache(il, self.types.v);
        (k.is_some() || v.is_some()).then_some(super::paged::tq::TqCaches { k, v })
    }

    /// Rows `ids` of layer `il`'s `kind` cache (`types.k` or `.v`, `ty`) as F16, `per_id` cells
    /// per id: TurboQuant's dequantizing kernel, or `None` for a ggml type.
    fn tq_rows<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        kind: &str,
        ty: KvType,
        ids: Tensor<'a>,
        per_id: usize,
    ) -> Option<Tensor<'a>> {
        let c = self.tq_cache(il, ty)?;
        let pool = self.tensor(kind, self.source(il));
        Some(super::turboquant::rows_f16(ctx, c, pool, ids, per_id))
    }

    /// The cells `cells` (I32 `[n_kv × n_seqs]`, each sequence's in order) of the cache layer
    /// `il` reads, K and V both `[head_dim, n_head_kv, n_kv, n_seqs]` F16 (for flash attention;
    /// gathered by cells, so a cache split by heads gathers on each device).
    /// Rows `ids` of the cache tensor `pool` as F16: on a device, straight from the kernel
    /// (no F32 rows, which at long contexts are most of a gather step's buffer); in host
    /// memory, as F32 rows and a cast (the CPU's get_rows writes F32 only).
    fn rows_f16<'a>(&self, ctx: &'a Context, pool: Tensor<'a>, ids: Tensor<'a>) -> Tensor<'a> {
        if self.host {
            ctx.cast(ctx.get_rows(pool, ids), types::F16)
        } else {
            ctx.get_rows_f16(pool, ids)
        }
    }

    /// The cells `cells` (I32 `[n_kv × n_seqs]`, each sequence's in order) of the cache layer
    /// `il` reads, K and V both `[head_dim, n_head_kv, n_kv, n_seqs]` F16 (for flash attention;
    /// gathered by cells, so a cache split by heads gathers on each device).
    pub fn read_cells<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        cells: Tensor<'a>,
        n_kv: i64,
        n_seqs: i64,
    ) -> (Tensor<'a>, Tensor<'a>) {
        let (d, row) = self.dims(il);
        let src = self.source(il);
        if self.compact(il) > 0 {
            // The granules once; K and V are views of them.
            let g = self.rows_f16(ctx, self.tensor("k", src), cells);
            return self.granule_views(ctx, il, g, n_kv, n_seqs);
        }
        let get = |kind, ty| {
            let rows = self
                .tq_rows(ctx, il, kind, ty, cells, 1)
                .unwrap_or_else(|| self.rows_f16(ctx, self.tensor(kind, src), cells));
            ctx.reshape_4d(rows, [d, row / d, n_kv, n_seqs])
        };
        (get("k", self.types.k), get("v", self.types.v))
    }

    /// [`Self::read`] for a compact layer: the granules gathered once, then K and V
    /// transposed made contiguous from their views.
    fn read_compact<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        blocks: Tensor<'a>,
        n_kv: i64,
        n_seqs: i64,
    ) -> (Tensor<'a>, Tensor<'a>) {
        let (_, row) = self.dims(il);
        let pool = self.tensor("k", self.source(il));
        let g = if blocks.ne()[0] == n_kv * n_seqs {
            self.rows_f16(ctx, pool, blocks)
        } else {
            // Whole blocks (without the pad cell).
            let n_blocks = pool.ne()[1] / KV_BLOCK as i64;
            let per = KV_BLOCK * pool.nb()[1];
            let rows = ctx.view_2d(pool, [row * KV_BLOCK as i64, n_blocks], per, 0);
            if self.types.v == KvType::F16 {
                // As I32 pairs of F16, so get_rows copies the bits.
                let rows = ctx.reinterpret(
                    rows,
                    types::I32,
                    [row * KV_BLOCK as i64 / 2, n_blocks, 1, 1],
                );
                let g = ctx.get_rows(rows, blocks);
                ctx.reinterpret(g, types::F16, [row, n_kv * n_seqs, 1, 1])
            } else {
                self.rows_f16(ctx, rows, blocks)
            }
        };
        let g = ctx.reshape_3d(g, row, n_kv, n_seqs);
        let (k, v) = self.granule_views(ctx, il, g, n_kv, n_seqs);
        (ctx.cont(k), ctx.cont(ctx.permute(v, [1, 2, 0, 3])))
    }

    /// The blocks `blocks` (I32 `[n_kv / KV_BLOCK × n_seqs]`, each sequence's in order) of the
    /// cache layer `il` reads as K `[head_dim, n_head_kv, n_kv, n_seqs]` and V transposed
    /// `[n_kv, head_dim, n_head_kv, n_seqs]`, both F16 and contiguous. Or with one index per
    /// cell (`[n_kv × n_seqs]`).
    pub fn read<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        blocks: Tensor<'a>,
        n_kv: i64,
        n_seqs: i64,
    ) -> (Tensor<'a>, Tensor<'a>) {
        if self.compact(il) > 0 {
            return self.read_compact(ctx, il, blocks, n_kv, n_seqs);
        }
        let (d, row) = self.dims(il);
        let src = self.source(il);
        let h = row / d;
        if blocks.ne()[0] == n_kv * n_seqs {
            // One index per cell (`Model::set_cell_gather`): rows of every head, so a cache
            // split by heads gathers on each device; F16, as the block gather reads them.
            let k = self
                .tq_rows(ctx, il, "k", self.types.k, blocks, 1)
                .unwrap_or_else(|| self.rows_f16(ctx, self.tensor("k", src), blocks));
            let v = self
                .tq_rows(ctx, il, "v", self.types.v, blocks, 1)
                .unwrap_or_else(|| self.rows_f16(ctx, self.tensor("v", src), blocks));
            let k = ctx.reshape_4d(k, [d, h, n_kv, n_seqs]);
            let vt = ctx.cont(ctx.transpose(ctx.reshape_3d(v, row, n_kv, n_seqs)));
            return (k, ctx.reshape_4d(vt, [n_kv, d, h, n_seqs]));
        }
        // Every listed block's K or V, F16 `[row × KV_BLOCK, listed]`.
        let gather = |kind, ty: KvType| {
            if let Some(rows) = self.tq_rows(ctx, il, kind, ty, blocks, KV_BLOCK) {
                return rows;
            }
            let pool = self.tensor(kind, src);
            // One row per block (a ring layer's pool has its own count).
            let n_blocks = pool.ne()[1] / KV_BLOCK as i64;
            if ty == KvType::F16 {
                // As I32 pairs of F16, so get_rows copies the bits.
                let rows = ctx.reinterpret(
                    pool,
                    types::I32,
                    [row * KV_BLOCK as i64 / 2, n_blocks, 1, 1],
                );
                ctx.get_rows(rows, blocks)
            } else {
                // Dequantized to F16, as the matmuls take it.
                let rows = ctx.reshape_2d(pool, row * KV_BLOCK as i64, n_blocks);
                self.rows_f16(ctx, rows, blocks)
            }
        };
        let k = ctx.reinterpret(gather("k", self.types.k), types::F16, [d, h, n_kv, n_seqs]);
        let v = ctx.reinterpret(
            gather("v", self.types.v),
            types::F16,
            [row, n_kv, n_seqs, 1],
        );
        let vt = ctx.cont(ctx.transpose(v));
        (k, ctx.reshape_4d(vt, [n_kv, d, h, n_seqs]))
    }
}

/// The DeltaNet layers' state after the last token, per slot: the conv inputs of the last
/// `d_conv - 1` tokens (`[d_conv - 1, conv_dim, 1, slots]`) and the delta-rule state
/// (`[head_v, head_v, n_v_heads, slots]`), both F32. Nemotron-H's Mamba-2 layers have the same
/// conv history and an SSM state of `[d_state, head_dim, n_head, slots]`
/// ([`Hparams::ssm_state_ne`]).
///
/// qwen4exp adds, for its per-layer-embedding layer, the conv inputs of the last
/// `(kernel - 1) × n-gram` tokens (`[history, hc × n_embd, 1, slots]`, F32) and, on the host,
/// the last `n-gram - 1` tokens (the n-gram hash needs them before the step's own tokens).
pub struct StatePool {
    ctx: Context,
    n_slots: usize,
    /// The recurrent layers.
    layers: Vec<usize>,
    /// The per-layer-embedding layer, if any.
    ple_layer: Option<usize>,
    /// Per slot, the last `n-gram - 1` tokens, oldest first ([`NO_TOKEN`] before the start).
    tokens: std::sync::Mutex<Vec<u32>>,
    /// Tokens kept per slot in `tokens`.
    n_hist: usize,
    /// Bytes of one token's output row of the state update (`ssm_d_inner` F32).
    out_row: usize,
    /// Whether the pool lives in host memory (CPU, or saved states off the GPU, #441): copies
    /// between it and a device pool go over the host link ([`Self::copy_slots`]).
    host: bool,
    /// The CUDA device, when the pool is in ggml's plain device memory of one: then
    /// [`Self::clear_slot`] zeroes the slot's ranges with one wait for them all.
    #[cfg(feature = "cuda")]
    cuda_device: Option<usize>,
}

/// A missing predecessor in [`StatePool`]'s token history (before the sequence start).
pub const NO_TOKEN: u32 = u32::MAX;

impl StatePool {
    pub fn new(hp: &Hparams, buft: BufferType, n_slots: usize) -> Result<Self, GgmlError> {
        Self::new_layers(hp, buft, n_slots, 0..hp.n_layer)
    }

    /// A pool for the layers `layers` only (a pipeline stage's): their recurrent states, and
    /// the per-layer-embedding state and token history if the stage has that layer.
    pub fn new_layers(
        hp: &Hparams,
        buft: BufferType,
        n_slots: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<Self, GgmlError> {
        let n_slots = n_slots.max(1);
        let rec: Vec<usize> = layers
            .clone()
            .filter(|&il| il < hp.n_layer && hp.is_recurrent(il))
            .collect();
        let conv_here = hp.arch.token_conv(hp).filter(|c| layers.contains(&c.layer));
        let ctx = Context::new(3 * rec.len().max(1) + 1, 0);
        let conv_dim = hp.ssm_d_inner + 2 * hp.ssm_n_group * hp.ssm_d_state;
        let [a, b, c] = hp.ssm_state_ne();
        let s = n_slots as i64;
        for &il in &rec {
            // Slots in dim 3, as for the delta-rule state.
            ctx.new_tensor(types::F32, &[hp.ssm_d_conv - 1, conv_dim, 1, s])
                .named(&format!("conv_state-{il}"));
            // The delta-rule (or SSM) states sit right after room for `n_slots` output rows of
            // gated_delta_net (ssm_scan), so a decode step's output can be placed with its
            // final states landing on slots 0.. (see `decode_output`).
            let row = hp.ssm_d_inner;
            let landing = ctx
                .new_tensor(types::F32, &[row * s + a * b * c * s])
                .named(&format!("ssm_landing-{il}"));
            let es = 4;
            ctx.view_4d(
                landing,
                [a, b, c, s],
                [
                    a as usize * es,
                    (a * b) as usize * es,
                    (a * b * c) as usize * es,
                ],
                (row * s) as usize * es,
            )
            .named(&format!("ssm_state-{il}"));
        }
        if let Some(c) = conv_here {
            ctx.new_tensor(types::F32, &[c.history, c.width, 1, s])
                .named(&format!("ple_state-{}", c.layer));
        }
        ctx.alloc_tensors(buft)?;
        ctx.clear_buffers(0);
        // Graph builds look the states up by name, per layer.
        ctx.index_names();
        let n_hist = conv_here.map_or(0, |c| c.n_hist);
        Ok(Self {
            ctx,
            n_slots,
            layers: rec,
            ple_layer: conv_here.map(|c| c.layer),
            tokens: std::sync::Mutex::new(vec![NO_TOKEN; n_hist * n_slots]),
            n_hist,
            out_row: hp.ssm_d_inner as usize * 4,
            host: buft.is_host(),
            #[cfg(feature = "cuda")]
            cuda_device: crate::residency::cuda::plain_cuda_device(buft),
        })
    }

    /// Whether the pool lives in host memory.
    pub fn is_host(&self) -> bool {
        self.host
    }

    /// Bytes the pool's tensors take.
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|&il| {
                let landing = self.ctx.tensor(&format!("ssm_landing-{il}"));
                self.conv(il).nbytes() + landing.map_or(0, |t| t.nbytes())
            })
            .sum::<usize>()
            + self.ple_layer.map_or(0, |il| self.ple(il).nbytes())
    }

    /// Bytes of one slot over all layers.
    pub fn slot_bytes(hp: &Hparams) -> usize {
        Self::slot_bytes_layers(hp, 0..hp.n_layer)
    }

    /// Bytes of one slot over the layers `layers`.
    pub fn slot_bytes_layers(hp: &Hparams, layers: std::ops::Range<usize>) -> usize {
        let n_rec = layers
            .clone()
            .filter(|&il| il < hp.n_layer && hp.is_recurrent(il))
            .count();
        let conv_dim = hp.ssm_d_inner + 2 * hp.ssm_n_group * hp.ssm_d_state;
        let state: i64 = hp.ssm_state_ne().iter().product();
        let ple = match hp.arch.token_conv(hp) {
            Some(c) if layers.contains(&c.layer) => 4 * (c.history * c.width) as usize,
            _ => 0,
        };
        n_rec * 4 * ((hp.ssm_d_conv - 1) * conv_dim + state) as usize + ple
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    fn tensors(&self) -> impl Iterator<Item = Tensor<'_>> + '_ {
        self.layers
            .iter()
            .flat_map(move |&il| [self.conv(il), self.ssm(il)])
            .chain(self.ple_layer.map(|il| self.ple(il)))
    }

    fn ple(&self, il: usize) -> Tensor<'_> {
        self.ctx
            .tensor(&format!("ple_state-{il}"))
            .unwrap_or_else(|| panic!("layer {il} has no per-layer-embedding state"))
    }

    /// Per-layer-embedding conv history of layer `il`, slots `slot..slot + n`:
    /// `[history, hc × n_embd, n]`.
    pub fn ple_slots<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        slot: usize,
        n: usize,
    ) -> Tensor<'a> {
        assert!(slot + n <= self.n_slots, "slots {slot}+{n} out of range");
        let v = Self::slots(ctx, self.ple(il), slot, n);
        let [a, b, _, _] = v.ne();
        ctx.reshape_3d(v, a, b, n as i64)
    }

    /// The token history of `slot` (oldest first).
    pub fn peek_tokens(&self, slot: usize) -> Vec<u32> {
        self.tokens.lock().unwrap()[slot * self.n_hist..][..self.n_hist].to_vec()
    }

    /// The token history of `slot` (oldest first), then `tokens` appended to it.
    pub fn push_tokens(&self, slot: usize, tokens: &[u32]) -> Vec<u32> {
        let mut all = self.tokens.lock().unwrap();
        let h = &mut all[slot * self.n_hist..][..self.n_hist];
        let before = h.to_vec();
        let joined: Vec<u32> = before.iter().chain(tokens).copied().collect();
        h.copy_from_slice(&joined[joined.len() - self.n_hist..]);
        before
    }

    /// Sets the token history of `slot` to the end of `history` (oldest first; [`NO_TOKEN`]
    /// before it when it is shorter), as after `before` and then `history`'s tokens.
    pub fn set_tokens(&self, slot: usize, before: &[u32], history: &[u32]) {
        if self.n_hist == 0 {
            return;
        }
        let mut all = self.tokens.lock().unwrap();
        let h = &mut all[slot * self.n_hist..][..self.n_hist];
        let joined: Vec<u32> = before.iter().chain(history).copied().collect();
        h.copy_from_slice(&joined[joined.len() - self.n_hist..]);
    }

    fn conv(&self, il: usize) -> Tensor<'_> {
        self.ctx
            .tensor(&format!("conv_state-{il}"))
            .unwrap_or_else(|| panic!("layer {il} has no recurrent state"))
    }

    fn ssm(&self, il: usize) -> Tensor<'_> {
        self.ctx
            .tensor(&format!("ssm_state-{il}"))
            .unwrap_or_else(|| panic!("layer {il} has no recurrent state"))
    }

    /// Slots `slot..slot + n` of `t` (the last dim), as a contiguous view.
    fn slots<'a>(ctx: &'a Context, t: Tensor<'a>, slot: usize, n: usize) -> Tensor<'a> {
        let [a, b, c, _] = t.ne();
        let nb = t.nb();
        ctx.view_4d(t, [a, b, c, n as i64], [nb[1], nb[2], nb[3]], slot * nb[3])
    }

    /// Conv history of layer `il`, slots `slot..slot + n`: `[d_conv - 1, conv_dim, n]`.
    pub fn conv_slots<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        slot: usize,
        n: usize,
    ) -> Tensor<'a> {
        assert!(slot + n <= self.n_slots, "slots {slot}+{n} out of range");
        let v = Self::slots(ctx, self.conv(il), slot, n);
        let [a, b, _, _] = v.ne();
        ctx.reshape_3d(v, a, b, n as i64)
    }

    /// Delta-rule state of layer `il`, slots `slot..slot + n`: `[head_v, head_v, n_v, n]`.
    pub fn ssm_slots<'a>(
        &'a self,
        ctx: &'a Context,
        il: usize,
        slot: usize,
        n: usize,
    ) -> Tensor<'a> {
        assert!(slot + n <= self.n_slots, "slots {slot}+{n} out of range");
        Self::slots(ctx, self.ssm(il), slot, n)
    }

    /// Gives `out`, the result of a gated_delta_net (or ssm_scan) over `n_rows` token rows and
    /// the states of slots `0..`, memory such that its final states are written straight into those slots
    /// (instead of being copied there after the op) and its token rows just before them.
    /// Returns false (and does nothing) for host memory: the CPU kernel's in-place behaviour
    /// is not guaranteed, and there the copy is cheap anyway.
    ///
    /// The op then reads and writes the same state memory. That is sound for ggml's CUDA
    /// kernel, where every thread loads its part of the state before the token loop and stores
    /// the same part after it.
    pub fn decode_output(&self, il: usize, out: Tensor<'_>, n_rows: usize) -> bool {
        let landing = self
            .ctx
            .tensor(&format!("ssm_landing-{il}"))
            .unwrap_or_else(|| panic!("layer {il} has no recurrent state"));
        if landing.is_host() || n_rows > self.n_slots {
            return false;
        }
        let row = self.out_row;
        let pad = self.n_slots * row;
        // SAFETY: `out` is an unallocated op result of the step's context; the range is inside
        // `landing` (asserted by place_in) and is the token-row pad plus the states it updates.
        unsafe { out.place_in(landing, pad - n_rows * row) };
        true
    }

    /// Back to the empty state (zeros) in `slot`.
    pub fn clear_slot(&self, slot: usize) {
        assert!(slot < self.n_slots);
        let ranges = self.slot_ranges(slot);
        if !self.zero_on_device(slot, &ranges) {
            for (t, off, n) in ranges {
                t.memset(off, n, 0);
            }
        }
        self.tokens.lock().unwrap()[slot * self.n_hist..][..self.n_hist].fill(NO_TOKEN);
    }

    /// Per tensor of the pool, the bytes of slot `slot`: `(tensor, offset, bytes)`.
    fn slot_ranges(&self, slot: usize) -> Vec<(Tensor<'_>, usize, usize)> {
        self.tensors()
            .map(|t| {
                let per = t.nbytes() / self.n_slots;
                (t, slot * per, per)
            })
            .collect()
    }

    /// Zeroes `ranges` (of [`Self::slot_ranges`]) in the pool's plain CUDA memory with one wait
    /// at the end, the same zeroes as one `memset` each but without a wait per tensor (#836:
    /// ~60 memset + synchronize pairs before a new prompt's first step). False (nothing done)
    /// for a pool in any other memory.
    #[cfg(feature = "cuda")]
    fn zero_on_device(&self, slot: usize, ranges: &[(Tensor<'_>, usize, usize)]) -> bool {
        let Some(device) = self.cuda_device else {
            return false;
        };
        let at: Vec<(usize, usize)> = ranges
            .iter()
            // SAFETY: an allocated tensor of the pool; the range is inside it.
            .map(|&(t, off, n)| (unsafe { (*t.raw()).data } as usize + off, n))
            .collect();
        // SAFETY: the slot's ranges of the pool's device memory; no step runs on a slot while it
        // is cleared.
        unsafe { crate::residency::cuda::zero_ranges_now(device, &at) }
            .unwrap_or_else(|e| panic!("clearing state slot {slot}: {e}"));
        true
    }

    #[cfg(not(feature = "cuda"))]
    fn zero_on_device(&self, _slot: usize, _ranges: &[(Tensor<'_>, usize, usize)]) -> bool {
        false
    }

    /// Runs the copies `(from pool, from slot, to pool, to slot)` in order, as one graph on
    /// `backend`; copies between a pool in host memory and one in device memory go over the
    /// host link instead, one by one. The pools must have the same model's shapes and live in
    /// `backend`'s memory or host memory.
    pub fn copy_slots(
        backend: &mut Backend,
        copies: &[(&StatePool, usize, &StatePool, usize)],
    ) -> Result<(), GgmlError> {
        if copies.is_empty() {
            return Ok(());
        }
        if copies.iter().any(|(a, _, b, _)| a.host != b.host) {
            // Between host and device memory: one copy at a time, in order, the host side read
            // or written by DMA once the work queued on the device is done.
            backend.synchronize();
            for &(src, s, dst, d) in copies {
                if src.host == dst.host {
                    Self::copy_slots(backend, &[(src, s, dst, d)])?;
                    continue;
                }
                assert!(s < src.n_slots && d < dst.n_slots, "slot out of range");
                for (a, b) in src.tensors().zip(dst.tensors()) {
                    assert_eq!(a.ne()[..3], b.ne()[..3], "state shapes differ");
                    // Slots in dim 3, each contiguous.
                    let per = a.nb()[3];
                    assert_eq!(per, b.nb()[3], "state layouts differ");
                    a.copy_bytes_to(s * per, b, d * per, per);
                }
                Self::copy_tokens(src, s, dst, d);
            }
            return Ok(());
        }
        let n_tensors: usize = copies.iter().map(|(a, ..)| a.tensors().count() * 8).sum();
        let ctx = Context::new(n_tensors + 16, n_tensors + 16);
        let mut graph = ctx.new_graph(n_tensors + 16);
        for &(src, s, dst, d) in copies {
            assert!(s < src.n_slots && d < dst.n_slots, "slot out of range");
            for (a, b) in src.tensors().zip(dst.tensors()) {
                assert_eq!(a.ne()[..3], b.ne()[..3], "state shapes differ");
                let from = Self::slots(&ctx, a, s, 1);
                let to = Self::slots(&ctx, b, d, 1);
                graph.expand(ctx.cpy(from, to));
            }
        }
        ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))?;
        backend.compute(&graph)?;
        for &(src, s, dst, d) in copies {
            Self::copy_tokens(src, s, dst, d);
        }
        Ok(())
    }

    /// Copies slot `s`'s token history of `src` to slot `d` of `dst`.
    fn copy_tokens(src: &StatePool, s: usize, dst: &StatePool, d: usize) {
        let h = src.tokens.lock().unwrap()[s * src.n_hist..][..src.n_hist].to_vec();
        dst.tokens.lock().unwrap()[d * dst.n_hist..][..dst.n_hist].copy_from_slice(&h);
    }
}

/// Everything one sequence carries from step to step, for single-sequence use: its own KV pool
/// (blocks `0..` in order) and one recurrent-state slot.
pub struct SeqState {
    pub(super) kv: KvPool,
    pub(super) rec: StatePool,
    pub(super) blocks: Vec<u32>,
    n_past: usize,
}

impl SeqState {
    /// An empty state for up to `max_ctx` tokens (rounded up to the attention padding), in
    /// buffers of `buft` (the backend's buffer type: the graph reads and writes them directly).
    pub fn new(hp: &Hparams, buft: BufferType, max_ctx: usize) -> Result<Self, GgmlError> {
        let max_ctx = max_ctx.max(1).div_ceil(ATTN_KV_PAD as usize) * ATTN_KV_PAD as usize;
        let n_blocks = max_ctx / KV_BLOCK;
        Ok(Self {
            kv: KvPool::new(hp, buft, n_blocks)?,
            rec: StatePool::new(hp, buft, 1)?,
            blocks: (0..n_blocks as u32).collect(),
            n_past: 0,
        })
    }

    /// Tokens processed so far (the position of the next token).
    pub fn n_past(&self) -> usize {
        self.n_past
    }

    /// Capacity in tokens.
    pub fn max_ctx(&self) -> usize {
        self.kv.n_blocks * KV_BLOCK
    }

    /// Back to an empty sequence. The recurrent state must start from zero; the KV cache need
    /// not be cleared, since positions at or past `n_past` are never visible.
    pub fn reset(&mut self) {
        self.n_past = 0;
        self.rec.clear_slot(0);
    }

    pub(super) fn advance(&mut self, n: usize) {
        self.n_past += n;
    }
}

mod digest;
pub use digest::fnv1a;

#[cfg(test)]
mod tests;
