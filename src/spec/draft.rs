//! `--spec draft --spec-draft small.gguf`: a small model that shares the target's vocabulary
//! drafts tokens for it.
//!
//! [`DraftModel`] is a [`ModelDrafter`] around a second Qwen3.5 model in an [`Executor`] of its
//! own: its own weights, paged KV pool and recurrent-state pool, on the target's device (CPU or
//! the same GPU). It reads no hidden states of the target; it only needs the token history the
//! engine hands every call ([`DraftRequest::tokens`]). It drafts greedily (a point-mass
//! proposal, `Draft::probs` empty), which keeps the output distribution the target's own for
//! any sampler (see `crate::spec::accept`) and the greedy output identical to plain decoding.
//!
//! # Load-time check
//!
//! The draft's token strings must be the target's: same count and same [`VocabId`] hash (an
//! FNV-1a hash over every token string of `tokenizer.ggml.tokens`). A mismatch is refused with
//! an error naming both models ([`check_same_vocab`]); token ids would otherwise mean different
//! text in the two models and every draft would be junk.
//!
//! # Per sequence
//!
//! Every verified token (the prompt, accepted drafts and the target's own picks) goes through
//! the draft model exactly once, in order: the sequence's **live** state (its blocks in the
//! draft's KV pool and one slot of the draft's recurrent states) always holds exactly
//! `n_live` verified tokens. Each call:
//!
//! 1. *Catch up.* The tokens in `tokens[n_live..goal]` run as one prefill-like group per
//!    distinct length, in **work** slots (the live state is copied there first and copied back
//!    after, so groups can be arranged to have consecutive slots as the DeltaNet ops need).
//!    `goal` is the whole history when drafts are wanted (it ends with the target's last
//!    pick, which the first draft follows) and else what the target just ran. A sequence
//!    that starts late (it resumed from the prefix cache, so the target never ran its first
//!    `start` positions) is caught up from position 0 in chunks of [`FEED_CHUNK`] tokens per
//!    call, drafting nothing meanwhile.
//! 2. *Draft.* The logits of the last caught-up token give the first draft. The other `max - 1`
//!    drafts are one-token decode steps, batched over all drafting sequences, which advance the
//!    **work** slots only. Drafts are written to the KV cells past `n_live`, which nothing
//!    reads before the next call's catch-up overwrites them.
//!
//! # Rollback
//!
//! There is none to do: speculative tokens never enter the live state. The recurrent state of a
//! DeltaNet layer cannot be truncated, so instead of the target's replay (`Executor::commit`,
//! which needs the tokens' inputs stored by a verify group) the drafter keeps drafted tokens in
//! a throwaway copy of the state: after a partial acceptance the next call feeds the sequence's
//! true tokens (the accepted drafts and the target's pick), from the unchanged live state.
//! The accepted drafts are thus computed twice (once drafted, once caught up), which costs
//! one small group per call, and the price of the copies is two state copies per sequence and
//! call (a small model's states are a few MiB). The paged KV rolls back by itself: cells past
//! `n_live` are invisible.
//!
//! Sequences the drafter has no room for (more sequences than slots, KV blocks exhausted) get
//! no drafts until it has; that never changes the output.

use std::collections::HashMap;
use std::path::Path;

use crate::ggml::{Backend, gpu_device};
use crate::gguf::{GgufFile, Value};
use crate::kv::{BlockPool, BlockTable};
use crate::model::core::StatePool;
use crate::model::core::{Executor, Group, KV_BLOCK, KvPool, Logits, Model, Pool, SeqInput};
use crate::spec::{DraftRequest, ModelDrafter, SeqId, TargetInfo};

/// Most tokens one call catches a sequence up by (bounds the attention scores buffer, as
/// `PREFILL_CHUNK` does in the engine).
pub const FEED_CHUNK: usize = 512;

/// A vocabulary's identity: how many tokens it has and a hash of all their strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocabId {
    pub count: usize,
    pub hash: u64,
}

impl VocabId {
    /// FNV-1a over each token's length and bytes, in order.
    pub fn of_tokens<'a>(tokens: impl IntoIterator<Item = &'a str>) -> Self {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut count = 0;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                hash = (hash ^ b as u64).wrapping_mul(PRIME);
            }
        };
        for t in tokens {
            // The length keeps ["ab", "c"] apart from ["a", "bc"].
            eat(&(t.len() as u64).to_le_bytes());
            eat(t.as_bytes());
            count += 1;
        }
        Self { count, hash }
    }

    /// The vocabulary of a GGUF (`tokenizer.ggml.tokens`).
    pub fn of_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let Some(Value::Array(tokens)) = gguf.get("tokenizer.ggml.tokens") else {
            return Err("it has no tokenizer.ggml.tokens".into());
        };
        let strings: Vec<&str> = tokens
            .iter()
            .map(|v| v.as_str().ok_or("tokenizer.ggml.tokens holds a non-string"))
            .collect::<Result<_, _>>()?;
        Ok(Self::of_tokens(strings))
    }
}

impl std::fmt::Display for VocabId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} tokens, hash {:016x}", self.count, self.hash)
    }
}

/// Refuses a draft model whose vocabulary is not the target's.
pub fn check_same_vocab(target: &Path, draft: &Path) -> Result<(), String> {
    let id = |path: &Path| -> Result<VocabId, String> {
        let gguf = GgufFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        VocabId::of_gguf(&gguf).map_err(|e| format!("{}: {e}", path.display()))
    };
    let (t, d) = (id(target)?, id(draft)?);
    if t != d {
        return Err(format!(
            "the draft model {} cannot draft for {}: their vocabularies differ \
             (draft: {d}; target: {t}); --spec draft needs the same tokenizer",
            draft.display(),
            target.display(),
        ));
    }
    Ok(())
}

/// One followed sequence.
struct DraftSeq {
    /// The live recurrent-state slot.
    slot: usize,
    /// Verified tokens in the live state and KV cache.
    n_live: usize,
    table: BlockTable,
    /// `table`'s blocks as the executor wants them.
    blocks: Vec<u32>,
}

/// What one call does for one sequence.
struct Plan {
    /// Index in the batch.
    idx: usize,
    seq: SeqId,
    /// Tokens already live; this call feeds `tokens[from..from + m]`.
    from: usize,
    m: usize,
    /// Drafts wanted (0: only catch up).
    want: usize,
}

impl Plan {
    fn goal(&self) -> usize {
        self.from + self.m
    }
}

/// A [`ModelDrafter`] running a small draft model (see the module docs).
pub struct DraftModel {
    exec: Executor,
    pool: BlockPool,
    seqs: HashMap<SeqId, DraftSeq>,
    /// Live slots not in use.
    free: Vec<usize>,
    /// Live slots are `0..max_seqs`, work slots `max_seqs..2 * max_seqs`.
    max_seqs: usize,
    /// Longest sequence the target runs.
    ctx_size: usize,
    /// Most drafts per sequence and call.
    k: usize,
    /// Token ids at or past this are not the target's: drafting stops before them.
    n_vocab: usize,
    chunk: usize,
}

impl DraftModel {
    /// Loads `target.draft` as the drafter (after checking its vocabulary against
    /// `target.model`), drafting up to `k` tokens per sequence and call.
    pub fn load(target: &TargetInfo, k: usize) -> Result<Self, String> {
        let path = target
            .draft
            .as_ref()
            .ok_or("--spec draft needs --spec-draft <model.gguf>")?;
        check_same_vocab(&target.model, path)?;
        let gguf = GgufFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;

        let mut device = None;
        let mut backend = if target.gpu {
            let dev = gpu_device().ok_or("no CUDA device for the draft model")?;
            let need: u64 = gguf.tensors().iter().map(|t| t.size).sum();
            let (free, _) = dev.memory();
            if need > free as u64 {
                return Err(format!(
                    "the draft model {} needs {:.1} GiB for its weights, {} has {:.1} GiB free",
                    path.display(),
                    need as f64 / (1u64 << 30) as f64,
                    dev.description(),
                    free as f64 / (1u64 << 30) as f64,
                ));
            }
            device = Some(dev);
            dev.init().map_err(|e| e.to_string())?
        } else {
            Backend::cpu().map_err(|e| e.to_string())?
        };
        backend.set_threads(target.threads);
        backend.set_use_ref(target.use_ref);
        let buft = backend.buffer_type();
        let model = Model::load(&gguf, |_| buft)
            .map_err(|e| format!("draft model {}: {e}", path.display()))?;
        let hp = model.hp.clone();

        let max_seqs = target.max_seqs.max(1);
        let max_tokens = target.ctx_size + k;
        let seq_blocks = max_tokens.div_ceil(KV_BLOCK);
        let state_bytes = 2 * max_seqs * StatePool::slot_bytes(&hp);
        let mut n_blocks = max_seqs * seq_blocks;
        if let Some(dev) = device {
            // A quarter of what the target's weights left; the target sizes its own cache
            // from the rest.
            let (free, _) = dev.memory();
            let room = (free / 4).saturating_sub(state_bytes);
            n_blocks = n_blocks.min(room / KvPool::block_bytes(&hp));
        }
        let n_blocks = n_blocks.max(seq_blocks);
        let mut exec = Executor::new(model, backend, n_blocks, 2 * max_seqs, 0).map_err(|e| {
            format!(
                "cannot allocate the draft model's state ({n_blocks} KV blocks, {max_seqs} \
                 sequences) on {}: {e}",
                buft.name()
            )
        })?;
        exec.set_max_seq_tokens(max_tokens);
        tracing::info!(
            draft = %path.display(),
            backend = %exec.backend_name(),
            kv_cache_tokens = n_blocks * KV_BLOCK,
            max_seqs,
            "draft model loaded"
        );
        Ok(Self {
            exec,
            pool: BlockPool::new(n_blocks as u32, KV_BLOCK as u32),
            seqs: HashMap::new(),
            free: (0..max_seqs).rev().collect(),
            max_seqs,
            ctx_size: target.ctx_size,
            k,
            n_vocab: target.n_vocab,
            chunk: FEED_CHUNK,
        })
    }

    /// Sets the most tokens one call catches a sequence up by (default [`FEED_CHUNK`]).
    pub fn set_feed_chunk(&mut self, tokens: usize) {
        self.chunk = tokens.max(1);
    }

    /// Sequences being followed.
    pub fn followed(&self) -> usize {
        self.seqs.len()
    }

    /// Verified tokens of `seq` in the draft model's state, if followed.
    pub fn live_tokens(&self, seq: SeqId) -> Option<usize> {
        self.seqs.get(&seq).map(|s| s.n_live)
    }

    /// Plans the call for one request, or `None` if there is nothing to do or no room.
    fn plan(&mut self, idx: usize, r: &DraftRequest<'_>) -> Option<Plan> {
        let goal = if r.max > 0 {
            r.tokens.len()
        } else {
            r.start + r.ran
        };
        let want = r.max.min(self.k);
        if goal > self.ctx_size.min(r.tokens.len()) {
            return None;
        }
        if !self.seqs.contains_key(&r.seq) {
            if goal == 0 {
                return None;
            }
            let slot = self.free.pop()?;
            self.exec.clear_slot(slot);
            self.seqs.insert(
                r.seq,
                DraftSeq {
                    slot,
                    n_live: 0,
                    table: BlockTable::new(),
                    blocks: Vec::new(),
                },
            );
        }
        let seq = self.seqs.get_mut(&r.seq).unwrap();
        let from = seq.n_live;
        if from >= goal {
            return None;
        }
        let m = (goal - from).min(self.chunk);
        let drafting = want > 0 && from + m == goal;
        // Room for the tokens and, to draft, the throwaway tokens past them.
        let need = from + m + if drafting { self.k } else { 0 };
        let have = seq.table.num_tokens() as usize;
        if need > have {
            seq.table
                .append(&mut self.pool, (need - have) as u32)
                .ok()?;
            seq.blocks = seq.table.blocks().iter().map(|b| b.0).collect();
        }
        Some(Plan {
            idx,
            seq: r.seq,
            from,
            m,
            want: if drafting { want } else { 0 },
        })
    }
}

impl ModelDrafter for DraftModel {
    fn hidden_layers(&self) -> Vec<usize> {
        Vec::new()
    }

    fn step(&mut self, batch: &mut [DraftRequest<'_>]) -> Result<(), String> {
        let mut plans: Vec<Plan> = Vec::new();
        for (idx, r) in batch.iter_mut().enumerate() {
            r.out.tokens.clear();
            r.out.probs.clear();
            if let Some(p) = self.plan(idx, r) {
                plans.push(p);
            }
        }
        if plans.is_empty() {
            return Ok(());
        }
        // Drafting sequences first, so the decode steps of the drafting loop run one group
        // over a prefix of the work slots; equal lengths together, one group each.
        plans.sort_by_key(|p| (p.want == 0, p.m, p.idx));
        let n_drafting = plans.iter().take_while(|p| p.want > 0).count();
        let max_seqs = self.max_seqs;
        let work = move |i: usize| max_seqs + i;

        // Live -> work.
        let copies: Vec<_> = plans
            .iter()
            .enumerate()
            .map(|(i, p)| (Pool::Seq, self.seqs[&p.seq].slot, Pool::Seq, work(i)))
            .collect();
        self.exec.copy_states(&copies).map_err(|e| e.to_string())?;

        // Catch up.
        let mut groups = Vec::new();
        let mut i = 0;
        while i < plans.len() {
            let m = plans[i].m;
            let n = plans[i..].iter().take_while(|p| p.m == m).count();
            groups.push(Group {
                slot: work(i),
                seqs: plans[i..i + n]
                    .iter()
                    .map(|p| SeqInput {
                        tokens: &batch[p.idx].tokens[p.from..p.from + m],
                        n_past: p.from,
                        blocks: &self.seqs[&p.seq].blocks,
                        ring: 0,
                        logits: if p.want > 0 {
                            Logits::Greedy
                        } else {
                            Logits::None
                        },
                        media: None,
                    })
                    .collect(),
                verify: false,
            });
            i += n;
        }
        let out = self.exec.step(&groups).map_err(|e| e.to_string())?;
        drop(groups);
        let mut drafts: Vec<Vec<u32>> = out.argmax.iter().map(|&t| vec![t]).collect();
        debug_assert_eq!(drafts.len(), n_drafting);

        // Work -> live: the caught-up state is the sequence's now.
        let copies: Vec<_> = copies.iter().map(|&(a, s, b, d)| (b, d, a, s)).collect();
        self.exec.copy_states(&copies).map_err(|e| e.to_string())?;
        for p in &plans {
            self.seqs.get_mut(&p.seq).unwrap().n_live = p.goal();
        }

        // Draft: the work slots, now throwaway, advance by the drafts. Sequences wanting
        // fewer drafts ride along; their extra drafts are dropped below.
        let rounds = plans[..n_drafting]
            .iter()
            .map(|p| p.want)
            .max()
            .unwrap_or(0);
        for j in 1..rounds {
            let last: Vec<u32> = drafts.iter().map(|d| d[j - 1]).collect();
            let groups = [Group {
                slot: work(0),
                seqs: plans[..n_drafting]
                    .iter()
                    .zip(&last)
                    .map(|(p, t)| SeqInput {
                        tokens: std::slice::from_ref(t),
                        n_past: p.goal() + j - 1,
                        blocks: &self.seqs[&p.seq].blocks,
                        ring: 0,
                        logits: Logits::Greedy,
                        media: None,
                    })
                    .collect(),
                verify: false,
            }];
            let out = self.exec.step(&groups).map_err(|e| e.to_string())?;
            for (d, t) in drafts.iter_mut().zip(out.argmax) {
                d.push(t);
            }
        }

        for (p, d) in plans[..n_drafting].iter().zip(drafts) {
            let out = &mut batch[p.idx].out;
            out.tokens.extend(
                d.into_iter()
                    .take(p.want)
                    .take_while(|&t| (t as usize) < self.n_vocab),
            );
        }
        Ok(())
    }

    fn release(&mut self, seq: SeqId) {
        if let Some(s) = self.seqs.remove(&seq) {
            s.table.release(&mut self.pool);
            self.free.push(s.slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_hash_covers_every_string_and_the_boundaries() {
        let a = VocabId::of_tokens(["a", "b", "c"]);
        assert_eq!(a, VocabId::of_tokens(["a", "b", "c"]));
        assert_eq!(a.count, 3);
        // Same count, one string differs.
        assert_ne!(a, VocabId::of_tokens(["a", "b", "d"]));
        // Same bytes, other boundaries.
        assert_ne!(
            VocabId::of_tokens(["ab", "c"]),
            VocabId::of_tokens(["a", "bc"])
        );
        // Order matters: ids are positions.
        assert_ne!(a, VocabId::of_tokens(["b", "a", "c"]));
        // Fewer tokens.
        assert_ne!(a.count, VocabId::of_tokens(["a", "b"]).count);
    }
}
