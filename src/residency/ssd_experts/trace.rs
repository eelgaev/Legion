//! The SSD tier's routing trace (`LEGION_SSD_EXPERT_TRACE=<path>`, a debug hook): every fetch's
//! routed expert ids, per layer and step, and the router lookahead's predictions of them, for
//! replaying slot policies and prefetches offline ([`super::replay`], `legion ssd-replay`).
//!
//! The file: the 4 bytes `LSXT`, little-endian u32 version (2; version 1 traces, without
//! predictions, read the same), the tier's layer count, the model's expert count and the tier's
//! slots per layer, then the tier's layer ids (u32 each). Then one record per fetch or
//! prediction, in the order they ran: u8 kind (0: a small step, its experts in slots; 1: a big
//! step, staged; 2: a prediction), u8 the predictor (a prediction's [`Predictor`], else 0), u16
//! the layer's ordinal in the tier, u32 the step's tokens, u32 the number of ids, and the ids as
//! u16. A fetch holds `n_used` ids per token in routing order; a prediction holds the same
//! number per token, ranked best first (the predictor's top experts, `n_used` plus a margin).

use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"LSXT";
const VERSION: u32 = 2;

/// Where a prediction of a layer's routing comes from (the record's predictor byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Predictor {
    /// The layer's router applied to the previous layer's MoE input (its post-attention
    /// hidden state, normed): the PILOT lookahead the tier's `--ssd-prefetch router` reads by.
    PostAttn,
    /// The layer's router applied to the previous layer's output (post-MoE), through this
    /// layer's own FFN mixer and norm (`--ssd-prefetch router-post`, `router-both`).
    PostMoe,
    /// The layer's router applied to the MTP head's hidden state of the step's draft.
    Mtp,
}

impl Predictor {
    pub const ALL: [Predictor; 3] = [Predictor::PostAttn, Predictor::PostMoe, Predictor::Mtp];

    pub fn code(self) -> u8 {
        self as u8
    }

    pub fn from_code(c: u8) -> Option<Self> {
        Self::ALL.get(c as usize).copied()
    }

    pub fn name(self) -> &'static str {
        match self {
            Predictor::PostAttn => "router(L+1) on L's post-attention state",
            Predictor::PostMoe => "router(L+1) on L's post-MoE state",
            Predictor::Mtp => "router(L) on the MTP draft's state",
        }
    }
}

/// The trace's header: what the tier was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// The tier's layers (model layer ids); records name them by ordinal.
    pub layers: Vec<usize>,
    pub n_expert: usize,
    pub n_slots: usize,
}

/// One fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// The layer's ordinal in [`Header::layers`].
    pub ord: usize,
    /// A big (staged) step: it reads its experts elsewhere and only fills empty slots.
    pub big: bool,
    pub n_tokens: usize,
    /// The routed ids, `n_used` per token.
    pub ids: Vec<u16>,
}

/// One prediction of a layer's routing, made before its fetch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prediction {
    /// The predicted layer's ordinal.
    pub ord: usize,
    pub predictor: Predictor,
    pub n_tokens: usize,
    /// Per token, `ids.len() / n_tokens` experts, best first.
    pub ids: Vec<u16>,
    /// The number of fetch records before it ([`Trace::records`] index of the next fetch).
    pub before: usize,
}

impl Prediction {
    /// Ids per token.
    pub fn width(&self) -> usize {
        self.ids.len() / self.n_tokens.max(1)
    }

    /// The distinct experts of each token's best `top`, rank by rank (every token's best, then
    /// every token's second, ...).
    pub fn top(&self, top: usize) -> Vec<usize> {
        let w = self.width();
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for r in 0..top.min(w) {
            for t in 0..self.n_tokens {
                let e = self.ids[t * w + r] as usize;
                if seen.insert(e) {
                    out.push(e);
                }
            }
        }
        out
    }
}

/// A trace being written.
pub struct Writer {
    out: BufWriter<std::fs::File>,
    failed: bool,
}

impl Writer {
    pub fn create(path: &Path, header: &Header) -> io::Result<Self> {
        let mut out = BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        out.write_all(&header_bytes(header))?;
        Ok(Self { out, failed: false })
    }

    /// Appends a fetch record; the first write error is logged and ends the trace.
    pub fn record(&mut self, ord: usize, big: bool, n_tokens: usize, ids: &[i32]) {
        self.write(big as u8, 0, ord, n_tokens, ids);
    }

    /// Appends a prediction of layer `ord`'s routing (`n_tokens` rows of ranked ids).
    pub fn predict(&mut self, ord: usize, p: Predictor, n_tokens: usize, ids: &[i32]) {
        self.write(2, p.code(), ord, n_tokens, ids);
    }

    fn write(&mut self, kind: u8, pred: u8, ord: usize, n_tokens: usize, ids: &[i32]) {
        if self.failed {
            return;
        }
        let mut b = Vec::with_capacity(12 + 2 * ids.len());
        b.push(kind);
        b.push(pred);
        b.extend_from_slice(&(ord as u16).to_le_bytes());
        b.extend_from_slice(&(n_tokens as u32).to_le_bytes());
        b.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        for &e in ids {
            b.extend_from_slice(&(e as u16).to_le_bytes());
        }
        if let Err(e) = self.out.write_all(&b) {
            tracing::warn!(error = %e, "SSD expert trace: write failed, trace ends here");
            self.failed = true;
        }
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.out.flush();
    }
}

fn header_bytes(h: &Header) -> Vec<u8> {
    let mut b = MAGIC.to_vec();
    for v in [
        VERSION,
        h.layers.len() as u32,
        h.n_expert as u32,
        h.n_slots as u32,
    ] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for &l in &h.layers {
        b.extend_from_slice(&(l as u32).to_le_bytes());
    }
    b
}

/// A whole trace, read back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trace {
    pub header: Header,
    pub records: Vec<Record>,
    /// The predictions (none in a version 1 trace).
    pub predictions: Vec<Prediction>,
}

impl Trace {
    pub fn read(path: &Path) -> io::Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        Self::parse(&bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })
    }

    /// Parses a trace; a record cut off at the end (a process that died mid-write) is dropped.
    pub fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < 20 || &b[..4] != MAGIC {
            return Err("not an SSD expert trace (no LSXT header)".into());
        }
        let u = |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().unwrap()) as usize;
        if !(1..=VERSION as usize).contains(&u(4)) {
            return Err(format!(
                "trace version {}, this reads versions 1 to {VERSION}",
                u(4)
            ));
        }
        let (n_layers, n_expert, n_slots) = (u(8), u(12), u(16));
        let mut at = 20;
        if b.len() < at + 4 * n_layers {
            return Err("the layer list is truncated".into());
        }
        let layers = (0..n_layers).map(|i| u(at + 4 * i)).collect();
        at += 4 * n_layers;
        let mut records = Vec::new();
        let mut predictions = Vec::new();
        while at + 12 <= b.len() {
            let kind = b[at];
            let ord = u16::from_le_bytes([b[at + 2], b[at + 3]]) as usize;
            let n_tokens = u(at + 4);
            let n = u(at + 8);
            let end = at + 12 + 2 * n;
            if end > b.len() {
                break;
            }
            if ord >= n_layers {
                return Err(format!("a record of layer ordinal {ord} of {n_layers}"));
            }
            let ids: Vec<u16> = b[at + 12..end]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| u16::from_le_bytes(*p))
                .collect();
            if let Some(&e) = ids.iter().find(|&&e| e as usize >= n_expert) {
                return Err(format!("expert id {e} of {n_expert}"));
            }
            match kind {
                0 | 1 => records.push(Record {
                    ord,
                    big: kind == 1,
                    n_tokens,
                    ids,
                }),
                2 => {
                    let predictor = Predictor::from_code(b[at + 1])
                        .ok_or_else(|| format!("predictor {} unknown", b[at + 1]))?;
                    if n_tokens == 0 || !ids.len().is_multiple_of(n_tokens) {
                        return Err(format!("a prediction of {} ids for {n_tokens} tokens", n));
                    }
                    predictions.push(Prediction {
                        ord,
                        predictor,
                        n_tokens,
                        ids,
                        before: records.len(),
                    });
                }
                k => return Err(format!("record kind {k} unknown")),
            }
            at = end;
        }
        Ok(Self {
            header: Header {
                layers,
                n_expert,
                n_slots,
            },
            records,
            predictions,
        })
    }
}
