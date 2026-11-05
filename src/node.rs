//! A model split over two machines (#562): `legion serve --node-rank 0` runs the token
//! embeddings and the layers before `--node-split`, `--node-rank 1` the layers from there on and
//! the output head. Rank 0 serves the HTTP API and drives the step loop; for each step it sends
//! rank 1 the step's sequences (positions, KV block tables, which logits) and the residual
//! stream its last layer wrote, and rank 1 answers with the logits rows or argmaxes. State
//! changes rank 1 must mirror (a slot cleared, states copied) go the same way, in order.
//!
//! The transport is plain TCP (`std::net`) with `TCP_NODELAY`, one connection, frames of
//! `[kind u32][length u64][payload]`, little-endian. Rank 0 queues its frames to a writer
//! thread, so a prompt chunk's residual (tens of MB) goes out while it computes the next
//! chunk; every request gets one reply, in order, which rank 0 reads only when it needs one
//! (a step with logits): prompt chunks without logits run on rank 1 while rank 0 computes the
//! next one. Rank 1 reads frames on a reader thread for the same reason.

pub mod local;
#[cfg(test)]
mod tests;

use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::ops::Range;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Frames before this version are refused: both ranks must run the same build.
const VERSION: u32 = 3;

/// The node settings of a `legion serve` (`--node-rank`, `--node-peer`, `--node-split`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeConfig {
    /// 0: embeddings and the first layers, the HTTP API; 1: the last layers and the head.
    pub rank: usize,
    /// Rank 1's address: rank 1 listens on it, rank 0 connects to it.
    pub peer: String,
    /// Rank 1's first layer; `None`: balanced by the layers' weight bytes.
    pub split: Option<usize>,
}

impl NodeConfig {
    /// This rank's layers of `n_layer` for a split at layer `split`.
    pub fn share(&self, n_layer: usize, split: usize) -> Range<usize> {
        if self.rank == 0 {
            0..split
        } else {
            split..n_layer
        }
    }
}

/// One sequence of a step on the wire (see [`crate::model::core::SeqInput`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireSeq {
    pub tokens: Vec<u32>,
    pub n_past: u32,
    pub blocks: Vec<u32>,
    pub ring: u32,
    /// [`crate::model::core::Logits`] as a tag and its count.
    pub logits: (u8, u32),
}

/// A group of a step on the wire (see [`crate::model::core::Group`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireGroup {
    pub slot: u32,
    pub verify: bool,
    pub seqs: Vec<WireSeq>,
}

/// What rank 0 asks of rank 1.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// The first frame: the model's layer count, the split and rank 0's KV blocks.
    Hello {
        version: u32,
        n_layer: u32,
        split: u32,
        n_blocks: u32,
    },
    /// Resize the KV pool to this many blocks (the smaller of both ranks' fits).
    Resize {
        n_blocks: u32,
    },
    /// One step: its groups and the residual rows entering rank 1's first layer.
    Step {
        groups: Vec<WireGroup>,
        rows: Vec<f32>,
    },
    ClearSlot {
        slot: u32,
    },
    /// `(from pool, slot, to pool, slot)`, pools as 0 (running) / 1 (saved).
    CopyStates {
        copies: Vec<(u8, u32, u8, u32)>,
    },
    /// Tensor parallelism over machines ([`crate::tp_net`]), rank 0 to each other rank: the
    /// first frame on the control connection (answered with [`Reply::Hello`]).
    TpHello {
        version: u32,
        n_ranks: u32,
        rank: u32,
        n_layer: u32,
        n_blocks: u32,
    },
    /// The same step as rank 0's, with the executor settings that cut a step into passes
    /// (each pass's all-reduces must line up on every rank): the micro-chunk and its taper,
    /// and the gather budget.
    TpStep {
        groups: Vec<WireGroup>,
        micro: u32,
        taper: u32,
        gather_budget: u64,
    },
    /// Save ring `ring`'s first `n` positions in copy `slot`, or (`restore`) put that copy
    /// back.
    SaveRing {
        ring: u32,
        slot: u32,
        n: u32,
        restore: bool,
    },
    /// Make room to save `slots` copies of rings.
    EnableRingSaves {
        slots: u32,
    },
    /// The same MTP pass as rank 0's (the head's attention and FFN are split like the
    /// target's, so its all-reduces line up on every rank): its groups, the gather budget that
    /// cuts it into passes, and the residual rows it reads (`[tokens][width]`).
    TpMtp {
        groups: Vec<WireGroup>,
        gather_budget: u64,
        h: Vec<f32>,
    },
    /// A draft round of MTP passes (`--tp-mtp-chain`, #771): the first pass as in
    /// [`Self::TpMtp`], then the further passes every rank runs on its own from the head's
    /// picks ([`crate::model::core::Executor::mtp_chain`]): per output row of the first pass,
    /// the most drafts it may make (`caps`), and the floor (`p_min`, as bits) under which a
    /// row stops.
    TpMtpChain {
        groups: Vec<WireGroup>,
        gather_budget: u64,
        h: Vec<f32>,
        caps: Vec<u32>,
        p_min: u32,
    },
    /// [`Self::TpMtp`] whose residual rows come in `parts`, one after another
    /// (`--tp-mtp-local-rows`, #780): sent ones are the next floats of `h`, the others a range
    /// of what one of this rank's last two [`Self::TpStep`]s captured, the same bits as rank
    /// 0's.
    TpMtpRows {
        groups: Vec<WireGroup>,
        gather_budget: u64,
        parts: Vec<RowsPart>,
        h: Vec<f32>,
    },
    /// The same commit of the recurrent states as rank 0's after a verify step: `(slot,
    /// keep)` pairs ([`crate::model::core::exec::Executor::commit`]).
    TpCommit {
        commits: Vec<(u32, u32)>,
    },
}

/// A piece of an MTP pass's residual rows ([`Request::TpMtpRows`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowsPart {
    /// The next `len` floats of the request's `h`.
    Sent { len: u64 },
    /// Floats `start..start + len` of captured layer `layer` (its index among the captured
    /// layers, [`crate::model::core::StepOutput::hidden`]) of the receiving rank's last step
    /// (`back` 0) or the step before it (`back` 1). `check`: [`rows_check`] of rank 0's copy
    /// (debug builds), 0 for none.
    Step {
        back: u32,
        layer: u32,
        start: u64,
        len: u64,
        check: u64,
    },
}

/// A hash of `v`'s bits (never 0), which a debug build's ranks compare to show that the rows
/// they each captured are the same ([`RowsPart::Step`]).
pub fn rows_check(v: &[f32]) -> u64 {
    // FNV-1a over the values' bits.
    let h = v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3)
    });
    h | 1
}

/// Rank 1's answer to each request, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Reply {
    Hello {
        n_blocks: u32,
    },
    Done,
    /// A step's output ([`crate::model::core::StepOutput`]) and rank 1's milliseconds on it.
    Step {
        argmax: Vec<u32>,
        rows: Vec<Vec<f32>>,
        ms: f32,
    },
    Error(String),
}

const K_HELLO: u32 = 1;
const K_RESIZE: u32 = 2;
const K_STEP: u32 = 3;
const K_CLEAR: u32 = 4;
const K_COPY: u32 = 5;
const K_TP_HELLO: u32 = 6;
const K_TP_STEP: u32 = 7;
const K_SAVE_RING: u32 = 8;
const K_RING_SAVES: u32 = 9;
const K_TP_MTP: u32 = 10;
const K_TP_COMMIT: u32 = 11;
const K_TP_MTP_CHAIN: u32 = 12;
const K_TP_MTP_ROWS: u32 = 13;
const R_HELLO: u32 = 101;
const R_DONE: u32 = 102;
const R_STEP: u32 = 103;
const R_ERROR: u32 = 104;

/// A frame being built: its kind, a header and an optional bulk of floats sent as is.
pub struct Frame {
    kind: u32,
    head: Vec<u8>,
    bulk: Vec<f32>,
}

fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_u32s(b: &mut Vec<u8>, v: &[u32]) {
    put_u32(b, v.len() as u32);
    for &x in v {
        put_u32(b, x);
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and every bit pattern of the bytes is valid to read as u8;
    // the platforms Legion runs on (x86-64, aarch64, ppc64le) are little-endian, which the
    // frames are.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn put_groups(head: &mut Vec<u8>, groups: &[WireGroup]) {
    put_u32(head, groups.len() as u32);
    for g in groups {
        put_u32(head, g.slot);
        head.push(u8::from(g.verify));
        put_u32(head, g.seqs.len() as u32);
        for s in &g.seqs {
            put_u32s(head, &s.tokens);
            put_u32(head, s.n_past);
            put_u32s(head, &s.blocks);
            put_u32(head, s.ring);
            head.push(s.logits.0);
            put_u32(head, s.logits.1);
        }
    }
}

fn get_groups(r: &mut Cursor<'_>) -> Result<Vec<WireGroup>, String> {
    let n = r.u32()? as usize;
    let mut groups = Vec::with_capacity(n.min(1 << 16));
    for _ in 0..n {
        let slot = r.u32()?;
        let verify = r.u8()? != 0;
        let ns = r.u32()? as usize;
        let mut seqs = Vec::with_capacity(ns.min(1 << 16));
        for _ in 0..ns {
            seqs.push(WireSeq {
                tokens: r.u32s()?,
                n_past: r.u32()?,
                blocks: r.u32s()?,
                ring: r.u32()?,
                logits: (r.u8()?, r.u32()?),
            });
        }
        groups.push(WireGroup { slot, verify, seqs });
    }
    Ok(groups)
}

impl Request {
    pub fn encode(self) -> Frame {
        let mut head = Vec::new();
        let mut bulk = Vec::new();
        let kind = match self {
            Request::Hello {
                version,
                n_layer,
                split,
                n_blocks,
            } => {
                for v in [version, n_layer, split, n_blocks] {
                    put_u32(&mut head, v);
                }
                K_HELLO
            }
            Request::Resize { n_blocks } => {
                put_u32(&mut head, n_blocks);
                K_RESIZE
            }
            Request::Step { groups, rows } => {
                put_groups(&mut head, &groups);
                bulk = rows;
                K_STEP
            }
            Request::ClearSlot { slot } => {
                put_u32(&mut head, slot);
                K_CLEAR
            }
            Request::CopyStates { copies } => {
                put_u32(&mut head, copies.len() as u32);
                for (a, x, b, y) in copies {
                    head.push(a);
                    put_u32(&mut head, x);
                    head.push(b);
                    put_u32(&mut head, y);
                }
                K_COPY
            }
            Request::TpHello {
                version,
                n_ranks,
                rank,
                n_layer,
                n_blocks,
            } => {
                for v in [version, n_ranks, rank, n_layer, n_blocks] {
                    put_u32(&mut head, v);
                }
                K_TP_HELLO
            }
            Request::TpStep {
                groups,
                micro,
                taper,
                gather_budget,
            } => {
                put_groups(&mut head, &groups);
                put_u32(&mut head, micro);
                put_u32(&mut head, taper);
                head.extend_from_slice(&gather_budget.to_le_bytes());
                K_TP_STEP
            }
            Request::SaveRing {
                ring,
                slot,
                n,
                restore,
            } => {
                put_u32(&mut head, ring);
                put_u32(&mut head, slot);
                put_u32(&mut head, n);
                head.push(u8::from(restore));
                K_SAVE_RING
            }
            Request::EnableRingSaves { slots } => {
                put_u32(&mut head, slots);
                K_RING_SAVES
            }
            Request::TpMtp {
                groups,
                gather_budget,
                h,
            } => {
                put_groups(&mut head, &groups);
                head.extend_from_slice(&gather_budget.to_le_bytes());
                bulk = h;
                K_TP_MTP
            }
            Request::TpMtpChain {
                groups,
                gather_budget,
                h,
                caps,
                p_min,
            } => {
                put_groups(&mut head, &groups);
                head.extend_from_slice(&gather_budget.to_le_bytes());
                put_u32s(&mut head, &caps);
                put_u32(&mut head, p_min);
                bulk = h;
                K_TP_MTP_CHAIN
            }
            Request::TpMtpRows {
                groups,
                gather_budget,
                parts,
                h,
            } => {
                put_groups(&mut head, &groups);
                head.extend_from_slice(&gather_budget.to_le_bytes());
                put_u32(&mut head, parts.len() as u32);
                for p in parts {
                    match p {
                        RowsPart::Sent { len } => {
                            head.push(0);
                            head.extend_from_slice(&len.to_le_bytes());
                        }
                        RowsPart::Step {
                            back,
                            layer,
                            start,
                            len,
                            check,
                        } => {
                            head.push(1);
                            put_u32(&mut head, back);
                            put_u32(&mut head, layer);
                            for v in [start, len, check] {
                                head.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                    }
                }
                bulk = h;
                K_TP_MTP_ROWS
            }
            Request::TpCommit { commits } => {
                put_u32(&mut head, commits.len() as u32);
                for (slot, keep) in commits {
                    put_u32(&mut head, slot);
                    put_u32(&mut head, keep);
                }
                K_TP_COMMIT
            }
        };
        Frame { kind, head, bulk }
    }

    fn decode(kind: u32, head: &[u8], bulk: Vec<f32>) -> Result<Self, String> {
        let mut r = Cursor { b: head, at: 0 };
        let req = match kind {
            K_HELLO => Request::Hello {
                version: r.u32()?,
                n_layer: r.u32()?,
                split: r.u32()?,
                n_blocks: r.u32()?,
            },
            K_RESIZE => Request::Resize { n_blocks: r.u32()? },
            K_STEP => Request::Step {
                groups: get_groups(&mut r)?,
                rows: bulk,
            },
            K_CLEAR => Request::ClearSlot { slot: r.u32()? },
            K_COPY => {
                let n = r.u32()? as usize;
                let mut copies = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    copies.push((r.u8()?, r.u32()?, r.u8()?, r.u32()?));
                }
                Request::CopyStates { copies }
            }
            K_TP_HELLO => Request::TpHello {
                version: r.u32()?,
                n_ranks: r.u32()?,
                rank: r.u32()?,
                n_layer: r.u32()?,
                n_blocks: r.u32()?,
            },
            K_TP_STEP => Request::TpStep {
                groups: get_groups(&mut r)?,
                micro: r.u32()?,
                taper: r.u32()?,
                gather_budget: r.u64()?,
            },
            K_SAVE_RING => Request::SaveRing {
                ring: r.u32()?,
                slot: r.u32()?,
                n: r.u32()?,
                restore: r.u8()? != 0,
            },
            K_RING_SAVES => Request::EnableRingSaves { slots: r.u32()? },
            K_TP_MTP => Request::TpMtp {
                groups: get_groups(&mut r)?,
                gather_budget: r.u64()?,
                h: bulk,
            },
            K_TP_MTP_CHAIN => Request::TpMtpChain {
                groups: get_groups(&mut r)?,
                gather_budget: r.u64()?,
                caps: r.u32s()?,
                p_min: r.u32()?,
                h: bulk,
            },
            K_TP_MTP_ROWS => {
                let groups = get_groups(&mut r)?;
                let gather_budget = r.u64()?;
                let n = r.u32()? as usize;
                let mut parts = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    parts.push(match r.u8()? {
                        0 => RowsPart::Sent { len: r.u64()? },
                        1 => RowsPart::Step {
                            back: r.u32()?,
                            layer: r.u32()?,
                            start: r.u64()?,
                            len: r.u64()?,
                            check: r.u64()?,
                        },
                        t => return Err(format!("unknown rows part {t}")),
                    });
                }
                Request::TpMtpRows {
                    groups,
                    gather_budget,
                    parts,
                    h: bulk,
                }
            }
            K_TP_COMMIT => {
                let n = r.u32()? as usize;
                let mut commits = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    commits.push((r.u32()?, r.u32()?));
                }
                Request::TpCommit { commits }
            }
            k => return Err(format!("unknown request kind {k}")),
        };
        r.end()?;
        Ok(req)
    }
}

impl Reply {
    pub fn encode(self) -> Frame {
        let mut head = Vec::new();
        let mut bulk = Vec::new();
        let kind = match self {
            Reply::Hello { n_blocks } => {
                put_u32(&mut head, n_blocks);
                R_HELLO
            }
            Reply::Done => R_DONE,
            Reply::Step { argmax, rows, ms } => {
                put_u32s(&mut head, &argmax);
                head.extend_from_slice(&ms.to_le_bytes());
                put_u32(&mut head, rows.len() as u32);
                for r in &rows {
                    put_u32(&mut head, r.len() as u32);
                }
                bulk = rows.concat();
                R_STEP
            }
            Reply::Error(e) => {
                head.extend_from_slice(e.as_bytes());
                R_ERROR
            }
        };
        Frame { kind, head, bulk }
    }

    fn decode(kind: u32, head: &[u8], bulk: Vec<f32>) -> Result<Self, String> {
        let mut r = Cursor { b: head, at: 0 };
        let reply = match kind {
            R_HELLO => Reply::Hello { n_blocks: r.u32()? },
            R_DONE => Reply::Done,
            R_STEP => {
                let argmax = r.u32s()?;
                let ms = f32::from_le_bytes(r.take(4)?.try_into().expect("4 bytes"));
                let n = r.u32()? as usize;
                let mut lens = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    lens.push(r.u32()? as usize);
                }
                if lens.iter().sum::<usize>() != bulk.len() {
                    return Err("step reply rows do not add up".into());
                }
                let mut rows = Vec::with_capacity(n);
                let mut at = 0;
                for l in lens {
                    rows.push(bulk[at..at + l].to_vec());
                    at += l;
                }
                Reply::Step { argmax, rows, ms }
            }
            R_ERROR => {
                let e = String::from_utf8_lossy(head).into_owned();
                r.at = head.len();
                Reply::Error(e)
            }
            k => return Err(format!("unknown reply kind {k}")),
        };
        r.end()?;
        Ok(reply)
    }
}

struct Cursor<'b> {
    b: &'b [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or("frame too short")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
    fn u32s(&mut self) -> Result<Vec<u32>, String> {
        let n = self.u32()? as usize;
        let b = self.take(n.checked_mul(4).ok_or("frame too short")?)?;
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }
    fn end(&self) -> Result<(), String> {
        if self.at == self.b.len() {
            Ok(())
        } else {
            Err("trailing bytes in a frame".into())
        }
    }
}

/// Writes `f`: kind, payload length, header length, header, then the floats.
pub fn write_frame(w: &mut impl Write, f: &Frame) -> std::io::Result<()> {
    let bulk = f32_bytes(&f.bulk);
    w.write_all(&f.kind.to_le_bytes())?;
    w.write_all(&(f.head.len() as u64).to_le_bytes())?;
    w.write_all(&(bulk.len() as u64).to_le_bytes())?;
    w.write_all(&f.head)?;
    w.write_all(bulk)?;
    w.flush()
}

/// A frame as read: its kind, head and bulk floats.
pub type RawFrame = (u32, Vec<u8>, Vec<f32>);

/// Reads one frame: its kind, header and floats. `Ok(None)` at a clean end of the stream.
pub fn read_frame(r: &mut impl Read) -> std::io::Result<Option<RawFrame>> {
    let mut kind = [0u8; 4];
    match r.read_exact(&mut kind) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let mut n = [0u8; 8];
    r.read_exact(&mut n)?;
    let head_len = u64::from_le_bytes(n) as usize;
    r.read_exact(&mut n)?;
    let bulk_len = u64::from_le_bytes(n) as usize;
    if !bulk_len.is_multiple_of(4) || head_len > 1 << 30 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "malformed frame",
        ));
    }
    let mut head = vec![0u8; head_len];
    r.read_exact(&mut head)?;
    let mut bulk = vec![0f32; bulk_len / 4];
    // SAFETY: as in `f32_bytes`; any bytes are a valid f32.
    let bytes = unsafe { std::slice::from_raw_parts_mut(bulk.as_mut_ptr().cast::<u8>(), bulk_len) };
    r.read_exact(bytes)?;
    Ok(Some((u32::from_le_bytes(kind), head, bulk)))
}

fn invalid(e: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
}

/// Per-step time on the link, for the log (rank 0's view).
#[derive(Debug, Default, Clone, Copy)]
pub struct LinkTimes {
    /// Steps that waited for their reply.
    pub steps: u64,
    /// Rank 1's own milliseconds on them.
    pub remote_ms: f64,
    /// From the request queued to its reply read, less rank 1's own time: the wire, the
    /// (de)serialization and any earlier requests still queued.
    pub link_ms: f64,
}

/// Rank 0's end of the connection.
pub struct Link {
    tx: Option<mpsc::Sender<Frame>>,
    writer: Option<std::thread::JoinHandle<std::io::Result<()>>>,
    rd: BufReader<TcpStream>,
    /// Replies not read yet, with when their request was queued.
    pending: std::collections::VecDeque<Instant>,
    pub times: LinkTimes,
}

impl Link {
    /// Connects to rank 1 at `addr`, retrying every second for up to `wait` (rank 1 may still
    /// be loading its layers).
    pub fn connect(addr: &str, wait: Duration) -> std::io::Result<Self> {
        let t0 = Instant::now();
        let mut logged = false;
        let stream = loop {
            let tried = addr
                .to_socket_addrs()
                .and_then(|mut a| {
                    a.next().ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::NotFound, "no address")
                    })
                })
                .and_then(|a| TcpStream::connect_timeout(&a, Duration::from_secs(5)));
            match tried {
                Ok(s) => break s,
                Err(e) if t0.elapsed() < wait => {
                    if !logged {
                        tracing::info!(peer = addr, error = %e, "waiting for node rank 1");
                        logged = true;
                    }
                    std::thread::sleep(Duration::from_secs(1));
                }
                Err(e) => return Err(e),
            }
        };
        Self::over(stream)
    }

    /// Rank 0's end over a connected stream.
    pub fn over(stream: TcpStream) -> std::io::Result<Self> {
        stream.set_nodelay(true)?;
        let out = stream.try_clone()?;
        let (tx, rx) = mpsc::channel::<Frame>();
        let writer = std::thread::Builder::new()
            .name("legion-node-tx".into())
            .spawn(move || {
                let mut w = BufWriter::with_capacity(1 << 20, out);
                for f in rx {
                    write_frame(&mut w, &f)?;
                }
                Ok(())
            })?;
        Ok(Self {
            tx: Some(tx),
            writer: Some(writer),
            rd: BufReader::with_capacity(1 << 20, stream),
            pending: Default::default(),
            times: LinkTimes::default(),
        })
    }

    /// Queues `req`; its reply is read by a later [`Self::wait`].
    pub fn send(&mut self, req: Request) -> Result<(), String> {
        let tx = self.tx.as_ref().ok_or("the node link is closed")?;
        if tx.send(req.encode()).is_err() {
            return Err(self.writer_error());
        }
        self.pending.push_back(Instant::now());
        Ok(())
    }

    fn writer_error(&mut self) -> String {
        self.tx = None;
        match self.writer.take().map(|w| w.join()) {
            Some(Ok(Err(e))) => format!("sending to node rank 1: {e}"),
            _ => "the node link's writer stopped".into(),
        }
    }

    fn read_reply(&mut self) -> Result<Reply, String> {
        match read_frame(&mut self.rd) {
            Ok(Some((k, h, b))) => Reply::decode(k, &h, b),
            Ok(None) => Err("node rank 1 closed the connection".into()),
            Err(e) => Err(format!("reading from node rank 1: {e}")),
        }
    }

    /// Reads every reply still owed and returns the last one; an error reply (of any of
    /// them) is the error.
    pub fn wait(&mut self) -> Result<Reply, String> {
        let mut last = None;
        let mut err = None;
        while let Some(sent) = self.pending.pop_front() {
            let r = self.read_reply()?;
            if let Reply::Step { ms, .. } = &r
                && self.pending.is_empty()
            {
                self.times.steps += 1;
                self.times.remote_ms += f64::from(*ms);
                self.times.link_ms +=
                    (sent.elapsed().as_secs_f64() * 1e3 - f64::from(*ms)).max(0.0);
            }
            match r {
                Reply::Error(e) => err = err.or(Some(e)),
                r => last = Some(r),
            }
        }
        match err {
            Some(e) => Err(format!("node rank 1: {e}")),
            None => last.ok_or_else(|| "no request to wait for".into()),
        }
    }

    /// Sends `req` and waits for its reply (and every one before).
    pub fn call(&mut self, req: Request) -> Result<Reply, String> {
        self.send(req)?;
        self.wait()
    }

    /// Replies still owed.
    pub fn owed(&self) -> usize {
        self.pending.len()
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
    }
}

/// Rank 1's side: accepts rank 0 on `listener`, one connection at a time, and answers its
/// requests with `handle` until it disconnects; then waits for the next one. Returns only on
/// a listener error.
pub fn serve(
    listener: TcpListener,
    mut handle: impl FnMut(Request) -> Reply,
) -> std::io::Result<()> {
    loop {
        let (stream, from) = listener.accept()?;
        tracing::info!(%from, "node rank 0 connected");
        if let Err(e) = serve_one(stream, &mut handle) {
            tracing::warn!(error = %e, "node link ended with an error");
        } else {
            tracing::info!("node rank 0 disconnected");
        }
    }
}

/// Answers one connection's requests (see [`serve`]).
pub fn serve_one(
    stream: TcpStream,
    handle: &mut impl FnMut(Request) -> Reply,
) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let input = stream.try_clone()?;
    // A few frames ahead: the next prompt chunk's residual arrives while this one computes.
    let (tx, rx) = mpsc::sync_channel::<std::io::Result<Request>>(4);
    let reader = std::thread::Builder::new()
        .name("legion-node-rx".into())
        .spawn(move || {
            let mut r = BufReader::with_capacity(1 << 20, input);
            loop {
                let item = match read_frame(&mut r) {
                    Ok(Some((k, h, b))) => Request::decode(k, &h, b).map_err(invalid),
                    Ok(None) => return,
                    Err(e) => Err(e),
                };
                let stop = item.is_err();
                if tx.send(item).is_err() || stop {
                    return;
                }
            }
        })?;
    let mut w = BufWriter::with_capacity(1 << 20, stream);
    let mut out = Ok(());
    for req in rx {
        let req = match req {
            Ok(r) => r,
            Err(e) => {
                out = Err(e);
                break;
            }
        };
        if let Err(e) = write_frame(&mut w, &handle(req).encode()) {
            out = Err(e);
            break;
        }
    }
    // The reader ends at the stream's end (or when it can no longer hand on a frame).
    let _ = w.get_ref().shutdown(std::net::Shutdown::Both);
    let _ = reader.join();
    out
}

/// The `Hello` rank 0 sends first.
pub fn hello(n_layer: usize, split: usize, n_blocks: usize) -> Request {
    Request::Hello {
        version: VERSION,
        n_layer: n_layer as u32,
        split: split as u32,
        n_blocks: n_blocks as u32,
    }
}

/// The `TpHello` rank 0 sends each other rank of tensor parallelism over machines.
pub fn tp_hello(n_ranks: usize, rank: usize, n_layer: usize, n_blocks: usize) -> Request {
    Request::TpHello {
        version: VERSION,
        n_ranks: n_ranks as u32,
        rank: rank as u32,
        n_layer: n_layer as u32,
        n_blocks: n_blocks as u32,
    }
}

/// A rank's check of a `TpHello` against its own rank, rank count and model.
pub fn check_tp_hello(
    (version, n_ranks, rank, n_layer): (u32, u32, u32, u32),
    mine: (usize, usize, usize),
) -> Result<(), String> {
    if version != VERSION {
        return Err(format!(
            "node protocol version {version}, this build speaks {VERSION}: run the same legion on every rank"
        ));
    }
    let theirs = (n_ranks as usize, rank as usize, n_layer as usize);
    if theirs != mine {
        return Err(format!(
            "rank 0 expects {n_ranks} ranks, this one as rank {rank}, with {n_layer} layers; this \
             rank is rank {} of {} with {} layers: give every rank the same model and --tp-peers",
            mine.1, mine.0, mine.2
        ));
    }
    Ok(())
}

/// Rank 1's check of a `Hello` against its own model and split.
pub fn check_hello(
    version: u32,
    n_layer: u32,
    split: u32,
    mine: (usize, usize),
) -> Result<(), String> {
    if version != VERSION {
        return Err(format!(
            "node protocol version {version}, this build speaks {VERSION}: run the same legion on both nodes"
        ));
    }
    if (n_layer as usize, split as usize) != mine {
        return Err(format!(
            "rank 0 has {n_layer} layers split at {split}, rank 1 {} split at {}: give both the same model and --node-split",
            mine.0, mine.1
        ));
    }
    Ok(())
}
