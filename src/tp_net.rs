//! Tensor parallelism over several machines, one GPU each (#619): `legion serve --tp-rank R
//! --tp-peers a:p,b:p,...` runs every layer on all the ranks at once, each with its share of the
//! attention heads and FFN rows ([`shard`], the Megatron layout of [`crate::model::core::tp`]),
//! and sums the partial outputs of the attention and FFN output projections over the network
//! ([`all_reduce`], a graph op). Rank 0 serves the HTTP API and drives the steps; the other ranks
//! run the same steps for it ([`crate::engine::lm`]'s `tp_node`).
//!
//! The connections: every rank keeps one data connection to every other rank (rank `i`
//! dials rank `j > i` at `--tp-peers[j]`), plain TCP with `TCP_NODELAY`, set up before the
//! model loads ([`connect`]); rank 0 adds one control connection to each other rank once it
//! has loaded ([`dial_control`]), which carries the steps (the frames of [`crate::node`]). Each
//! connection starts with a preamble (magic, protocol version, kind, the dialer's rank and the
//! rank count) that the far end checks and answers with its own.
//!
//! The all-reduce ([`Comm`]): per peer a writer thread sends this rank's partial and a reader
//! thread takes the peer's into a buffer, both in all-reduce order (a sequence number every
//! frame carries and the reader checks), and each rank adds every rank's partial in rank order,
//! so all of them get the same bits. On CUDA the stream never waits for the host: the GPU copies
//! its partial into pinned memory and raises a flag the writers poll, then waits (a stream
//! memory op) for the readers' flags and adds the partials in a kernel.

#[cfg(test)]
mod tests;

mod comm;
pub mod ep_place;
pub mod micro;
mod op;
mod profile;
mod rdma;
pub mod seq_par;
pub mod shard;

pub use profile::StepProfile;

pub use comm::{ArStats, Comm};
pub use op::all_reduce;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Both ends of every connection must speak this version (the same build on every rank).
/// 2: a data connection starts with the transport offers (`rdma::Offer`).
pub const VERSION: u32 = 2;
/// The most ranks the sum kernel adds (`src/cuda/tp_net.cu`).
pub const MAX_RANKS: usize = 8;
const MAGIC: u32 = u32::from_le_bytes(*b"LgTP");
const KIND_DATA: u32 = 1;
const KIND_CONTROL: u32 = 2;

/// How the all-reduce's partials travel between two ranks (`--tp-transport`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Transport {
    /// RDMA when both ends have an RDMA device (RoCE v2) on the addresses of their data
    /// connection and run on a GPU, else TCP.
    #[default]
    Auto,
    /// The TCP data connections.
    Tcp,
    /// RDMA writes; fails to start without it.
    Rdma,
}

/// The multi-machine tensor parallelism of a `legion serve` (`--tp-rank`, `--tp-peers`,
/// `--tp-timeout`, `--tp-transport`, `--tp-profile`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TpNetConfig {
    /// This machine's rank: 0 serves the API and drives the steps.
    pub rank: usize,
    /// Every rank's address, `ip:port`, in rank order: rank `r > 0` listens on entry `r`
    /// (rank 0's entry is not used: rank 0 only dials).
    pub peers: Vec<String>,
    /// How long a rank waits for a peer's partial (or its own GPU) before it gives the step up.
    pub timeout: Duration,
    /// How the partials travel.
    pub transport: Transport,
    /// Log where the steps' time went every this many steps (0: never).
    pub profile: usize,
    /// The type the partials travel in (`--tp-wire`).
    pub wire: Wire,
    /// Steps of at least this many tokens run the per-token work between the sums on a share
    /// of the tokens per rank (`--tp-seq-par`, [`seq_par`]); 0: never.
    pub seq_par: usize,
    /// Rank 0, `--spec mtp`: send a draft round's first MTP pass with the round's caps and
    /// confidence floor, and let every rank run the further passes on its own
    /// (`--tp-mtp-chain`, #771); off: one request per pass.
    pub mtp_chain: bool,
    /// Rank 0, `--spec mtp`: an MTP pass's rows that are a range of the last step's output
    /// (a prompt chunk's take-in) go to the other ranks as that range, and each reads its own
    /// copy (`--tp-mtp-local-rows`, #780); off: their values are sent.
    pub mtp_local_rows: bool,
    /// qwen4exp: each rank holds whole routed experts, its share of the expert ids, instead of
    /// every expert's share of the rows (`--tp-moe-ep`, #798; [`shard::qwen4::Experts`]).
    pub moe_ep: bool,
    /// Those steps in two micro-batches, one's exchanges on the wire while the other computes
    /// (`--tp-overlap`, [`micro`]).
    pub overlap: Overlap,
    /// With `moe_ep`: which rank holds each routed expert, per layer (`--tp-moe-ep-place`,
    /// #803, [`ep_place`]); `None`: the contiguous split.
    pub moe_ep_place: Option<std::sync::Arc<ep_place::Placement>>,
}

/// What a `--tp-seq-par` step runs in two micro-batches (`--tp-overlap`, [`micro`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Overlap {
    /// Nothing: one exchange at a time, the GPU waiting for each.
    #[default]
    Off,
    /// The hyper-connection mixers: each sum's reduce-scatter and the next all-gather go in two
    /// halves of every rank's tokens, one half's exchange on the wire while the other half's
    /// mixer computes.
    Mixers,
    /// The mixers and the MoE: also the routed experts run once per half, so each half's
    /// all-gather and reduce-scatter overlap the other half's experts (each half reads the
    /// expert weights again).
    Ffn,
}

/// The type the partials of the sums travel in (`--tp-wire`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Wire {
    /// F32, the type of the partials.
    #[default]
    F32,
    /// BF16: half the bytes, for sums of at least [`BF16_MIN_VALUES`] values (prompt steps;
    /// smaller ones, decode steps, stay F32: they are latency bound). Every rank's partial, its
    /// own included, is rounded to BF16 (nearest even), then they are added in F32 in rank
    /// order, so the ranks still get the same bits.
    Bf16,
}

/// The smallest sum (values of the whole tensor, 1 MiB in F32) that `--tp-wire bf16` sends in
/// BF16: about 100 tokens of Qwen3.8-Flash-Next, more than its decode steps.
pub const BF16_MIN_VALUES: usize = 1 << 18;

impl Wire {
    /// Bytes per element on the wire.
    pub fn bytes(self) -> usize {
        match self {
            Wire::F32 => 4,
            Wire::Bf16 => 2,
        }
    }
}

impl TpNetConfig {
    pub fn n_ranks(&self) -> usize {
        self.peers.len()
    }

    /// The settings' own consistency: at least two ranks, at most [`MAX_RANKS`], this rank
    /// among them.
    pub fn check(&self) -> Result<(), String> {
        let n = self.n_ranks();
        if !(2..=MAX_RANKS).contains(&n) {
            return Err(format!(
                "--tp-peers: {n} addresses; tensor parallelism over machines takes 2 to {MAX_RANKS}"
            ));
        }
        if self.rank >= n {
            return Err(format!(
                "--tp-rank {}: only {n} ranks in --tp-peers",
                self.rank
            ));
        }
        Ok(())
    }
}

fn invalid(e: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.into())
}

fn put(w: &mut impl Write, v: &[u32]) -> std::io::Result<()> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.write_all(&b)?;
    w.flush()
}

fn get<const N: usize>(r: &mut impl Read) -> std::io::Result<[u32; N]> {
    let mut b = [[0u8; 4]; N];
    for x in &mut b {
        r.read_exact(x)?;
    }
    Ok(b.map(u32::from_le_bytes))
}

/// A connection's preamble: `[magic, version, kind, rank, n_ranks]`.
fn send_preamble(s: &mut TcpStream, kind: u32, rank: usize, n: usize) -> std::io::Result<()> {
    put(s, &[MAGIC, VERSION, kind, rank as u32, n as u32])
}

/// Reads and checks a preamble: `(kind, rank)`.
fn read_preamble(s: &mut TcpStream, n: usize) -> std::io::Result<(u32, usize)> {
    let [magic, version, kind, rank, count] = get::<5>(s)?;
    if magic != MAGIC {
        return Err(invalid("not a legion tensor-parallel peer"));
    }
    if version != VERSION {
        return Err(invalid(format!(
            "a peer speaks tensor-parallel protocol {version}, this build {VERSION}: run the \
             same legion on every rank"
        )));
    }
    if count as usize != n {
        return Err(invalid(format!(
            "a peer has {count} ranks in --tp-peers, this rank {n}"
        )));
    }
    Ok((kind, rank as usize))
}

/// Dials `addr`, retrying every second for up to `wait` (the peer may still be starting).
fn dial(addr: &str, wait: Duration) -> std::io::Result<TcpStream> {
    let t0 = Instant::now();
    let mut logged = false;
    loop {
        let tried = addr
            .to_socket_addrs()
            .and_then(|mut a| {
                a.next()
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no address"))
            })
            .and_then(|a| TcpStream::connect_timeout(&a, Duration::from_secs(5)));
        match tried {
            Ok(s) => {
                s.set_nodelay(true)?;
                return Ok(s);
            }
            Err(e) if t0.elapsed() < wait => {
                if !logged {
                    tracing::info!(peer = addr, error = %e, "tensor parallel: waiting for a peer");
                    logged = true;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Dials rank `to` at `addr` as `kind`, and checks its answer.
fn dial_rank(
    addr: &str,
    kind: u32,
    rank: usize,
    to: usize,
    n: usize,
    wait: Duration,
) -> std::io::Result<TcpStream> {
    let mut s = dial(addr, wait)?;
    send_preamble(&mut s, kind, rank, n)?;
    s.set_read_timeout(Some(wait.max(Duration::from_secs(1))))?;
    let (k, r) = read_preamble(&mut s, n)?;
    s.set_read_timeout(None)?;
    if k != kind || r != to {
        return Err(invalid(format!(
            "{addr} answered as rank {r}, not rank {to}: check --tp-peers on every rank"
        )));
    }
    Ok(s)
}

/// This rank's connections to the others.
pub struct Mesh {
    pub rank: usize,
    pub n: usize,
    /// The data connection to each other rank, by rank (`None` at this rank's own index).
    pub data: Vec<Option<TcpStream>>,
    /// Where rank 0's control connection arrives (ranks above 0).
    pub control: Option<ControlListener>,
}

/// A rank's listener, after the data connections: rank 0's control connection comes next.
pub struct ControlListener {
    listener: TcpListener,
    rank: usize,
    n: usize,
    /// A control connection that came while the data connections were still being made.
    early: Option<TcpStream>,
}

impl ControlListener {
    /// The next control connection from rank 0, answered (rank 0 waits for the answer: this
    /// rank has loaded).
    pub fn accept(&mut self) -> std::io::Result<TcpStream> {
        if let Some(mut s) = self.early.take() {
            send_preamble(&mut s, KIND_CONTROL, self.rank, self.n)?;
            return Ok(s);
        }
        loop {
            let (mut s, from) = self.listener.accept()?;
            match read_preamble(&mut s, self.n) {
                Ok((KIND_CONTROL, 0)) => {
                    s.set_nodelay(true)?;
                    send_preamble(&mut s, KIND_CONTROL, self.rank, self.n)?;
                    return Ok(s);
                }
                Ok((kind, r)) => {
                    tracing::warn!(%from, kind, rank = r, "tensor parallel: refused a connection");
                }
                Err(e) => {
                    tracing::warn!(%from, error = %e, "tensor parallel: refused a connection")
                }
            }
        }
    }
}

/// Connects this rank to every other (see the module docs); waits up to `wait` for each rank
/// it dials, and for as long as it takes for the ranks that dial it.
pub fn connect(cfg: &TpNetConfig, wait: Duration) -> std::io::Result<Mesh> {
    cfg.check().map_err(invalid)?;
    let (rank, n) = (cfg.rank, cfg.n_ranks());
    // Listening first: the lower ranks dial this one while it dials the higher ones.
    let listener = if rank > 0 {
        Some(TcpListener::bind(&cfg.peers[rank]).map_err(|e| {
            std::io::Error::new(e.kind(), format!("--tp-peers {}: {e}", cfg.peers[rank]))
        })?)
    } else {
        None
    };
    let mut data: Vec<Option<TcpStream>> = (0..n).map(|_| None).collect();
    for (j, slot) in data.iter_mut().enumerate().skip(rank + 1) {
        *slot = Some(dial_rank(&cfg.peers[j], KIND_DATA, rank, j, n, wait)?);
    }
    let mut early = None;
    if let Some(l) = &listener {
        let mut need = rank;
        while need > 0 {
            let (mut s, from) = l.accept()?;
            let (kind, r) = match read_preamble(&mut s, n) {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(%from, error = %e, "tensor parallel: refused a connection");
                    continue;
                }
            };
            match kind {
                KIND_DATA if r < rank && data[r].is_none() => {
                    send_preamble(&mut s, KIND_DATA, rank, n)?;
                    s.set_nodelay(true)?;
                    data[r] = Some(s);
                    need -= 1;
                }
                KIND_CONTROL if r == 0 && early.is_none() => {
                    s.set_nodelay(true)?;
                    early = Some(s);
                }
                _ => tracing::warn!(%from, kind, rank = r, "tensor parallel: refused a connection"),
            }
        }
    }
    let control = listener.map(|listener| ControlListener {
        listener,
        rank,
        n,
        early,
    });
    Ok(Mesh {
        rank,
        n,
        data,
        control,
    })
}

/// Rank 0: the control connection to rank `to`; waits up to `wait` for it to answer (it
/// answers once it has loaded).
pub fn dial_control(cfg: &TpNetConfig, to: usize, wait: Duration) -> std::io::Result<TcpStream> {
    dial_rank(&cfg.peers[to], KIND_CONTROL, 0, to, cfg.n_ranks(), wait)
}

/// The communicators graphs refer to by index (an external op's params): see [`register`].
static COMMS: Mutex<Vec<Option<Arc<Comm>>>> = Mutex::new(Vec::new());

/// Makes `comm` reachable from graph ops by the returned id until the handle drops.
pub fn register(comm: Arc<Comm>) -> CommHandle {
    let mut c = COMMS.lock().expect("comms lock");
    let id = match c.iter().position(Option::is_none) {
        Some(i) => {
            c[i] = Some(comm.clone());
            i
        }
        None => {
            c.push(Some(comm.clone()));
            c.len() - 1
        }
    };
    CommHandle {
        id: id as i32,
        comm,
    }
}

/// The communicator `id`, while registered.
pub(crate) fn comm(id: i32) -> Option<Arc<Comm>> {
    let c = COMMS.lock().expect("comms lock");
    c.get(usize::try_from(id).ok()?).cloned().flatten()
}

/// A registered communicator: ops find it by [`Self::id`]; dropping the handle unregisters it
/// and stops its threads.
pub struct CommHandle {
    id: i32,
    comm: Arc<Comm>,
}

impl CommHandle {
    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn comm(&self) -> &Arc<Comm> {
        &self.comm
    }
}

impl Drop for CommHandle {
    fn drop(&mut self) {
        if let Ok(mut c) = COMMS.lock()
            && let Some(slot) = c.get_mut(self.id as usize)
        {
            *slot = None;
        }
        self.comm.shutdown();
    }
}
