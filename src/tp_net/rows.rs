//! The lookup-table rows connection (#1176): a rank above 0 asks rank 0 for the n-gram (or
//! Engram) table rows its row cache misses, and rank 0 reads them from its own disk and sends
//! their bytes back ([`crate::residency::lookup_table::PeerRows`]).
//!
//! Every rank needs the same rows every step, and on two DGX Sparks the second rank's copy of
//! the checkpoint is the first's disk over NFS, whose 4 KiB reads top out at ~108K a second at
//! any depth (the local disk: ~658K): a cold 4,096-token prompt step's ~60K page reads took
//! rank 1 ~530 ms against rank 0's ~110, and rank 0 waited for it at the step's first sum.
//! Over this connection the rows come at rank 0's disk speed plus ~10 MB on the link. The same
//! bytes either way; when an ask fails the asking rank reads the rows itself.
//!
//! Rank 0 dials one such connection to every other rank after the control connection
//! ([`super::dial_rows`]); each rank takes it on a thread of its own
//! ([`super::ControlListener::accept_rows`]). Frames, little endian:
//! - ask: `id: u64, urgent: u32, n: u32, rows: [u32; n]`;
//! - answer: `id: u64, status: u32 (0 rows, 1 error), len: u64, bytes: [u8; len]` (the rows'
//!   bytes in the order asked, or the error's text).
//!
//! Rank 0 answers asks on a few threads at once, in the order they finish: a decode step's
//! few rows do not wait behind a prompt chunk's.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use crate::residency::lookup_table::{PeerDone, PeerRows};

#[cfg(test)]
mod tests;

/// Rank 0's threads that answer asks.
const ANSWERERS: usize = 4;

/// The most rows one ask may name (a prompt step's 16 rows a token for 64K tokens).
const MAX_ROWS: usize = 1 << 20;

/// Reads rows for an ask: `(rows, urgent)` to their bytes (rank 0's
/// [`crate::residency::lookup_table::LookupTable::read_for_peer`]).
pub type ReadRows = Arc<dyn Fn(&[u32], bool) -> io::Result<Vec<u8>> + Send + Sync>;

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

/// One ask's frame.
fn ask_frame(id: u64, urgent: bool, rows: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(16 + 4 * rows.len());
    b.extend_from_slice(&id.to_le_bytes());
    b.extend_from_slice(&u32::from(urgent).to_le_bytes());
    b.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for r in rows {
        b.extend_from_slice(&r.to_le_bytes());
    }
    b
}

/// Reads one ask: `(id, urgent, rows)`.
fn read_ask(r: &mut impl Read) -> io::Result<(u64, bool, Vec<u32>)> {
    let mut head = [0u8; 16];
    r.read_exact(&mut head)?;
    let n = u32_at(&head, 12) as usize;
    if n > MAX_ROWS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("a rows ask of {n} rows"),
        ));
    }
    let mut b = vec![0u8; 4 * n];
    r.read_exact(&mut b)?;
    let rows = b.chunks_exact(4).map(|c| u32_at(c, 0)).collect();
    Ok((u64_at(&head, 0), u32_at(&head, 8) != 0, rows))
}

/// One answer's frame head (its bytes follow).
fn answer_head(id: u64, ok: bool, len: usize) -> [u8; 20] {
    let mut h = [0u8; 20];
    h[..8].copy_from_slice(&id.to_le_bytes());
    h[8..12].copy_from_slice(&u32::from(!ok).to_le_bytes());
    h[12..].copy_from_slice(&(len as u64).to_le_bytes());
    h
}

/// Reads one answer: `(id, the rows' bytes or the error)`.
fn read_answer(r: &mut impl Read) -> io::Result<(u64, Result<Vec<u8>, String>)> {
    let mut head = [0u8; 20];
    r.read_exact(&mut head)?;
    let len = u64_at(&head, 12) as usize;
    if len > MAX_ROWS * 4096 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("a rows answer of {len} bytes"),
        ));
    }
    let mut b = vec![0u8; len];
    r.read_exact(&mut b)?;
    let got = match u32_at(&head, 8) {
        0 => Ok(b),
        _ => Err(String::from_utf8_lossy(&b).into_owned()),
    };
    Ok((u64_at(&head, 0), got))
}

/// Rank 0: answers the asks that come on `stream` with `read`, on threads of their own, until
/// the connection ends. `log`: one line per ask (`--tp-profile`; asks come only for batches
/// too big for the asking rank's AIO, about one or two a prompt step).
pub fn serve(stream: TcpStream, read: ReadRows, log: bool) -> io::Result<()> {
    let writer = Arc::new(Mutex::new(stream.try_clone()?));
    let (tx, rx) = mpsc::channel::<(u64, bool, Vec<u32>, Instant)>();
    let rx = Arc::new(Mutex::new(rx));
    for i in 0..ANSWERERS {
        let (rx, writer, read) = (rx.clone(), writer.clone(), read.clone());
        std::thread::Builder::new()
            .name(format!("tp-rows-{i}"))
            .spawn(move || {
                loop {
                    let next = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    let Ok((id, urgent, rows, came)) = next else {
                        return;
                    };
                    let t0 = Instant::now();
                    let got = read(&rows, urgent);
                    let t_read = t0.elapsed();
                    let (ok, bytes) = match got {
                        Ok(b) => (true, b),
                        Err(e) => (false, e.to_string().into_bytes()),
                    };
                    let mut w = writer.lock().unwrap_or_else(|e| e.into_inner());
                    let sent = w
                        .write_all(&answer_head(id, ok, bytes.len()))
                        .and_then(|()| w.write_all(&bytes))
                        .and_then(|()| w.flush());
                    drop(w);
                    if let Err(e) = sent {
                        tracing::warn!(error = %e, "tensor parallel: the rows connection failed");
                        return;
                    }
                    if log {
                        tracing::info!(
                            rows = rows.len(),
                            urgent,
                            ok,
                            mib = format!("{:.2}", bytes.len() as f64 / (1u64 << 20) as f64),
                            queued_ms = format!("{:.2}", (t0 - came).as_secs_f64() * 1e3),
                            read_ms = format!("{:.2}", t_read.as_secs_f64() * 1e3),
                            send_ms = format!("{:.2}", (t0.elapsed() - t_read).as_secs_f64() * 1e3),
                            "tensor parallel: rows read for a peer"
                        );
                    }
                }
            })?;
    }
    let mut reader = stream;
    std::thread::Builder::new()
        .name("tp-rows-in".into())
        .spawn(move || {
            loop {
                match read_ask(&mut reader) {
                    Ok((id, urgent, rows)) => {
                        if tx.send((id, urgent, rows, Instant::now())).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        if e.kind() != io::ErrorKind::UnexpectedEof {
                            tracing::warn!(error = %e, "tensor parallel: the rows connection ended");
                        }
                        // Dropping `tx` ends the answering threads.
                        return;
                    }
                }
            }
        })?;
    Ok(())
}

/// A rank above 0's end: asks rank 0 for rows ([`PeerRows`]).
pub struct Client {
    writer: Mutex<TcpStream>,
    state: Arc<Mutex<ClientState>>,
    /// One line per answer: its rows and round trip (`--tp-profile`).
    log: bool,
}

#[derive(Default)]
struct ClientState {
    next: u64,
    /// The asks not answered yet.
    waiting: HashMap<u64, PeerDone>,
    /// Why the connection ended: every ask from then on fails at once.
    ended: Option<String>,
}

impl ClientState {
    /// Ends the connection: the asks still waiting fail (their rows are read here).
    fn end(state: &Mutex<ClientState>, why: String) {
        let waiting = {
            let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
            if st.ended.is_none() {
                tracing::warn!(
                    error = %why,
                    "tensor parallel: the rows connection to rank 0 ended; reading the table here"
                );
                st.ended = Some(why.clone());
            }
            std::mem::take(&mut st.waiting)
        };
        for (_, done) in waiting {
            done(Err(io::Error::other(why.clone())));
        }
    }
}

impl Client {
    /// Takes `stream` (the rows connection from rank 0) and reads its answers on a thread.
    /// `log`: one line per answer (`--tp-profile`).
    pub fn start(stream: TcpStream, log: bool) -> io::Result<Arc<Self>> {
        let state = Arc::new(Mutex::new(ClientState::default()));
        let mut reader = stream.try_clone()?;
        let st = state.clone();
        std::thread::Builder::new()
            .name("tp-rows-answers".into())
            .spawn(move || {
                loop {
                    match read_answer(&mut reader) {
                        Ok((id, got)) => {
                            let done = st
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .waiting
                                .remove(&id);
                            if let Some(done) = done {
                                done(got.map_err(io::Error::other));
                            }
                        }
                        Err(e) => {
                            ClientState::end(&st, e.to_string());
                            return;
                        }
                    }
                }
            })?;
        Ok(Arc::new(Self {
            writer: Mutex::new(stream),
            state,
            log,
        }))
    }
}

impl PeerRows for Client {
    fn ask(&self, rows: Vec<u32>, urgent: bool, done: PeerDone) {
        let done: PeerDone = if self.log {
            let (n, asked) = (rows.len(), Instant::now());
            Box::new(move |got: io::Result<Vec<u8>>| {
                tracing::info!(
                    rows = n,
                    urgent,
                    ok = got.is_ok(),
                    round_trip_ms = format!("{:.2}", asked.elapsed().as_secs_f64() * 1e3),
                    "tensor parallel: rows from rank 0"
                );
                done(got)
            })
        } else {
            done
        };
        let id = {
            let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(why) = &st.ended {
                let why = why.clone();
                drop(st);
                done(Err(io::Error::other(why)));
                return;
            }
            let id = st.next;
            st.next += 1;
            // Before the ask goes out: its answer may come before this thread returns.
            st.waiting.insert(id, done);
            id
        };
        let frame = ask_frame(id, urgent, &rows);
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let sent = w.write_all(&frame).and_then(|()| w.flush());
        drop(w);
        if let Err(e) = sent {
            ClientState::end(&self.state, e.to_string());
        }
    }
}
