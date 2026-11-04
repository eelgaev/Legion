//! A control connection to a rank that runs on another thread of this process (tensor
//! parallelism over the two GPUs of a pair, `--tp-per-gpu`, #691): [`super::Link`]'s requests
//! and replies over channels, in order, without encoding them.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use super::{Reply, Request};

/// How long a waiting side polls before it blocks: the next request (or reply) of a decode
/// loop usually comes within this, and a blocked thread takes longer to wake.
const SPIN: Duration = Duration::from_micros(200);

/// Receives from `rx`, polling for up to [`SPIN`] before blocking. `None`: the other end is gone.
fn recv<T>(rx: &mpsc::Receiver<T>) -> Option<T> {
    let t0 = Instant::now();
    loop {
        match rx.try_recv() {
            Ok(v) => return Some(v),
            Err(mpsc::TryRecvError::Disconnected) => return None,
            Err(mpsc::TryRecvError::Empty) if t0.elapsed() < SPIN => std::hint::spin_loop(),
            Err(mpsc::TryRecvError::Empty) => return rx.recv().ok(),
        }
    }
}

/// The leading rank's end.
pub struct LocalLink {
    tx: Option<mpsc::Sender<Request>>,
    rx: mpsc::Receiver<Reply>,
    /// Replies not read yet.
    pending: usize,
}

/// The following rank's end: answer the requests with [`Self::serve`].
pub struct LocalServe {
    rx: mpsc::Receiver<Request>,
    tx: mpsc::Sender<Reply>,
}

/// A connected pair of ends.
pub fn local_link() -> (LocalLink, LocalServe) {
    let (req_tx, req_rx) = mpsc::channel();
    let (rep_tx, rep_rx) = mpsc::channel();
    (
        LocalLink {
            tx: Some(req_tx),
            rx: rep_rx,
            pending: 0,
        },
        LocalServe {
            rx: req_rx,
            tx: rep_tx,
        },
    )
}

impl LocalLink {
    /// Queues `req`; its reply is read by a later [`Self::wait`].
    pub fn send(&mut self, req: Request) -> Result<(), String> {
        let tx = self.tx.as_ref().ok_or("the rank's link is closed")?;
        tx.send(req).map_err(|_| "the rank stopped".to_string())?;
        self.pending += 1;
        Ok(())
    }

    /// Reads every reply still owed and returns the last one; an error reply (of any of them)
    /// is the error.
    pub fn wait(&mut self) -> Result<Reply, String> {
        let mut last = None;
        let mut err = None;
        while self.pending > 0 {
            let r = recv(&self.rx).ok_or("the rank stopped")?;
            self.pending -= 1;
            match r {
                Reply::Error(e) => err = err.or(Some(e)),
                r => last = Some(r),
            }
        }
        match err {
            Some(e) => Err(e),
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
        self.pending
    }
}

impl LocalServe {
    /// Answers the requests with `handle`, in order, until the leading end drops.
    pub fn serve(self, handle: &mut impl FnMut(Request) -> Reply) {
        while let Some(req) = recv(&self.rx) {
            if self.tx.send(handle(req)).is_err() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests;
