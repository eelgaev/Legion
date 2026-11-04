//! Linux AIO for the direct tier's small batches (#506).
//!
//! A decode step's or a draft's rows are a few pages (16 per token). Handed to the I/O threads,
//! each read waits for a thread to wake, and on the GB10 its cores sleep in an idle state that
//! takes ~0.4 ms to leave: 16 random 4 KiB reads took ~1.05 ms (median) from queueing to the
//! last row, ~0.15 ms submitted as AIO from the thread that asks for them. The kernel then runs
//! the reads on the device's queues with no thread to wake, and the thread that collects the
//! batch reaps the completions. The bytes are the same either way.
//!
//! Only for `O_DIRECT` files (buffered AIO runs inside `io_submit`). Batches bigger than
//! `--ngram-aio-max` pages, or that find too few free slots, go to the I/O threads as before.

use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::{Arc, Mutex};

use super::{Job, PAGE, Pending, deliver, read_full};

/// The kernel's `struct iocb` (`linux/aio_abi.h`). `aio_key` and `aio_rw_flags` swap places on
/// big-endian targets; both stay zero here, so one layout serves either.
#[repr(C)]
#[derive(Default)]
struct Iocb {
    data: u64,
    key: u32,
    rw_flags: i32,
    opcode: u16,
    reqprio: i16,
    fildes: u32,
    buf: u64,
    nbytes: u64,
    offset: i64,
    reserved2: u64,
    flags: u32,
    resfd: u32,
}

/// The kernel's `struct io_event`.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IoEvent {
    data: u64,
    obj: u64,
    res: i64,
    res2: i64,
}

const IOCB_CMD_PREAD: u16 = 0;

/// Bytes of one slot's buffer: a read is at most two pages (a row straddling a boundary).
const SLOT_BYTES: usize = 2 * PAGE as usize;

/// Reads in flight through one [`Aio`].
pub(super) const DEPTH: usize = 256;

/// An AIO context on one file, with [`DEPTH`] page-aligned read buffers (slots).
pub(super) struct Aio {
    ctx: libc::c_ulong,
    file: Arc<File>,
    slots: Mutex<Slots>,
    /// Held by the thread reaping completions (its event buffer).
    reap: Mutex<Vec<IoEvent>>,
    buf: *mut u8,
    layout: std::alloc::Layout,
}

// SAFETY: `buf` is only touched through slots, which `slots` hands to one read at a time; the
// context is a kernel handle usable from any thread.
unsafe impl Send for Aio {}
unsafe impl Sync for Aio {}

struct Slots {
    free: Vec<usize>,
    /// The read each slot carries.
    jobs: Vec<Option<Job>>,
}

impl Aio {
    /// A context for `file` (opened `O_DIRECT`); `None` where the kernel has no AIO for it.
    pub(super) fn new(file: Arc<File>) -> Option<Self> {
        let mut ctx: libc::c_ulong = 0;
        // SAFETY: io_setup writes the context id to `ctx`.
        let r = unsafe { libc::syscall(libc::SYS_io_setup, DEPTH as libc::c_long, &mut ctx) };
        if r != 0 {
            tracing::debug!(
                error = %io::Error::last_os_error(),
                "no Linux AIO for the lookup table: its I/O threads read every batch"
            );
            return None;
        }
        let layout = std::alloc::Layout::from_size_align(DEPTH * SLOT_BYTES, PAGE as usize)
            .expect("slot layout");
        // SAFETY: a non-zero size.
        let buf = unsafe { std::alloc::alloc_zeroed(layout) };
        if buf.is_null() {
            // SAFETY: the context just set up, with nothing in flight.
            unsafe { libc::syscall(libc::SYS_io_destroy, ctx) };
            return None;
        }
        Some(Self {
            ctx,
            file,
            slots: Mutex::new(Slots {
                free: (0..DEPTH).rev().collect(),
                jobs: (0..DEPTH).map(|_| None).collect(),
            }),
            reap: Mutex::new(vec![IoEvent::default(); DEPTH]),
            buf,
            layout,
        })
    }

    /// The start of slot `i`'s buffer ([`SLOT_BYTES`]).
    fn slot(&self, i: usize) -> *mut u8 {
        assert!(i < DEPTH, "aio slot out of range");
        // SAFETY: within the allocation.
        unsafe { self.buf.add(i * SLOT_BYTES) }
    }

    /// Submits `jobs` (their [`Pending`]s already count them in `aio_jobs`); returns those it
    /// could not take, for the I/O threads, after taking them back out of `aio_jobs`.
    pub(super) fn submit(&self, jobs: Vec<Job>) -> Vec<Job> {
        let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
        let n = jobs.len().min(slots.free.len());
        let mut jobs = jobs.into_iter();
        let mut cbs: Vec<Iocb> = Vec::with_capacity(n);
        let mut taken = Vec::with_capacity(n);
        for job in jobs.by_ref().take(n) {
            let i = slots.free.pop().expect("counted free slots");
            assert!(job.len <= SLOT_BYTES, "a read longer than a slot");
            cbs.push(Iocb {
                data: i as u64,
                opcode: IOCB_CMD_PREAD,
                fildes: self.file.as_raw_fd() as u32,
                buf: self.slot(i) as u64,
                nbytes: job.len as u64,
                offset: job.offset as i64,
                ..Default::default()
            });
            slots.jobs[i] = Some(job);
            taken.push(i);
        }
        let mut ptrs: Vec<*mut Iocb> = cbs.iter_mut().map(|c| c as *mut Iocb).collect();
        let mut done = 0usize;
        while done < ptrs.len() {
            // SAFETY: `ptrs[done..]` point at live iocbs whose buffers are owned slots; the
            // kernel copies the iocbs during the call.
            let r = unsafe {
                libc::syscall(
                    libc::SYS_io_submit,
                    self.ctx,
                    (ptrs.len() - done) as libc::c_long,
                    ptrs[done..].as_mut_ptr(),
                )
            };
            if r > 0 {
                done += r as usize;
                continue;
            }
            if r < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }
        // What the kernel did not take goes back to the threads.
        let mut rest = Vec::new();
        for &i in &taken[done..] {
            let job = slots.jobs[i].take().expect("a slot's job");
            slots.free.push(i);
            rest.push(job);
        }
        drop(slots);
        // With them, the ones that found no free slot.
        rest.extend(jobs);
        for job in &rest {
            job.pending
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .aio_jobs -= 1;
        }
        rest
    }

    /// Reaps completions until `pending` has no reads in flight here.
    pub(super) fn wait(&self, pending: &Pending) {
        loop {
            let mut events = self.reap.lock().unwrap_or_else(|e| e.into_inner());
            // Another thread may have reaped them while this one waited for the lock.
            if pending
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .aio_jobs
                == 0
            {
                return;
            }
            // SAFETY: `events` holds DEPTH events; no timeout: one of `pending`'s reads is in
            // flight, so an event comes.
            let r = unsafe {
                libc::syscall(
                    libc::SYS_io_getevents,
                    self.ctx,
                    1 as libc::c_long,
                    DEPTH as libc::c_long,
                    events.as_mut_ptr(),
                    std::ptr::null_mut::<libc::timespec>(),
                )
            };
            if r < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                // Not expected with a valid context: fail every read in flight rather than
                // wait forever.
                self.fail_all(&e);
                return;
            }
            for ev in &events[..r as usize] {
                self.complete(ev.data as usize, ev.res);
            }
        }
    }

    /// Delivers slot `i`'s read, which ended with `res` (bytes read, or a negative errno),
    /// and frees the slot.
    fn complete(&self, i: usize, res: i64) {
        // None: a read [`Self::fail_all`] already failed (its slot stays out of use).
        let Some(job) = self.slots.lock().unwrap_or_else(|e| e.into_inner()).jobs[i].take() else {
            return;
        };
        assert!(job.len <= SLOT_BYTES, "a read longer than a slot");
        // SAFETY: the read into slot `i` completed and nothing else uses the slot until it is
        // freed below.
        let buf = unsafe { std::slice::from_raw_parts_mut(self.slot(i), job.len) };
        let read = match res {
            r if r < 0 => Err(io::Error::from_raw_os_error(-r as i32)),
            // Short of the end of the file, a short read is retried whole, synchronously.
            r if (r as usize) < job.len => read_full(&self.file, buf, job.offset),
            r => Ok(r as usize),
        };
        deliver(&job, read, buf, true);
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .free
            .push(i);
    }

    /// Fails every read in flight with `e`. Their slots stay out of use: the kernel may still
    /// write into them.
    fn fail_all(&self, e: &io::Error) {
        let jobs: Vec<Job> = {
            let mut slots = self.slots.lock().unwrap_or_else(|e| e.into_inner());
            slots.jobs.iter_mut().filter_map(Option::take).collect()
        };
        for job in &jobs {
            deliver(job, Err(io::Error::new(e.kind(), e.to_string())), &[], true);
        }
    }
}

impl Drop for Aio {
    fn drop(&mut self) {
        // io_destroy waits for the reads still in flight, so the buffers outlive them.
        // SAFETY: our context, destroyed once.
        unsafe { libc::syscall(libc::SYS_io_destroy, self.ctx) };
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.buf, self.layout) };
    }
}
