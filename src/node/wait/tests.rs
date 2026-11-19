use super::*;
use std::io::Write;

const QUICK: Pace = Pace {
    spin: Duration::from_micros(200),
    nap: Duration::from_micros(50),
    nap_for: Duration::from_millis(20),
};

#[test]
fn until_returns_once_ready() {
    let mut n = 0;
    assert!(until(QUICK, || {
        n += 1;
        n == 40
    }));
    assert_eq!(n, 40);
}

#[test]
fn until_gives_up_after_its_naps() {
    let t0 = Instant::now();
    assert!(!until(QUICK, || false));
    assert!(t0.elapsed() >= QUICK.nap_for);
}

#[test]
fn recv_takes_queued_items_and_sees_the_end() {
    let (tx, rx) = mpsc::channel();
    tx.send(7).unwrap();
    assert_eq!(recv(&rx, Some(QUICK), None), Some(7));
    let late = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(2));
        tx.send(8).unwrap();
    });
    assert_eq!(recv(&rx, Some(QUICK), None), Some(8));
    late.join().unwrap();
    assert_eq!(recv(&rx, Some(QUICK), None), None);
    let (tx, rx) = mpsc::channel::<u32>();
    drop(tx);
    assert_eq!(recv(&rx, None, None), None);
}

#[test]
fn recv_blocks_after_its_naps() {
    let (tx, rx) = mpsc::channel();
    let late = std::thread::spawn(move || {
        std::thread::sleep(QUICK.nap_for * 3);
        tx.send(1).unwrap();
    });
    assert_eq!(recv(&rx, Some(QUICK), None), Some(1));
    late.join().unwrap();
}

#[test]
fn a_rung_bell_is_seen_once() {
    let b = Bell::default();
    assert!(!b.take());
    b.ring();
    // A nap after a ring returns at once (the ring is not lost).
    let t0 = Instant::now();
    b.nap(Duration::from_secs(5));
    assert!(t0.elapsed() < Duration::from_secs(1));
    assert!(b.take());
    assert!(!b.take());
}

#[test]
fn a_ring_ends_a_nap_and_spins_again() {
    // Long naps: only a ring ends one early.
    const SLOW: Pace = Pace {
        spin: Duration::from_micros(100),
        nap: Duration::from_secs(2),
        nap_for: Duration::from_secs(30),
    };
    let bell = std::sync::Arc::new(Bell::default());
    let (tx, rx) = mpsc::channel();
    let b = bell.clone();
    let late = std::thread::spawn(move || {
        // Well into the waiter's first nap.
        std::thread::sleep(Duration::from_millis(50));
        b.ring();
        std::thread::sleep(Duration::from_millis(5));
        tx.send(3).unwrap();
    });
    let t0 = Instant::now();
    assert_eq!(recv(&rx, Some(SLOW), Some(&*bell)), Some(3));
    assert!(t0.elapsed() < Duration::from_secs(1), "{:?}", t0.elapsed());
    late.join().unwrap();
    // A ring while the wait spins restarts its spin; it still ends when `ready` is.
    let mut n = 0;
    assert!(until_rung(QUICK, Some(&*bell), || {
        n += 1;
        if n == 100 {
            bell.ring();
        }
        n == 200
    }));
    assert!(!bell.take());
}

#[test]
fn a_socket_is_readable_once_bytes_or_its_end_come() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let mut a = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (b, _) = listener.accept().unwrap();
    let r = std::io::BufReader::new(b);
    assert!(!readable(r.get_ref()));
    a.write_all(&[5]).unwrap();
    assert!(until(QUICK, || readable(r.get_ref())));
    // A peek leaves the byte to read.
    assert!(readable(r.get_ref()));
    for_frame(&r, Some(QUICK), None);
    let mut r = r;
    let mut one = [0u8; 1];
    std::io::Read::read_exact(&mut r, &mut one).unwrap();
    assert_eq!(one, [5]);
    drop(a);
    assert!(until(QUICK, || readable(r.get_ref())));
}

#[cfg(target_os = "linux")]
#[test]
fn set_slack_changes_only_the_calling_thread() {
    fn get() -> libc::c_int {
        let none: libc::c_ulong = 0;
        // SAFETY: PR_GET_TIMERSLACK takes no pointer and returns the calling thread's slack.
        unsafe { libc::prctl(libc::PR_GET_TIMERSLACK, none, none, none, none) }
    }
    let before = get();
    let inner = std::thread::spawn(|| {
        assert!(set_slack(SLACK_NS));
        get()
    })
    .join()
    .unwrap();
    assert_eq!(u64::try_from(inner).unwrap(), SLACK_NS);
    assert_eq!(get(), before);
}
