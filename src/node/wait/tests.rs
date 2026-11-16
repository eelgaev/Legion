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
    assert_eq!(recv(&rx, Some(QUICK)), Some(7));
    let late = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(2));
        tx.send(8).unwrap();
    });
    assert_eq!(recv(&rx, Some(QUICK)), Some(8));
    late.join().unwrap();
    assert_eq!(recv(&rx, Some(QUICK)), None);
    let (tx, rx) = mpsc::channel::<u32>();
    drop(tx);
    assert_eq!(recv(&rx, None), None);
}

#[test]
fn recv_blocks_after_its_naps() {
    let (tx, rx) = mpsc::channel();
    let late = std::thread::spawn(move || {
        std::thread::sleep(QUICK.nap_for * 3);
        tx.send(1).unwrap();
    });
    assert_eq!(recv(&rx, Some(QUICK)), Some(1));
    late.join().unwrap();
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
    for_frame(&r, Some(QUICK));
    let mut r = r;
    let mut one = [0u8; 1];
    std::io::Read::read_exact(&mut r, &mut one).unwrap();
    assert_eq!(one, [5]);
    drop(a);
    assert!(until(QUICK, || readable(r.get_ref())));
}
