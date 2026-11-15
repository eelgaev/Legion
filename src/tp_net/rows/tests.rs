use super::*;
use std::net::TcpListener;
use std::sync::mpsc::Receiver;
use std::time::Duration;

/// A connected pair over loopback: (rank 0's end, the asking rank's end).
fn pair() -> (TcpStream, TcpStream) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let a = TcpStream::connect(l.local_addr().unwrap()).unwrap();
    let (b, _) = l.accept().unwrap();
    (b, a)
}

/// A row's bytes in the fake table: 3 bytes from its index.
fn row(r: u32) -> [u8; 3] {
    [r as u8, (r >> 8) as u8, 0x5a]
}

fn table() -> ReadRows {
    Arc::new(|rows: &[u32], _urgent: bool| {
        if rows.contains(&u32::MAX) {
            return Err(io::Error::other("row outside the table"));
        }
        Ok(rows.iter().flat_map(|&r| row(r)).collect())
    })
}

fn ask(c: &Client, rows: Vec<u32>, urgent: bool) -> Receiver<io::Result<Vec<u8>>> {
    let (tx, rx) = mpsc::channel();
    c.ask(
        rows,
        urgent,
        Box::new(move |got| {
            let _ = tx.send(got);
        }),
    );
    rx
}

const WAIT: Duration = Duration::from_secs(10);

/// Asks come back with the rows' bytes in the order asked, an ask the reader fails with its
/// error, and the connection stays up for the next ask.
#[test]
fn asks_get_their_rows_and_errors() {
    let (r0, r1) = pair();
    // Logging on: the answers are the same.
    serve(r0, table(), true).unwrap();
    let c = Client::start(r1, true).unwrap();
    let many: Vec<u32> = (0..5000).map(|i| i * 7).collect();
    let a = ask(&c, vec![3, 9, 300], true);
    let b = ask(&c, many.clone(), false);
    let e = ask(&c, vec![1, u32::MAX], true);
    let want: Vec<u8> = [3, 9, 300].iter().flat_map(|&r| row(r)).collect();
    assert_eq!(a.recv_timeout(WAIT).unwrap().unwrap(), want);
    let want: Vec<u8> = many.iter().flat_map(|&r| row(r)).collect();
    assert_eq!(b.recv_timeout(WAIT).unwrap().unwrap(), want);
    let err = e.recv_timeout(WAIT).unwrap().unwrap_err();
    assert!(err.to_string().contains("outside"), "{err}");
    assert_eq!(
        ask(&c, vec![], true).recv_timeout(WAIT).unwrap().unwrap(),
        Vec::<u8>::new()
    );
}

/// When rank 0's end closes, the asks still waiting fail, and so does every later one, at once.
#[test]
fn asks_fail_once_the_connection_ends() {
    let (r0, r1) = pair();
    let c = Client::start(r1, false).unwrap();
    // Nobody answers: the ask waits.
    let waiting = ask(&c, vec![1, 2], true);
    assert!(waiting.recv_timeout(Duration::from_millis(100)).is_err());
    drop(r0);
    assert!(waiting.recv_timeout(WAIT).unwrap().is_err());
    assert!(ask(&c, vec![4], false).recv_timeout(WAIT).unwrap().is_err());
}

/// The frames read back as written.
#[test]
fn frames_round_trip() {
    let f = ask_frame(7, true, &[1, u32::MAX, 0]);
    let (id, urgent, rows) = read_ask(&mut f.as_slice()).unwrap();
    assert_eq!((id, urgent, rows), (7, true, vec![1, u32::MAX, 0]));
    let mut a = answer_head(u64::MAX, false, 4).to_vec();
    a.extend_from_slice(b"oops");
    let (id, got) = read_answer(&mut a.as_slice()).unwrap();
    assert_eq!((id, got), (u64::MAX, Err("oops".to_string())));
    let mut a = answer_head(2, true, 2).to_vec();
    a.extend_from_slice(&[9, 8]);
    assert_eq!(read_answer(&mut a.as_slice()).unwrap(), (2, Ok(vec![9, 8])));
}
