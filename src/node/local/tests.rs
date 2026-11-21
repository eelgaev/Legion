//! The in-process control connection: requests answered in order, replies owed and drained,
//! an error reply reported, and both ends noticing the other stop.

use super::*;

#[test]
fn requests_are_answered_in_order() {
    let (mut link, serve) = local_link();
    let rank = std::thread::spawn(move || {
        let mut seen = Vec::new();
        serve.serve(&mut |req| match req {
            Request::Resize { n_blocks } => {
                seen.push(n_blocks);
                Reply::Hello { n_blocks }
            }
            _ => Reply::Error("unexpected".into()),
        });
        seen
    });
    for n in 1..=3 {
        link.send(Request::Resize { n_blocks: n }).unwrap();
    }
    assert_eq!(link.owed(), 3);
    // The last reply; every owed one is read.
    assert_eq!(link.wait().unwrap(), Reply::Hello { n_blocks: 3 });
    assert_eq!(link.owed(), 0);
    assert_eq!(
        link.call(Request::Resize { n_blocks: 9 }).unwrap(),
        Reply::Hello { n_blocks: 9 }
    );
    drop(link);
    assert_eq!(rank.join().unwrap(), vec![1, 2, 3, 9]);
}

#[test]
fn an_error_reply_is_the_error() {
    let (mut link, serve) = local_link();
    let rank = std::thread::spawn(move || {
        serve.serve(&mut |req| match req {
            Request::ClearSlot { slot: 1 } => Reply::Error("slot 1".into()),
            _ => Reply::Done,
        })
    });
    link.send(Request::ClearSlot { slot: 0 }).unwrap();
    link.send(Request::ClearSlot { slot: 1 }).unwrap();
    link.send(Request::ClearSlot { slot: 2 }).unwrap();
    assert_eq!(link.wait().unwrap_err(), "slot 1");
    assert_eq!(link.owed(), 0);
    drop(link);
    rank.join().unwrap();
}

#[test]
fn older_replies_are_read_and_the_last_ones_left() {
    let (mut link, serve) = local_link();
    let rank = std::thread::spawn(move || {
        serve.serve(&mut |req| match req {
            Request::ClearSlot { slot: 1 } => Reply::Error("slot 1".into()),
            Request::Resize { n_blocks } => Reply::Hello { n_blocks },
            _ => Reply::Done,
        })
    });
    for n in 1..=4 {
        link.send(Request::Resize { n_blocks: n }).unwrap();
    }
    // Nothing to read with no more owed than left.
    link.wait_older(4).unwrap();
    assert_eq!(link.owed(), 4);
    link.wait_older(1).unwrap();
    assert_eq!(link.owed(), 1);
    // The one left is the last request's.
    assert_eq!(link.wait().unwrap(), Reply::Hello { n_blocks: 4 });
    // An error among the older ones is the error; the ones after it stay owed.
    link.send(Request::ClearSlot { slot: 1 }).unwrap();
    link.send(Request::ClearSlot { slot: 2 }).unwrap();
    assert_eq!(link.wait_older(1).unwrap_err(), "slot 1");
    assert_eq!(link.owed(), 1);
    assert_eq!(link.wait().unwrap(), Reply::Done);
    drop(link);
    rank.join().unwrap();
}

#[test]
fn a_stopped_rank_is_an_error() {
    let (mut link, serve) = local_link();
    drop(serve);
    assert!(link.send(Request::ClearSlot { slot: 0 }).is_err());
    assert!(link.wait().is_err());
}
