use super::*;
use std::sync::mpsc::channel;

fn key(p: &(Vec<u32>, u32, usize)) -> WaveKey<'_> {
    WaveKey {
        prompt: &p.0,
        max_tokens: p.1,
        min_tokens: 0,
        ignore_eos: false,
    }
}

#[test]
fn order_is_by_content_then_arrival() {
    // (prompt, max_tokens, arrival)
    let mut held = vec![
        (vec![5, 1], 16, 0),
        (vec![2, 9, 9], 16, 1),
        (vec![5, 1], 8, 2),
        (vec![2, 9, 9], 16, 3),
        (vec![2, 9], 16, 4),
    ];
    order(&mut held, key);
    let arrival: Vec<usize> = held.iter().map(|h| h.2).collect();
    assert_eq!(arrival, vec![4, 1, 3, 2, 0]);
}

#[test]
fn order_does_not_follow_arrival() {
    let a = vec![(vec![3], 4, 0), (vec![1], 4, 1), (vec![2], 4, 2)];
    let mut b = a.clone();
    b.reverse();
    let mut a = a;
    order(&mut a, key);
    order(&mut b, key);
    let pa: Vec<&Vec<u32>> = a.iter().map(|h| &h.0).collect();
    let pb: Vec<&Vec<u32>> = b.iter().map(|h| &h.0).collect();
    assert_eq!(pa, pb);
}

#[test]
fn gather_takes_what_is_queued_and_stops_when_quiet() {
    let (tx, rx) = channel();
    for i in 0..5 {
        tx.send(i).unwrap();
    }
    let mut held = vec![-1];
    let t = Instant::now();
    assert!(gather(&rx, Duration::from_millis(20), &mut held));
    assert_eq!(held, vec![-1, 0, 1, 2, 3, 4]);
    // One quiet window after the last one, well short of the cap.
    assert!(t.elapsed() < Duration::from_millis(20) * MAX_WINDOWS);
}

#[test]
fn gather_waits_for_late_arrivals_within_the_window() {
    let (tx, rx) = channel();
    let sender = std::thread::spawn(move || {
        for i in 0..3 {
            std::thread::sleep(Duration::from_millis(5));
            tx.send(i).unwrap();
        }
        // Keep the channel open past the gather.
        std::thread::sleep(Duration::from_millis(300));
    });
    let mut held = Vec::new();
    assert!(gather(&rx, Duration::from_millis(100), &mut held));
    assert_eq!(held, vec![0, 1, 2]);
    sender.join().unwrap();
}

#[test]
fn gather_stops_at_the_cap() {
    let (tx, rx) = channel();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let s = stop.clone();
    let sender = std::thread::spawn(move || {
        // A request every 2 ms, faster than the 10 ms window, until told to stop.
        while !s.load(std::sync::atomic::Ordering::Relaxed) {
            if tx.send(0).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    });
    let mut held = Vec::new();
    let t = Instant::now();
    assert!(gather(&rx, Duration::from_millis(10), &mut held));
    let took = t.elapsed();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    sender.join().unwrap();
    assert!(took >= Duration::from_millis(10) * MAX_WINDOWS);
    assert!(took < Duration::from_millis(10) * MAX_WINDOWS * 5);
    assert!(!held.is_empty());
}

#[test]
fn gather_reports_a_closed_channel() {
    let (tx, rx) = channel::<u32>();
    tx.send(7).unwrap();
    drop(tx);
    let mut held = Vec::new();
    assert!(!gather(&rx, Duration::from_millis(10), &mut held));
    assert_eq!(held, vec![7]);
}
