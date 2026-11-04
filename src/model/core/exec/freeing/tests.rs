use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use super::{Freeing, doomed};

/// Counts its drops.
struct Counted(Arc<AtomicUsize>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn values(n: usize, drops: &Arc<AtomicUsize>) -> Vec<Box<dyn Send>> {
    (0..n)
        .map(|_| Box::new(Counted(drops.clone())) as Box<dyn Send>)
        .collect()
}

#[test]
fn nothing_pending_does_not_wait() {
    assert!(!Freeing::default().wait());
}

#[test]
fn wait_returns_once_another_thread_dropped_every_value() {
    let drops = Arc::new(AtomicUsize::new(0));
    let (doomed, mut freeing) = doomed(values(3, &drops));
    assert_eq!(doomed.len(), 3);
    let t = std::thread::spawn(move || {
        // Slow, like a cudaFree that took 50 ms.
        std::thread::sleep(Duration::from_millis(50));
        drop(doomed);
    });
    assert!(freeing.wait());
    assert_eq!(
        drops.load(Ordering::SeqCst),
        3,
        "woke before the values were gone"
    );
    // Only the first wait waits.
    assert!(!freeing.wait());
    t.join().unwrap();
}

#[test]
fn dropped_in_place_wakes_too() {
    // A restore that never starts its thread drops the closure holding them right there.
    let drops = Arc::new(AtomicUsize::new(0));
    let (doomed, mut freeing) = doomed(values(2, &drops));
    let free = move || drop(doomed);
    drop(free);
    assert!(freeing.wait());
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}
