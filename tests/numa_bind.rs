//! `--numa` (#422) on the real kernel: a thread bound with `MPOL_BIND` reads its policy back,
//! and memory it touches afterwards lands on the node it was bound to. GPU-free; any machine
//! (one node is enough). Skips where seccomp forbids the memory-policy calls (containers).
#![cfg(target_os = "linux")]

mod common;

use legion::hw::numa::Topology;
use legion::hw::numa_bind::{self, Inherited, MPOL_BIND};

#[test]
fn a_bound_thread_allocates_on_its_node() {
    let topo = Topology::detect().unwrap();
    let node = topo
        .nodes
        .iter()
        .find(|n| !n.cpus.is_empty() && n.mem_free > 64 << 20)
        .map_or(0, |n| n.id);
    // A thread of its own: the policy stays with it, not with the test harness.
    std::thread::spawn(move || {
        if let Err(e) = numa_bind::bind_thread_memory(&[node]) {
            eprintln!("skipped: set_mempolicy not allowed here ({e})");
            return;
        }
        let policy = numa_bind::thread_policy().unwrap();
        assert_eq!(policy.mode, MPOL_BIND);
        assert_eq!(policy.nodes, [node]);
        // Threads inherit it (what `legion serve`'s runtime threads rely on).
        let child = std::thread::spawn(numa_bind::thread_policy)
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(child, policy);
        let buf = vec![1u8; 8 << 20];
        for offset in [0, buf.len() / 2, buf.len() - 1] {
            assert_eq!(
                numa_bind::page_node(&buf[offset]).unwrap(),
                node,
                "page at {offset}"
            );
        }
        let inherited = Inherited::read().unwrap();
        assert_eq!(inherited.policy, policy);
        assert!(!inherited.cpus.is_empty());
    })
    .join()
    .unwrap();
}
