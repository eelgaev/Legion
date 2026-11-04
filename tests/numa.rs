use std::fs;
use std::path::Path;

use legion::hw::numa::{
    NumaNode, Topology, parallel_fill_on_node, parse_cpulist, pci_numa_node_in, pin_current_thread,
    run_on_node,
};
use nix::sched::{CpuSet, sched_getaffinity};
use nix::unistd::Pid;

fn affinity() -> Vec<u32> {
    let set = sched_getaffinity(Pid::from_raw(0)).unwrap();
    (0..CpuSet::count() as u32)
        .filter(|&c| set.is_set(c as usize).unwrap())
        .collect()
}

fn write_node(root: &Path, id: u32, cpulist: &str, total_kb: u64, free_kb: u64) {
    let dir = root.join(format!("node{id}"));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("cpulist"), format!("{cpulist}\n")).unwrap();
    fs::write(
        dir.join("meminfo"),
        format!(
            "Node {id} MemTotal:       {total_kb} kB\nNode {id} MemFree:        {free_kb} kB\n\
             Node {id} MemUsed:        1 kB\n"
        ),
    )
    .unwrap();
}

fn topo_of(cpus: Vec<u32>) -> Topology {
    Topology {
        nodes: vec![NumaNode {
            id: 3,
            cpus,
            mem_total: 0,
            mem_free: 0,
        }],
    }
}

#[test]
fn parses_cpulists() {
    assert_eq!(parse_cpulist("0-3,8,10-11\n"), [0, 1, 2, 3, 8, 10, 11]);
    assert_eq!(parse_cpulist(" 5 , 1-2 ,2"), [1, 2, 5]);
    assert_eq!(parse_cpulist("80-159"), (80..160).collect::<Vec<_>>());
    assert!(parse_cpulist("").is_empty());
    assert!(parse_cpulist("x,3-1,-,a-b").is_empty());
}

#[test]
fn two_node_ac922_layout() {
    let dir = tempfile::tempdir().unwrap();
    write_node(dir.path(), 8, "80-159", 262_144, 100_000);
    write_node(dir.path(), 0, "0-79", 524_288, 400_000);
    // Non-node entries are ignored.
    fs::write(dir.path().join("online"), "0,8\n").unwrap();
    fs::create_dir(dir.path().join("power")).unwrap();

    let topo = Topology::from_sysfs(dir.path()).unwrap();
    assert_eq!(topo.nodes.len(), 2);
    assert_eq!(topo.nodes[0].id, 0);
    assert_eq!(topo.nodes[1].id, 8);
    assert_eq!(topo.nodes[0].cpus, (0..80).collect::<Vec<_>>());
    assert_eq!(topo.nodes[1].cpus, (80..160).collect::<Vec<_>>());
    assert_eq!(topo.nodes[0].mem_total, 524_288 * 1024);
    assert_eq!(topo.nodes[0].mem_free, 400_000 * 1024);
    assert_eq!(topo.nodes[1].mem_total, 262_144 * 1024);
    assert_eq!(topo.node_of_cpu(0), Some(0));
    assert_eq!(topo.node_of_cpu(79), Some(0));
    assert_eq!(topo.node_of_cpu(80), Some(8));
    assert_eq!(topo.node_of_cpu(159), Some(8));
    assert_eq!(topo.node_of_cpu(160), None);
}

#[test]
fn single_node_layout_and_missing_meminfo() {
    let dir = tempfile::tempdir().unwrap();
    let node = dir.path().join("node0");
    fs::create_dir(&node).unwrap();
    fs::write(node.join("cpulist"), "0-7\n").unwrap();
    let topo = Topology::from_sysfs(dir.path()).unwrap();
    assert_eq!(topo.nodes.len(), 1);
    assert_eq!(topo.nodes[0].cpus, (0..8).collect::<Vec<_>>());
    assert_eq!(topo.nodes[0].mem_total, 0);
    assert_eq!(topo.nodes[0].mem_free, 0);
}

#[test]
fn missing_or_empty_sysfs_degrades_to_one_node() {
    let dir = tempfile::tempdir().unwrap();
    for root in [dir.path().join("does-not-exist"), dir.path().to_path_buf()] {
        let topo = Topology::from_sysfs(&root).unwrap();
        assert_eq!(topo.nodes.len(), 1);
        assert_eq!(topo.nodes[0].id, 0);
        assert!(!topo.nodes[0].cpus.is_empty());
    }
    let topo = Topology::detect().unwrap();
    assert!(!topo.nodes.is_empty());
    assert!(topo.nodes.iter().all(|n| !n.cpus.is_empty() || n.id > 0));
}

#[test]
fn pci_node_lookup() {
    let dir = tempfile::tempdir().unwrap();
    for (bdf, v) in [
        ("0004:04:00.0", "8\n"),
        ("0000:01:00.0", "-1\n"),
        ("bad", "x\n"),
    ] {
        let d = dir.path().join(bdf);
        fs::create_dir(&d).unwrap();
        fs::write(d.join("numa_node"), v).unwrap();
    }
    assert_eq!(pci_numa_node_in(dir.path(), "0004:04:00.0"), Some(8));
    assert_eq!(pci_numa_node_in(dir.path(), "0000:01:00.0"), None);
    assert_eq!(pci_numa_node_in(dir.path(), "bad"), None);
    assert_eq!(pci_numa_node_in(dir.path(), "missing"), None);
    assert_eq!(pci_numa_node_in(dir.path(), "../x"), None);
    assert_eq!(pci_numa_node_in(dir.path(), ""), None);
}

#[test]
fn pin_round_trip() {
    let original = affinity();
    let target = *original.last().unwrap();
    let previous = pin_current_thread(&[target]).unwrap();
    assert_eq!(previous, original);
    assert_eq!(affinity(), [target]);
    let pinned_prev = pin_current_thread(&previous).unwrap();
    assert_eq!(pinned_prev, [target]);
    assert_eq!(affinity(), original);
}

#[test]
fn pin_rejects_bad_masks() {
    let before = affinity();
    assert!(pin_current_thread(&[]).is_err());
    assert!(pin_current_thread(&[1 << 20]).is_err());
    assert_eq!(affinity(), before);
}

#[test]
fn run_on_node_observes_node_affinity() {
    let allowed = affinity();
    let cpus: Vec<u32> = allowed.iter().copied().take(2).collect();
    let topo = topo_of(cpus.clone());
    let seen = run_on_node(&topo, 3, affinity);
    assert_eq!(seen, cpus);
    // Even when the caller is pinned to one CPU, the worker gets the full node list.
    let last = *allowed.last().unwrap();
    let previous = pin_current_thread(&[last]).unwrap();
    let seen = run_on_node(&topo, 3, affinity);
    pin_current_thread(&previous).unwrap();
    assert_eq!(seen, cpus);
}

#[test]
fn run_on_node_with_high_cpu_ids() {
    // Exercises ids >= 64 when the machine has them.
    let high: Vec<u32> = affinity()
        .into_iter()
        .filter(|&c| c >= 64)
        .take(3)
        .collect();
    if high.is_empty() {
        return;
    }
    let topo = topo_of(high.clone());
    assert_eq!(run_on_node(&topo, 3, affinity), high);
}

#[test]
fn parallel_fill_handles_uneven_sizes() {
    let topo = topo_of(affinity());
    for (len, threads) in [
        (0, 4),
        (1, 4),
        (7, 3),
        (1000, 7),
        (4096, 4),
        (10_001, 16),
        (5, 100),
    ] {
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
        let mut buf = vec![0u8; len];
        parallel_fill_on_node(&topo, 3, threads, &mut buf, &src);
        assert_eq!(buf, src, "len={len} threads={threads}");
    }
}

#[test]
fn parallel_fill_leaves_caller_affinity_alone() {
    let before = affinity();
    let topo = topo_of(before.clone());
    let src = vec![9u8; 100];
    let mut buf = vec![0u8; 100];
    parallel_fill_on_node(&topo, 3, 3, &mut buf, &src);
    assert_eq!(buf, src);
    assert_eq!(affinity(), before);
}

#[test]
#[should_panic(expected = "unknown NUMA node")]
fn unknown_node_panics_clearly() {
    run_on_node(&topo_of(vec![0]), 99, || ());
}
