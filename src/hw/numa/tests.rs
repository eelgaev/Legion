use super::*;

#[test]
fn copy_workers_exclude_host_core_and_smt_siblings() {
    assert_eq!(copy_worker_cpus(&[0, 1, 80, 81], 0, &[0, 80]), [1, 81]);
    assert_eq!(copy_worker_cpus(&[0, 1], 0, &[]), [1]);
    assert!(copy_worker_cpus(&[0, 80], 0, &[0, 80]).is_empty());
}

#[test]
fn helper_threads_leave_cpu0_and_its_core() {
    // Node 0 of a 4-way SMT machine: CPU 0's core is 0-3.
    let node: Vec<u32> = (0..8).collect();
    let all: Vec<u32> = (0..16).collect();
    assert_eq!(helper_set(&node, &all, &[0, 1, 2, 3]), [4, 5, 6, 7]);
    // Without sysfs siblings CPU 0 still goes.
    assert_eq!(helper_set(&node, &all, &[]), [1, 2, 3, 4, 5, 6, 7]);
    // Only CPUs the process may use (numactl --cpunodebind of another node: none).
    assert_eq!(helper_set(&node, &[5, 6, 9], &[0, 1, 2, 3]), [5, 6]);
    assert!(helper_set(&node, &[8, 9], &[]).is_empty());
}

fn node(id: u32, cpus: &[u32], mem_total: u64) -> NumaNode {
    NumaNode {
        id,
        cpus: cpus.to_vec(),
        mem_total,
        mem_free: mem_total / 2,
    }
}

/// An AC922: two sockets, then four GPUs' memory as CPU-less nodes.
fn ac922() -> Topology {
    Topology {
        nodes: vec![
            node(0, &[0, 1], 256 << 30),
            node(8, &[80, 81], 256 << 30),
            node(252, &[], 16 << 30),
            node(253, &[], 16 << 30),
        ],
    }
}

#[test]
fn page_cache_binds_to_cpu_nodes_only_beside_memory_only_nodes() {
    assert_eq!(cpu_nodes_beside_memory_only(&ac922()), Some(vec![0, 8]));
    // Every node has CPUs: nothing to keep the page cache from.
    let plain = Topology {
        nodes: vec![node(0, &[0, 1], 64 << 30), node(1, &[2, 3], 64 << 30)],
    };
    assert_eq!(cpu_nodes_beside_memory_only(&plain), None);
    // A CPU-less node without memory (offline hotplug slot) does not count.
    let empty = Topology {
        nodes: vec![node(0, &[0, 1], 64 << 30), node(1, &[], 0)],
    };
    assert_eq!(cpu_nodes_beside_memory_only(&empty), None);
}

#[test]
fn page_cache_on_memory_only_nodes_from_fixture() {
    let dir = tempfile::tempdir().unwrap();
    for (id, file_kb) in [(0, 900_000_000u64), (8, 1), (252, 2_189_312), (253, 1024)] {
        let n = dir.path().join(format!("node{id}"));
        std::fs::create_dir(&n).unwrap();
        std::fs::write(
            n.join("meminfo"),
            format!("Node {id} MemTotal: 1 kB\nNode {id} FilePages:      {file_kb} kB\n"),
        )
        .unwrap();
    }
    // Node 252's 2.1 GiB; not node 253's 1 MiB (under the floor), nor any CPU node's.
    assert_eq!(
        memory_only_page_cache_in(dir.path(), &ac922(), 256 << 20),
        [(252, 2_189_312 << 10)]
    );
    assert_eq!(
        memory_only_page_cache_in(dir.path(), &ac922(), 0),
        [(252, 2_189_312 << 10), (253, 1024 << 10)]
    );
}

#[test]
fn distances_parse_until_garbage() {
    assert_eq!(parse_distances("10 20 40 80\n"), [10, 20, 40, 80]);
    assert_eq!(parse_distances(" 10  20 x 30"), [10, 20]);
    assert!(parse_distances("").is_empty());
}

#[test]
fn cpu_nodes_stay_cpu_less_ones_go_to_the_nearest() {
    // An AC922: the sockets' nodes stay; a GPU memory node goes to the nearer socket.
    let t = ac922();
    assert_eq!(nearest_cpu_node(0, &t, &[]), Some(0));
    assert_eq!(nearest_cpu_node(8, &t, &[]), Some(8));
    assert_eq!(nearest_cpu_node(253, &t, &[80, 40, 80, 10]), Some(8));
    // No distances: the lowest node with CPUs.
    assert_eq!(nearest_cpu_node(252, &t, &[]), Some(0));
    // A node the topology does not know: as given.
    assert_eq!(nearest_cpu_node(5, &t, &[]), Some(5));
    // A GB10: one node.
    let gb10 = Topology {
        nodes: vec![node(0, &(0..20).collect::<Vec<u32>>(), 128 << 30)],
    };
    assert_eq!(nearest_cpu_node(0, &gb10, &[10]), Some(0));
    // No node with CPUs at all: none.
    let none = Topology {
        nodes: vec![node(1, &[], 96 << 30)],
    };
    assert_eq!(nearest_cpu_node(1, &none, &[10]), None);
}

#[test]
fn cpu_node_near_from_fixture() {
    // A GH200: Grace on node 0, the HBM as CPU-less node 1.
    let dir = tempfile::tempdir().unwrap();
    for (id, cpus, distance) in [(0, "0-71\n", "10 80\n"), (1, "\n", "80 10\n")] {
        let n = dir.path().join(format!("node{id}"));
        std::fs::create_dir(&n).unwrap();
        std::fs::write(n.join("cpulist"), cpus).unwrap();
        std::fs::write(n.join("distance"), distance).unwrap();
        std::fs::write(n.join("meminfo"), format!("Node {id} MemTotal: 1024 kB\n")).unwrap();
    }
    assert_eq!(cpu_node_near_in(dir.path(), 0), 0);
    assert_eq!(cpu_node_near_in(dir.path(), 1), 0);
    // A node sysfs does not list: as given.
    assert_eq!(cpu_node_near_in(dir.path(), 7), 7);
}
