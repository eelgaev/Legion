use super::*;
use crate::hw::numa::NumaNode;

const GIB: u64 = 1 << 30;

fn node(id: u32, cpus: std::ops::Range<u32>, mem_gib: u64) -> NumaNode {
    NumaNode {
        id,
        cpus: cpus.collect(),
        mem_total: mem_gib * GIB,
        mem_free: mem_gib * GIB,
    }
}

/// An AC922: two sockets (nodes 0 and 8), each GPU's memory a CPU-less node (252-255).
fn ac922() -> Topology {
    let mut nodes = vec![node(0, 0..80, 256), node(8, 80..160, 256)];
    nodes.extend((252..256).map(|id| node(id, 0..0, 16)));
    Topology { nodes }
}

/// A process started without numactl, taskset or a cpuset.
fn free(topo: &Topology) -> Inherited {
    Inherited {
        cpus: topo.nodes.iter().flat_map(|n| n.cpus.clone()).collect(),
        policy: MemPolicy::default(),
        mems_allowed: topo.nodes.iter().map(|n| n.id).collect(),
    }
}

fn auto(topo: &Topology, gpus: &[Option<u32>], inherited: &Inherited) -> Plan {
    plan(&NumaMode::Auto, topo, gpus, inherited).unwrap()
}

#[test]
fn parses_the_flag() {
    assert_eq!("auto".parse(), Ok(NumaMode::Auto));
    assert_eq!("off".parse(), Ok(NumaMode::Off));
    assert_eq!("8".parse(), Ok(NumaMode::Nodes(vec![8])));
    assert_eq!("8, 0,8".parse(), Ok(NumaMode::Nodes(vec![0, 8])));
    assert!("".parse::<NumaMode>().is_err());
    assert!("node8".parse::<NumaMode>().is_err());
    assert!("0,".parse::<NumaMode>().is_err());
    assert_eq!(NumaMode::Nodes(vec![0, 8]).to_string(), "0,8");
}

#[test]
fn auto_binds_to_the_gpus_node() {
    let t = ac922();
    let p = auto(&t, &[Some(8), Some(8)], &free(&t));
    assert_eq!(
        p,
        Plan::Bind {
            nodes: vec![8],
            cpus: (80..160).collect()
        }
    );
}

#[test]
fn auto_with_gpus_on_two_nodes_places_per_gpu() {
    let t = ac922();
    let gpus = [Some(0), Some(0), Some(8), Some(8)];
    assert_eq!(
        auto(&t, &gpus, &free(&t)),
        Plan::PerGpu { nodes: vec![0, 8] }
    );
}

#[test]
fn unknown_gpu_nodes_count_only_where_known() {
    let t = ac922();
    assert!(matches!(
        auto(&t, &[None, Some(8)], &free(&t)),
        Plan::Bind { nodes, .. } if nodes == [8]
    ));
    // A node without CPUs (a GPU's own memory) is not one to bind to.
    assert_eq!(
        auto(&t, &[None, Some(254)], &free(&t)),
        Plan::Skip("the GPUs' NUMA nodes are unknown".into())
    );
    assert_eq!(
        auto(&t, &[], &free(&t)),
        Plan::Skip("no GPUs in use".into())
    );
}

#[test]
fn one_node_machines_are_left_alone() {
    // A desktop, a GB10, or sysfs missing (the fallback topology is one node).
    let desktop = Topology {
        nodes: vec![node(0, 0..32, 64)],
    };
    assert_eq!(
        auto(&desktop, &[Some(0)], &free(&desktop)),
        Plan::Skip("one NUMA node".into())
    );
    // GH200: Grace's CPUs on node 0, the Hopper's memory a CPU-less node 1.
    let gh200 = Topology {
        nodes: vec![node(0, 0..72, 480), node(1, 0..0, 96)],
    };
    assert_eq!(
        auto(&gh200, &[Some(0)], &free(&gh200)),
        Plan::Skip("one NUMA node".into())
    );
    assert_eq!(
        plan(&NumaMode::Nodes(vec![0]), &gh200, &[Some(0)], &free(&gh200)),
        Ok(Plan::Skip("one NUMA node".into()))
    );
}

#[test]
fn off_does_nothing() {
    let t = ac922();
    assert_eq!(
        plan(&NumaMode::Off, &t, &[Some(8)], &free(&t)),
        Ok(Plan::Off)
    );
}

#[test]
fn explicit_nodes_bind_there_and_bad_ones_fail() {
    let t = ac922();
    assert_eq!(
        plan(&NumaMode::Nodes(vec![0, 8]), &t, &[Some(8)], &free(&t)),
        Ok(Plan::Bind {
            nodes: vec![0, 8],
            cpus: (0..160).collect()
        })
    );
    let err = plan(&NumaMode::Nodes(vec![3]), &t, &[], &free(&t)).unwrap_err();
    assert!(err.contains("no NUMA node 3"), "{err}");
    let err = plan(&NumaMode::Nodes(vec![252]), &t, &[], &free(&t)).unwrap_err();
    assert!(err.contains("no CPUs"), "{err}");
}

#[test]
fn an_outer_numactl_is_kept() {
    let t = ac922();
    // numactl --cpunodebind=8 --membind=8
    let numactl = Inherited {
        cpus: (80..160).collect(),
        policy: MemPolicy {
            mode: MPOL_BIND,
            nodes: vec![8],
        },
        ..free(&t)
    };
    let p = auto(&t, &[Some(8)], &numactl);
    assert_eq!(
        p,
        Plan::Outer {
            what: "CPUs 80-159, memory policy bind 8".into(),
            cpu_nodes: vec![8],
            mem_nodes: Some(vec![8]),
        }
    );
    // An explicit node does not override it either.
    assert!(matches!(
        plan(&NumaMode::Nodes(vec![0]), &t, &[Some(8)], &numactl),
        Ok(Plan::Outer { .. })
    ));
    let placed = Placement {
        mode: NumaMode::Auto,
        plan: p,
        gpus: vec![Some(8)],
        error: None,
    };
    assert!(placed.warnings().is_empty());
}

#[test]
fn taskset_or_a_cpuset_alone_is_kept() {
    let t = ac922();
    let taskset = Inherited {
        cpus: (80..120).collect(),
        ..free(&t)
    };
    assert!(matches!(
        auto(&t, &[Some(8)], &taskset),
        Plan::Outer { mem_nodes: None, cpu_nodes, .. } if cpu_nodes == [8]
    ));
    let cpuset = Inherited {
        mems_allowed: vec![0],
        ..free(&t)
    };
    assert!(matches!(
        auto(&t, &[Some(8)], &cpuset),
        Plan::Outer { mem_nodes: Some(m), .. } if m == [0]
    ));
    // A cpuset that allows every node with memory restricts nothing.
    let all_mems = Inherited {
        mems_allowed: vec![0, 8, 252, 253, 254, 255],
        ..free(&t)
    };
    assert!(matches!(auto(&t, &[Some(8)], &all_mems), Plan::Bind { .. }));
}

#[test]
fn warns_when_memory_or_cpus_are_off_the_gpus_node() {
    let t = ac922();
    // numactl --cpunodebind=0 --membind=0 with GPUs 2,3 (node 8).
    let wrong = Inherited {
        cpus: (0..80).collect(),
        policy: MemPolicy {
            mode: MPOL_BIND,
            nodes: vec![0],
        },
        ..free(&t)
    };
    let placed = Placement {
        mode: NumaMode::Auto,
        plan: auto(&t, &[Some(8), Some(8)], &wrong),
        gpus: vec![Some(8), Some(8)],
        error: None,
    };
    let w = placed.warnings();
    assert_eq!(w.len(), 4, "{w:?}");
    assert!(w[0].contains("CUDA0 is on NUMA node 8") && w[0].contains("restricted to node 0"));
    assert!(w[1].contains("CPUs are on node 0"));
    // --numa 0 with the GPUs on node 8.
    let placed = Placement {
        mode: NumaMode::Nodes(vec![0]),
        plan: plan(&NumaMode::Nodes(vec![0]), &t, &[Some(8)], &free(&t)).unwrap(),
        gpus: vec![Some(8)],
        error: None,
    };
    let lines = placed.lines();
    assert_eq!(
        lines[0],
        "--numa 0: CPUs 0-79 and memory bound to NUMA node 0"
    );
    assert_eq!(
        lines[1],
        "CUDA0: NUMA node 8 · pinned buffers on its node, other host memory on node 0"
    );
    assert!(lines[2].starts_with("WARN CUDA0 is on NUMA node 8 but --numa 0 binds"));
}

#[test]
fn report_lines() {
    let t = ac922();
    let gpus = vec![Some(8), Some(8)];
    let placed = Placement {
        mode: NumaMode::Auto,
        plan: auto(&t, &gpus, &free(&t)),
        gpus,
        error: None,
    };
    assert_eq!(
        placed.lines(),
        [
            "--numa auto: CPUs 80-159 and memory bound to NUMA node 8",
            "CUDA0: NUMA node 8 · host buffers on node 8",
            "CUDA1: NUMA node 8 · host buffers on node 8",
        ]
    );
    let gpus = vec![Some(0), Some(8)];
    let placed = Placement {
        mode: NumaMode::Auto,
        plan: auto(&t, &gpus, &free(&t)),
        gpus,
        error: None,
    };
    let lines = placed.lines();
    assert!(lines[0].starts_with("--numa auto: GPUs on NUMA nodes 0,8"));
    assert_eq!(lines[1], "CUDA0: NUMA node 0 · its host buffers on node 0");
    assert_eq!(lines[2], "CUDA1: NUMA node 8 · its host buffers on node 8");
    // No-op cases: one line, no GPU lines and no warning.
    let skip = Placement {
        mode: NumaMode::Auto,
        plan: Plan::Skip("one NUMA node".into()),
        gpus: vec![Some(0)],
        error: None,
    };
    assert_eq!(
        skip.lines(),
        ["--numa auto: nothing to bind (one NUMA node)"]
    );
    let failed = Placement {
        error: Some("Operation not permitted".into()),
        ..placed
    };
    assert!(failed.lines().last().unwrap().contains("could not bind"));
}

#[test]
fn node_masks() {
    let m = node_mask(&[0, 8, 65]).unwrap();
    assert_eq!(m[0], (1 << 0) | (1 << 8));
    assert_eq!(
        m[65 / WORD_BITS] & (1 << (65 % WORD_BITS)),
        1 << (65 % WORD_BITS)
    );
    assert!(node_mask(&[MASK_BITS as u32]).is_err());
}
