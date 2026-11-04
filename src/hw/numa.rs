//! Linux NUMA discovery and CPU affinity for node-local first-touch work.

use std::fs;
use std::io;
use std::path::{Component, Path};
use std::thread;

use nix::sched::{CpuSet, sched_getaffinity, sched_getcpu, sched_setaffinity};
use nix::unistd::Pid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaNode {
    pub id: u32,
    pub cpus: Vec<u32>,
    /// Total memory in bytes (zero when unavailable).
    pub mem_total: u64,
    /// Free memory in bytes (zero when unavailable).
    pub mem_free: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topology {
    pub nodes: Vec<NumaNode>,
}

impl Topology {
    /// Discover nodes under `/sys/devices/system/node`.
    pub fn detect() -> io::Result<Self> {
        Self::from_sysfs(Path::new("/sys/devices/system/node"))
    }

    /// Read a node directory, sorting nodes by their numeric IDs.
    /// Missing CPU lists are skipped; missing memory information is reported as zero.
    /// If no nodes can be discovered, return node 0 with all online CPUs.
    pub fn from_sysfs(root: &Path) -> io::Result<Self> {
        let entries = match fs::read_dir(root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return fallback_topology(),
            Err(e) => return Err(e),
        };
        let mut nodes = Vec::new();
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let Some(id) = name
                .to_str()
                .and_then(|s| s.strip_prefix("node"))
                .and_then(|s| s.parse::<u32>().ok())
            else {
                continue;
            };
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let Some(cpulist) = read_optional(&entry.path().join("cpulist"))? else {
                continue;
            };
            let meminfo = read_optional(&entry.path().join("meminfo"))?.unwrap_or_default();
            nodes.push(NumaNode {
                id,
                cpus: parse_cpulist(&cpulist),
                mem_total: memory_bytes(&meminfo, "MemTotal:")?,
                mem_free: memory_bytes(&meminfo, "MemFree:")?,
            });
        }
        if nodes.is_empty() {
            return fallback_topology();
        }
        nodes.sort_by_key(|node| node.id);
        Ok(Self { nodes })
    }

    pub fn node_of_cpu(&self, cpu: u32) -> Option<u32> {
        self.nodes
            .iter()
            .find(|node| node.cpus.contains(&cpu))
            .map(|node| node.id)
    }
}

fn read_optional(path: &Path) -> io::Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

fn memory_bytes(info: &str, key: &str) -> io::Result<u64> {
    for line in info.lines() {
        let mut words = line.split_whitespace();
        if words.any(|word| word == key) {
            return words
                .next()
                .and_then(|value| value.parse::<u64>().ok())
                .filter(|_| words.next() == Some("kB"))
                .and_then(|kb| kb.checked_mul(1024))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid NUMA memory size")
                });
        }
    }
    Ok(0)
}

fn fallback_topology() -> io::Result<Topology> {
    let mut cpus = fs::read_to_string("/sys/devices/system/cpu/online")
        .map(|s| parse_cpulist(&s))
        .unwrap_or_default();
    if cpus.is_empty() {
        cpus.extend(0..thread::available_parallelism()?.get() as u32);
    }
    Ok(Topology {
        nodes: vec![NumaNode {
            id: 0,
            cpus,
            mem_total: 0,
            mem_free: 0,
        }],
    })
}

/// NUMA node of a PCI device; absent, unknown (-1), or malformed entries return `None`.
/// A node's free memory in bytes now (`MemFree` of its sysfs meminfo), if known.
pub fn node_mem_free(node: u32) -> Option<u64> {
    let info = fs::read_to_string(format!("/sys/devices/system/node/node{node}/meminfo")).ok()?;
    memory_bytes(&info, "MemFree:").ok().filter(|&b| b > 0)
}

/// The nodes with CPUs, when `topo` also has memory-only nodes (memory, no CPUs: on an AC922
/// the GPUs' own memory, onlined as NUMA nodes); `None` when every node has CPUs. Binding page
/// cache to these keeps it out of the GPUs' memory: when the CPU nodes are short of free
/// memory the kernel reclaims there instead of spilling onto the GPU nodes (#772).
pub fn cpu_nodes_beside_memory_only(topo: &Topology) -> Option<Vec<u32>> {
    let memory_only = topo
        .nodes
        .iter()
        .any(|n| n.cpus.is_empty() && n.mem_total > 0);
    let cpu_nodes: Vec<u32> = topo
        .nodes
        .iter()
        .filter(|n| !n.cpus.is_empty())
        .map(|n| n.id)
        .collect();
    (memory_only && !cpu_nodes.is_empty()).then_some(cpu_nodes)
}

/// Page cache (`FilePages`) of at least `min` bytes on `topo`'s memory-only nodes, as `(node,
/// bytes)`, read from `/sys/devices/system/node`. On an AC922 that is GPU memory the KV pool
/// and the expert cache cannot have, left by other processes' file reads (#772).
pub fn memory_only_page_cache(topo: &Topology, min: u64) -> Vec<(u32, u64)> {
    memory_only_page_cache_in(Path::new("/sys/devices/system/node"), topo, min)
}

/// [`memory_only_page_cache`] under a fixture directory of `node<N>` directories.
pub fn memory_only_page_cache_in(root: &Path, topo: &Topology, min: u64) -> Vec<(u32, u64)> {
    topo.nodes
        .iter()
        .filter(|n| n.cpus.is_empty() && n.mem_total > 0)
        .filter_map(|n| {
            let info = fs::read_to_string(root.join(format!("node{}", n.id)).join("meminfo"));
            let bytes = memory_bytes(&info.ok()?, "FilePages:").ok()?;
            (bytes >= min.max(1)).then_some((n.id, bytes))
        })
        .collect()
}

pub fn pci_numa_node(bdf: &str) -> Option<u32> {
    pci_numa_node_in(Path::new("/sys/bus/pci/devices"), bdf)
}

/// Like [`pci_numa_node`], with a fixture directory containing PCI device directories.
pub fn pci_numa_node_in(root: &Path, bdf: &str) -> Option<u32> {
    let mut components = Path::new(bdf).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return None;
    }
    fs::read_to_string(root.join(bdf).join("numa_node"))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Parse `0-3,8,10-11` CPU lists into sorted, unique IDs.
/// Whitespace is ignored and malformed or descending entries are skipped.
pub fn parse_cpulist(s: &str) -> Vec<u32> {
    let mut cpus = Vec::new();
    for part in s.split(',').map(str::trim) {
        if let Some((first, last)) = part.split_once('-') {
            if let (Ok(first), Ok(last)) = (first.trim().parse::<u32>(), last.trim().parse::<u32>())
            {
                cpus.extend(first..=last);
            }
        } else if let Ok(cpu) = part.parse() {
            cpus.push(cpu);
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

/// The calling thread's CPUs.
pub fn current_affinity() -> io::Result<Vec<u32>> {
    let set = sched_getaffinity(Pid::from_raw(0)).map_err(io::Error::from)?;
    Ok((0..CpuSet::count() as u32)
        .filter(|&cpu| set.is_set(cpu as usize).unwrap_or(false))
        .collect())
}

/// Pin the calling thread, returning its entire previous mask for restoration.
/// Empty masks and IDs outside `CpuSet`'s capacity return `InvalidInput`.
/// Linux may restrict the requested mask to online CPUs permitted by the cpuset.
pub fn pin_current_thread(cpus: &[u32]) -> io::Result<Vec<u32>> {
    let invalid = || io::Error::new(io::ErrorKind::InvalidInput, "invalid CPU mask");
    if cpus.is_empty() {
        return Err(invalid());
    }
    let mut mask = CpuSet::new();
    for &cpu in cpus {
        mask.set(cpu as usize).map_err(|_| invalid())?;
    }
    let previous = current_affinity()?;
    sched_setaffinity(Pid::from_raw(0), &mask).map_err(io::Error::from)?;
    Ok(previous)
}

fn node_cpus(topo: &Topology, node: u32) -> &[u32] {
    &topo
        .nodes
        .iter()
        .find(|n| n.id == node)
        .unwrap_or_else(|| panic!("unknown NUMA node {node}"))
        .cpus
}

/// Spawn a thread, explicitly replace its inherited affinity with the node's CPU
/// list, and return `f`'s result. The caller's affinity is unchanged.
///
/// Panics if the node is unknown, pinning fails, or `f` panics. CPU-less nodes
/// cannot run work. Missing sysfs memory information does not affect execution.
pub fn run_on_node<T: Send + 'static>(
    topo: &Topology,
    node: u32,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let cpus = node_cpus(topo, node).to_vec();
    thread::spawn(move || {
        // Linux inherits the creator's affinity: always replace even a single-CPU pin.
        pin_current_thread(&cpus).expect("pinning NUMA worker");
        f()
    })
    .join()
    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

/// CPUs for a long-lived thread working for a device on `node` (the engine thread, a pipeline
/// stage's launches, and what they spawn): the node's CPUs this process may run on, without
/// CPU 0's core, where the kernel's housekeeping and many interrupts land. Empty when that
/// leaves none (the caller then keeps its affinity). Call it from a thread with the process's
/// own affinity (e.g. at load).
///
/// Adapted from Strata-AC922 (eelgaev/Strata-AC922, commits 4f00ead and 46ccb94, MIT), which
/// moved its helper, stage and gather threads off CPU 0 and the host thread's core.
pub fn helper_cpus(topo: &Topology, node: u32) -> Vec<u32> {
    let Some(n) = topo.nodes.iter().find(|n| n.id == node) else {
        return Vec::new();
    };
    let allowed = current_affinity().unwrap_or_default();
    let cpu0 = fs::read_to_string("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list")
        .map(|s| parse_cpulist(&s))
        .unwrap_or_default();
    helper_set(&n.cpus, &allowed, &cpu0)
}

/// The CPUs of `node` that are `allowed`, without CPU 0 and the CPUs in `avoid`.
fn helper_set(node: &[u32], allowed: &[u32], avoid: &[u32]) -> Vec<u32> {
    node.iter()
        .copied()
        .filter(|c| *c != 0 && allowed.contains(c) && !avoid.contains(c))
        .collect()
}

/// `cpus` without `host_cpu` and its SMT `siblings`.
pub(crate) fn copy_worker_cpus(cpus: &[u32], host_cpu: u32, siblings: &[u32]) -> Vec<u32> {
    cpus.iter()
        .copied()
        .filter(|cpu| *cpu != host_cpu && !siblings.contains(cpu))
        .collect()
}

/// Copy equal-sized chunks (differing by at most one byte) on node-pinned threads.
/// Copy workers exclude the caller's current core, including SMT siblings when
/// sysfs exposes them. If that leaves no CPUs, use one node-pinned thread.
/// An explicit pin releases the caller's inherited single-CPU affinity.
///
/// Panics on unequal buffer lengths, zero threads, an unknown node, or a pinning
/// failure. Empty buffers need no workers. Only the destination's first write
/// establishes first-touch placement; existing pages are not migrated.
pub fn parallel_fill_on_node(
    topo: &Topology,
    node: u32,
    threads: usize,
    buf: &mut [u8],
    src: &[u8],
) {
    assert_eq!(buf.len(), src.len(), "buffer lengths must match");
    assert!(threads > 0, "thread count must be positive");
    let cpus = node_cpus(topo, node);
    if buf.is_empty() {
        return;
    }
    let workers = if let Ok(host_cpu) = sched_getcpu() {
        let siblings = fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{host_cpu}/topology/thread_siblings_list"
        ))
        .map(|s| parse_cpulist(&s))
        .unwrap_or_default();
        copy_worker_cpus(cpus, host_cpu as u32, &siblings)
    } else {
        cpus.to_vec()
    };
    let (workers, count) = if workers.is_empty() {
        (cpus, 1)
    } else {
        (workers.as_slice(), threads.min(buf.len()))
    };
    let size = buf.len() / count;
    let remainder = buf.len() % count;
    thread::scope(|scope| {
        let mut tail = buf;
        let mut source = src;
        for i in 0..count {
            let len = size + usize::from(i < remainder);
            let (chunk, rest) = tail.split_at_mut(len);
            let (input, remaining) = source.split_at(len);
            tail = rest;
            source = remaining;
            scope.spawn(move || {
                pin_current_thread(workers).expect("pinning NUMA copy worker");
                chunk.copy_from_slice(input);
            });
        }
    });
}

#[cfg(test)]
mod tests;
