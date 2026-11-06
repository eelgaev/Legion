//! `legion serve --numa` (#422): binds the process to the NUMA node of the GPUs it serves with,
//! as `numactl --cpunodebind=N --membind=N` did from the outside, so the host memory the GPUs
//! read (pageable buffers, the page cache of the model file, the n-gram table) and the threads
//! feeding them sit on the GPUs' socket. On an AC922 the wrong node costs GPUs 2,3 a third of
//! their host bandwidth (93 vs 62 GiB/s).
//!
//! It runs once, from `main` before the async runtime starts, so every thread the process makes
//! afterwards inherits the CPU affinity and the memory policy (`set_mempolicy` is per thread).
//! GPUs on several nodes (a 4-GPU AC922) are not bound as a whole: each GPU's pinned buffers
//! already go to its own node (`cuMemCreate` on a host NUMA node), and its threads are pinned
//! there ([`per_gpu_threads`]).
//!
//! Plain Linux syscalls through `libc` over the sysfs topology of [`super::numa`], rather than
//! hwloc: nothing to link, and the node layout of the machines Legion targets is simple.

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::io;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};

use nix::sched::{CpuSet, sched_getaffinity, sched_setaffinity};
use nix::unistd::Pid;

use super::compress_cpulist;
use super::numa::{Topology, parse_cpulist};

/// `--numa`: where the process's CPUs and memory go.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum NumaMode {
    /// The GPUs' node, or each GPU's own when they sit on several.
    #[default]
    Auto,
    /// Leave placement to the OS (and any outer numactl).
    Off,
    /// Bind to these nodes (sorted, unique, at least one).
    Nodes(Vec<u32>),
}

impl FromStr for NumaMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim() {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            list => {
                let mut nodes = list
                    .split(',')
                    .map(|n| {
                        n.trim().parse::<u32>().map_err(|_| {
                            format!("expected auto, off or node ids like 0 or 0,8; got {s:?}")
                        })
                    })
                    .collect::<Result<Vec<u32>, String>>()?;
                nodes.sort_unstable();
                nodes.dedup();
                Ok(Self::Nodes(nodes))
            }
        }
    }
}

impl fmt::Display for NumaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Off => f.write_str("off"),
            Self::Nodes(nodes) => f.write_str(&node_list(nodes)),
        }
    }
}

/// Linux memory policy modes and `get_mempolicy` flags (`<linux/mempolicy.h>`).
pub const MPOL_DEFAULT: i32 = 0;
pub const MPOL_PREFERRED: i32 = 1;
pub const MPOL_BIND: i32 = 2;
pub const MPOL_INTERLEAVE: i32 = 3;
pub const MPOL_LOCAL: i32 = 4;
pub const MPOL_PREFERRED_MANY: i32 = 5;
pub const MPOL_WEIGHTED_INTERLEAVE: i32 = 6;
const MPOL_MODE_FLAGS: i32 = (1 << 15) | (1 << 14) | (1 << 13);
const MPOL_F_NODE: libc::c_ulong = 1;
const MPOL_F_ADDR: libc::c_ulong = 2;

/// Node masks passed to the kernel hold this many nodes (more than any kernel's `MAX_NUMNODES`).
const MASK_BITS: usize = 4096;
const WORD_BITS: usize = libc::c_ulong::BITS as usize;

/// A thread's memory policy: its mode (`MPOL_*`, mode flags removed) and nodes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemPolicy {
    pub mode: i32,
    pub nodes: Vec<u32>,
}

/// The placement the process started with: what an outer numactl, taskset or cpuset left it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inherited {
    /// The CPUs the process may run on.
    pub cpus: Vec<u32>,
    pub policy: MemPolicy,
    /// The cpuset's memory nodes (`Mems_allowed_list`; empty when unknown).
    pub mems_allowed: Vec<u32>,
}

impl Inherited {
    /// The calling thread's affinity and memory policy, and the process's cpuset memory nodes.
    pub fn read() -> io::Result<Self> {
        let set = sched_getaffinity(Pid::from_raw(0)).map_err(io::Error::from)?;
        let cpus = (0..CpuSet::count() as u32)
            .filter(|&cpu| set.is_set(cpu as usize).unwrap_or(false))
            .collect();
        let mems_allowed = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("Mems_allowed_list:"))
                    .map(parse_cpulist)
            })
            .unwrap_or_default();
        Ok(Self {
            cpus,
            policy: thread_policy()?,
            mems_allowed,
        })
    }
}

/// What `--numa` does on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// `--numa off`.
    Off,
    /// Nothing to bind, and nothing to warn about: why.
    Skip(String),
    /// An outer numactl, taskset or cpuset restricted the process: kept as it is.
    Outer {
        /// What restricts it, for the log.
        what: String,
        /// The nodes whose CPUs the process may run on.
        cpu_nodes: Vec<u32>,
        /// The nodes its memory is restricted to (`None`: not restricted).
        mem_nodes: Option<Vec<u32>>,
    },
    /// Bind the process's CPUs and memory (`MPOL_BIND`) to `nodes`.
    Bind { nodes: Vec<u32>, cpus: Vec<u32> },
    /// `--numa auto` on a discrete PCIe host ([`prefer_on_pcie`]): the process's CPUs bound to
    /// `node`, its memory preferred there (`MPOL_PREFERRED`), so a model bigger than the
    /// node's free memory spills to the other socket instead of failing to load.
    Prefer { node: u32, cpus: Vec<u32> },
    /// GPUs on several nodes: each GPU's threads and pinned buffers on its own node; the rest
    /// first-touch (never interleaved).
    PerGpu { nodes: Vec<u32> },
}

/// What `--numa` did: the plan, the GPUs it was made for, and an error applying it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub mode: NumaMode,
    pub plan: Plan,
    /// Per CUDA device in use (CUDA0, CUDA1, ...): its host NUMA node, if known.
    pub gpus: Vec<Option<u32>>,
    /// Binding failed (the process then runs as it started).
    pub error: Option<String>,
}

/// Plans `mode` for GPUs on `gpus` (per CUDA device, its node if known) on `topo`, from the
/// placement the process started with. Errors only for `--numa <nodes>` naming a node that does
/// not exist or has no CPUs.
pub fn plan(
    mode: &NumaMode,
    topo: &Topology,
    gpus: &[Option<u32>],
    inherited: &Inherited,
) -> Result<Plan, String> {
    if *mode == NumaMode::Off {
        return Ok(Plan::Off);
    }
    let cpu_nodes: Vec<u32> = topo
        .nodes
        .iter()
        .filter(|n| !n.cpus.is_empty())
        .map(|n| n.id)
        .collect();
    if let NumaMode::Nodes(want) = mode {
        for &id in want {
            match topo.nodes.iter().find(|n| n.id == id) {
                None => {
                    let all: Vec<u32> = topo.nodes.iter().map(|n| n.id).collect();
                    return Err(format!(
                        "--numa: no NUMA node {id} (this machine has {})",
                        node_list(&all)
                    ));
                }
                Some(n) if n.cpus.is_empty() => {
                    return Err(format!(
                        "--numa: NUMA node {id} has no CPUs (memory only); name a node with CPUs"
                    ));
                }
                Some(_) => {}
            }
        }
    }
    if cpu_nodes.len() < 2 {
        return Ok(Plan::Skip("one NUMA node".into()));
    }
    if let Some(outer) = outer(topo, inherited) {
        return Ok(outer);
    }
    let nodes = match mode {
        NumaMode::Nodes(want) => want.clone(),
        _ => {
            let mut nodes: Vec<u32> = gpus
                .iter()
                .flatten()
                .copied()
                .filter(|g| cpu_nodes.contains(g))
                .collect();
            nodes.sort_unstable();
            nodes.dedup();
            match nodes.len() {
                0 if gpus.is_empty() => return Ok(Plan::Skip("no GPUs in use".into())),
                0 => return Ok(Plan::Skip("the GPUs' NUMA nodes are unknown".into())),
                1 => nodes,
                _ => return Ok(Plan::PerGpu { nodes }),
            }
        }
    };
    let cpus: BTreeSet<u32> = topo
        .nodes
        .iter()
        .filter(|n| nodes.contains(&n.id))
        .flat_map(|n| n.cpus.iter().copied())
        .collect();
    Ok(Plan::Bind {
        nodes,
        cpus: cpus.into_iter().collect(),
    })
}

/// `plan` for `--numa auto` on a host where binding memory to one node is the wrong default
/// (`prefers`: [`crate::residency::fallback::numa_auto_prefers`], a discrete PCIe host): its
/// one-node [`Plan::Bind`] becomes a [`Plan::Prefer`] of that node, the CPUs bound as before.
/// Every other mode, plan or host keeps the plan as it is (an AC922 and the GB10 bind).
pub fn prefer_on_pcie(plan: Plan, mode: &NumaMode, prefers: bool) -> Plan {
    match plan {
        Plan::Bind { nodes, cpus } if prefers && *mode == NumaMode::Auto && nodes.len() == 1 => {
            Plan::Prefer {
                node: nodes[0],
                cpus,
            }
        }
        plan => plan,
    }
}

/// Whether `--numa auto` should prefer rather than bind memory on this host: only an x86 box
/// is ever a discrete PCIe host here (an AC922 is POWER, Grace and the GB10 are Arm), so the
/// others never ask NVML.
fn auto_prefers() -> bool {
    std::env::consts::ARCH == "x86_64"
        && crate::residency::fallback::numa_auto_prefers(crate::residency::Platform::of(
            &super::HardwareInfo::detect(),
        ))
}

/// The outer restriction the process started under, if any: CPUs short of the machine's, a
/// memory policy other than the default, or cpuset memory nodes short of the machine's.
fn outer(topo: &Topology, inherited: &Inherited) -> Option<Plan> {
    let cpus_restricted = topo
        .nodes
        .iter()
        .flat_map(|n| &n.cpus)
        .any(|c| !inherited.cpus.contains(c));
    let policy = &inherited.policy;
    let policy_set = policy.mode != MPOL_DEFAULT;
    let mems_restricted = !inherited.mems_allowed.is_empty()
        && topo
            .nodes
            .iter()
            .filter(|n| n.mem_total > 0)
            .any(|n| !inherited.mems_allowed.contains(&n.id));
    if !(cpus_restricted || policy_set || mems_restricted) {
        return None;
    }
    let mut what = Vec::new();
    if cpus_restricted {
        what.push(format!("CPUs {}", compress_cpulist(&inherited.cpus)));
    }
    if policy_set {
        what.push(if policy.nodes.is_empty() {
            format!("memory policy {}", policy_name(policy.mode))
        } else {
            format!(
                "memory policy {} {}",
                policy_name(policy.mode),
                node_list(&policy.nodes)
            )
        });
    }
    if mems_restricted {
        what.push(format!(
            "cpuset memory nodes {}",
            node_list(&inherited.mems_allowed)
        ));
    }
    let cpu_nodes = topo
        .nodes
        .iter()
        .filter(|n| n.cpus.iter().any(|c| inherited.cpus.contains(c)))
        .map(|n| n.id)
        .collect();
    let mem_nodes = if policy_set && !policy.nodes.is_empty() {
        Some(policy.nodes.clone())
    } else if mems_restricted {
        Some(inherited.mems_allowed.clone())
    } else {
        None
    };
    Some(Plan::Outer {
        what: what.join(", "),
        cpu_nodes,
        mem_nodes,
    })
}

fn policy_name(mode: i32) -> &'static str {
    match mode {
        MPOL_DEFAULT => "default",
        MPOL_PREFERRED => "preferred",
        MPOL_BIND => "bind",
        MPOL_INTERLEAVE => "interleave",
        MPOL_LOCAL => "local",
        MPOL_PREFERRED_MANY => "preferred-many",
        MPOL_WEIGHTED_INTERLEAVE => "weighted-interleave",
        _ => "unknown",
    }
}

/// `[0, 8]` -> `0,8`.
fn node_list(nodes: &[u32]) -> String {
    nodes
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

impl Placement {
    /// Plans `mode` for the GPUs on `gpus` and applies it to the calling thread's memory policy
    /// and every thread's CPU affinity. Call it before other threads start, so they inherit
    /// the policy. A machine whose topology or placement cannot be read is left alone.
    pub fn apply(mode: NumaMode, gpus: Vec<Option<u32>>) -> Result<Self, String> {
        let state = Topology::detect().and_then(|t| Ok((t, Inherited::read()?)));
        let plan = match &state {
            Ok((topo, inherited)) => plan(&mode, topo, &gpus, inherited)?,
            Err(e) => Plan::Skip(format!("cannot read the NUMA topology: {e}")),
        };
        let plan = match plan {
            Plan::Bind { .. } if mode == NumaMode::Auto => {
                prefer_on_pcie(plan, &mode, auto_prefers())
            }
            plan => plan,
        };
        let error = match &plan {
            Plan::Bind { nodes, cpus } => bind_thread_memory(nodes)
                .and_then(|()| set_process_affinity(cpus))
                .err()
                .map(|e| e.to_string()),
            Plan::Prefer { node, cpus } => prefer_thread_memory(*node)
                .and_then(|()| set_process_affinity(cpus))
                .err()
                .map(|e| e.to_string()),
            Plan::PerGpu { .. } => {
                PER_GPU_THREADS.store(true, Ordering::Relaxed);
                None
            }
            _ => None,
        };
        Ok(Self {
            mode,
            plan,
            gpus,
            error,
        })
    }

    /// The summary for the log and the startup report.
    pub fn summary(&self) -> String {
        let mode = &self.mode;
        match &self.plan {
            Plan::Off => "--numa off: placement left to the OS".into(),
            Plan::Skip(why) => format!("--numa {mode}: nothing to bind ({why})"),
            Plan::Outer { what, .. } => {
                format!("--numa {mode}: kept the placement the process started with ({what})")
            }
            Plan::Bind { nodes, cpus } => format!(
                "--numa {mode}: CPUs {} and memory bound to NUMA node {}",
                compress_cpulist(cpus),
                node_list(nodes)
            ),
            Plan::Prefer { node, cpus } => format!(
                "--numa {mode}: CPUs {} bound to NUMA node {node}, memory preferred there (a \
                 PCIe host: memory past the node's goes to the others)",
                compress_cpulist(cpus)
            ),
            Plan::PerGpu { nodes } => format!(
                "--numa {mode}: GPUs on NUMA nodes {}: each GPU's threads and pinned host \
                 buffers on its own node, the rest first-touch",
                node_list(nodes)
            ),
        }
    }

    /// The startup report's lines: the summary, where each GPU's host buffers go, and warnings.
    pub fn lines(&self) -> Vec<String> {
        let mut lines = vec![self.summary()];
        if matches!(
            self.plan,
            Plan::Bind { .. } | Plan::Prefer { .. } | Plan::PerGpu { .. } | Plan::Outer { .. }
        ) {
            for (i, gpu) in self.gpus.iter().enumerate() {
                let node = gpu.map_or_else(|| "unknown".into(), |g| g.to_string());
                let host = match (&self.plan, gpu) {
                    (Plan::Bind { nodes, .. }, Some(g)) if nodes.contains(g) => {
                        format!("host buffers on node {g}")
                    }
                    (Plan::Bind { nodes, .. }, _) => format!(
                        "pinned buffers on its node, other host memory on node {}",
                        node_list(nodes)
                    ),
                    (Plan::Prefer { node, .. }, Some(g)) if node == g => {
                        format!("host buffers on node {g}")
                    }
                    (Plan::Prefer { node, .. }, _) => format!(
                        "pinned buffers on its node, other host memory preferably on node {node}"
                    ),
                    (Plan::PerGpu { .. }, Some(g)) => format!("its host buffers on node {g}"),
                    (
                        Plan::Outer {
                            mem_nodes: Some(m), ..
                        },
                        _,
                    ) => format!("host memory on node {} (inherited)", node_list(m)),
                    _ => "host memory placed by first touch".into(),
                };
                lines.push(format!("CUDA{i}: NUMA node {node} · {host}"));
            }
        }
        lines.extend(self.warnings().into_iter().map(|w| format!("WARN {w}")));
        lines
    }

    /// Host memory or threads away from a GPU's node, and a failure to bind.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, g) in self.gpus.iter().enumerate() {
            let Some(g) = g else { continue };
            match &self.plan {
                Plan::Bind { nodes, .. } if !nodes.contains(g) => out.push(format!(
                    "CUDA{i} is on NUMA node {g} but --numa {} binds host memory to node {}: \
                     what it reads from there crosses the socket",
                    self.mode,
                    node_list(nodes)
                )),
                Plan::Prefer { node, .. } if node != g => out.push(format!(
                    "CUDA{i} is on NUMA node {g} but --numa {} prefers host memory on node \
                     {node}: what it reads from there crosses the socket",
                    self.mode
                )),
                Plan::Outer {
                    cpu_nodes,
                    mem_nodes,
                    ..
                } => {
                    if let Some(m) = mem_nodes.as_ref().filter(|m| !m.contains(g)) {
                        out.push(format!(
                            "CUDA{i} is on NUMA node {g} but the process's memory is restricted \
                             to node {} (an outer numactl or cpuset)",
                            node_list(m)
                        ));
                    }
                    if !cpu_nodes.contains(g) {
                        out.push(format!(
                            "CUDA{i} is on NUMA node {g} but the process's CPUs are on node {} \
                             (an outer numactl, taskset or cpuset)",
                            node_list(cpu_nodes)
                        ));
                    }
                }
                _ => {}
            }
        }
        if let Some(e) = &self.error {
            out.push(format!(
                "--numa {}: could not bind the process ({e}); it runs unbound",
                self.mode
            ));
        }
        out
    }
}

static PER_GPU_THREADS: AtomicBool = AtomicBool::new(false);

/// Whether `--numa` asked for each GPU's threads on its own node (GPUs on several nodes): the
/// load then pins every stage's threads to its GPU's node even without a pinned expert arena,
/// which pins them anyway.
pub fn per_gpu_threads() -> bool {
    PER_GPU_THREADS.load(Ordering::Relaxed)
}

/// The host NUMA node of CUDA device `device` (ggml's index, after `CUDA_VISIBLE_DEVICES`),
/// from the driver without creating a context; `None` when unknown or without CUDA.
#[cfg(feature = "cuda")]
pub fn gpu_node(device: usize) -> Option<u32> {
    crate::residency::cuda::device_numa_node(device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_node(_device: usize) -> Option<u32> {
    None
}

/// The CUDA devices visible to the process, from the driver without creating a context; 0
/// without CUDA.
#[cfg(feature = "cuda")]
pub fn gpu_count() -> usize {
    crate::residency::cuda::device_count()
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_count() -> usize {
    0
}

/// Places the calling thread, and the threads it starts from now on, on NUMA node `node`: its
/// CPUs (without CPU 0's core, [`super::numa::helper_cpus`]) and its memory (`MPOL_BIND`). For a
/// replica's load (`--tp-replicas`), whose engine thread and helpers it starts inherit both.
pub fn bind_thread_to_node(node: u32) -> io::Result<()> {
    let topo = super::numa::Topology::detect()?;
    let cpus = super::numa::helper_cpus(&topo, node);
    if cpus.is_empty() {
        return Err(io::Error::other(format!(
            "NUMA node {node} has no CPUs to run on"
        )));
    }
    bind_thread_memory(&[node])?;
    super::numa::pin_current_thread(&cpus).map(|_| ())
}

fn node_mask(nodes: &[u32]) -> io::Result<Vec<libc::c_ulong>> {
    let mut mask = vec![0 as libc::c_ulong; MASK_BITS / WORD_BITS];
    for &n in nodes {
        let n = n as usize;
        if n >= MASK_BITS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("NUMA node {n} out of range"),
            ));
        }
        mask[n / WORD_BITS] |= 1 << (n % WORD_BITS);
    }
    Ok(mask)
}

/// Binds the calling thread's future allocations (and those of the threads it starts) to
/// `nodes` (`MPOL_BIND`), as `numactl --membind` does.
pub fn bind_thread_memory(nodes: &[u32]) -> io::Result<()> {
    let mask = node_mask(nodes)?;
    // SAFETY: the mask holds MASK_BITS bits; the kernel reads at most that many (it drops the
    // last of `maxnode`, hence the + 1, as libnuma passes it).
    let r = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            MPOL_BIND as libc::c_long,
            mask.as_ptr(),
            (MASK_BITS + 1) as libc::c_ulong,
        )
    };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Prefers `node` for the calling thread's future allocations (and those of the threads it
/// starts), falling back to the other nodes when it is full (`MPOL_PREFERRED`), as
/// `numactl --preferred` does.
pub fn prefer_thread_memory(node: u32) -> io::Result<()> {
    let mask = node_mask(&[node])?;
    // SAFETY: as `bind_thread_memory`.
    let r = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            MPOL_PREFERRED as libc::c_long,
            mask.as_ptr(),
            (MASK_BITS + 1) as libc::c_ulong,
        )
    };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// The calling thread's memory policy.
pub fn thread_policy() -> io::Result<MemPolicy> {
    let mut mode: libc::c_int = 0;
    let mut mask = vec![0 as libc::c_ulong; MASK_BITS / WORD_BITS];
    // SAFETY: writes one int and at most MASK_BITS bits of mask.
    let r = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut mode as *mut libc::c_int,
            mask.as_mut_ptr(),
            (MASK_BITS + 1) as libc::c_ulong,
            std::ptr::null_mut::<libc::c_void>(),
            0 as libc::c_ulong,
        )
    };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    let nodes = (0..MASK_BITS)
        .filter(|&n| mask[n / WORD_BITS] & (1 << (n % WORD_BITS)) != 0)
        .map(|n| n as u32)
        .collect();
    Ok(MemPolicy {
        mode: mode & !MPOL_MODE_FLAGS,
        nodes,
    })
}

/// The NUMA node of the page holding `addr` (it must be touched first).
pub fn page_node(addr: *const u8) -> io::Result<u32> {
    let mut node: libc::c_int = -1;
    // SAFETY: writes one int; the kernel only looks `addr` up in this process's mappings.
    let r = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut node as *mut libc::c_int,
            std::ptr::null_mut::<libc::c_ulong>(),
            0 as libc::c_ulong,
            addr as *mut libc::c_void,
            MPOL_F_NODE | MPOL_F_ADDR,
        )
    };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    u32::try_from(node).map_err(|_| io::Error::other("no NUMA node for the page"))
}

/// Sets every thread of the process to `cpus` (threads made later inherit it).
fn set_process_affinity(cpus: &[u32]) -> io::Result<()> {
    let mut mask = CpuSet::new();
    for &cpu in cpus {
        mask.set(cpu as usize)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "CPU out of range"))?;
    }
    sched_setaffinity(Pid::from_raw(0), &mask).map_err(io::Error::from)?;
    // Threads already running (a driver's, say) too; one that has exited meanwhile is fine.
    if let Ok(tasks) = fs::read_dir("/proc/self/task") {
        for task in tasks.flatten() {
            if let Some(tid) = task.file_name().to_str().and_then(|t| t.parse().ok()) {
                let _ = sched_setaffinity(Pid::from_raw(tid), &mask);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
