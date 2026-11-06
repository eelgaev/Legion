//! The choices the residency layer makes on GPUs that lack what an AC922 or a DGX Spark has:
//! no ATS or unified memory (kernels cannot read pageable host memory), no host NUMA node for
//! the GPU (sysfs `numa_node` -1, common on x86 PCIe boxes), or no peer access between two GPUs.
//! Pure functions over what the driver reports, so the rules are tested without a GPU; the
//! CUDA side (`cuda.rs`) asks the driver and acts on the answer.
//!
//! The rule for every choice: where the device has the capability (an AC922's NVLink + ATS, a
//! GB10's shared memory, a GPU with a known NUMA node), the answer is what the code did before
//! these fallbacks existed, so those machines take the same paths with the same bits.

/// The node value meaning "pinned host memory with no NUMA node" for the host allocators
/// (`CU_MEM_LOCATION_TYPE_HOST`, CUDA 12.2+ drivers): a GPU whose host NUMA node is unknown.
/// Never a real node id (Linux numbers nodes far below it).
pub const NO_NODE: u32 = u32::MAX;

/// Where a pinned host allocation through VMM (`cuMemCreate`) goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostLoc {
    /// `CU_MEM_LOCATION_TYPE_HOST_NUMA` on this node.
    Numa(u32),
    /// `CU_MEM_LOCATION_TYPE_HOST`: host memory, no node asked for.
    Host,
}

impl HostLoc {
    /// The location of a host allocator's `node` ([`NO_NODE`]: [`HostLoc::Host`]).
    pub fn of(node: u32) -> Self {
        if node == NO_NODE {
            HostLoc::Host
        } else {
            HostLoc::Numa(node)
        }
    }

    /// The node value the allocators take for this location.
    pub fn node(self) -> u32 {
        match self {
            HostLoc::Numa(n) => n,
            HostLoc::Host => NO_NODE,
        }
    }
}

/// A host allocator's `node` as a NUMA node: `None` for [`NO_NODE`] (host memory of no node).
pub fn known_node(node: u32) -> Option<u32> {
    (node != NO_NODE).then_some(node)
}

/// A host allocator's `node` for the log: its number, or "none" for [`NO_NODE`].
pub fn node_label(node: u32) -> String {
    known_node(node).map_or_else(|| "none".to_string(), |n| n.to_string())
}

/// The node pinned host memory for a GPU goes on, given its NUMA node as the driver or sysfs
/// report it (`known`), whether the driver can allocate host NUMA memory on a node
/// (`numa_ok`), and whether it can allocate host memory with no node (`host_ok`, asked only
/// when the node is unknown). `None`: no pinned host memory through VMM for this GPU.
///
/// - A known node: that node, as before (`None` when the driver cannot place memory there).
/// - No node: [`NO_NODE`] (`CU_MEM_LOCATION_TYPE_HOST`) when the driver takes it, else node 0
///   through `HOST_NUMA`, which the CUDA 12.0-12.x documentation asks for "on systems where
///   NUMA is not available".
pub fn host_node(
    known: Option<u32>,
    numa_ok: impl Fn(u32) -> bool,
    host_ok: impl FnOnce() -> bool,
) -> Option<u32> {
    match known {
        Some(n) => numa_ok(n).then_some(n),
        None if host_ok() => Some(NO_NODE),
        None => numa_ok(0).then_some(0),
    }
}

/// What [`host_node`] chose for a GPU without a known node, for the log line said once.
pub fn host_node_note(chosen: Option<u32>) -> &'static str {
    match chosen.map(HostLoc::of) {
        Some(HostLoc::Host) => {
            "the GPU reports no host NUMA node: pinned host memory (step-buffer spill, KV host \
             tier and tail, expert arena) uses CU_MEM_LOCATION_TYPE_HOST"
        }
        Some(HostLoc::Numa(_)) => {
            "the GPU reports no host NUMA node and the driver predates CU_MEM_LOCATION_TYPE_HOST: \
             pinned host memory goes through HOST_NUMA node 0"
        }
        None => {
            "the GPU reports no host NUMA node and the driver allocates no pinned host memory \
             through VMM: no step-buffer spill, KV host tier or expert arena"
        }
    }
}

/// How GPU kernels read routed experts kept in plain (pageable) host memory: the model file's
/// mapping or a CPU buffer, when there is no pinned expert arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostExpertRead {
    /// The GPU reads pageable memory (ATS on an AC922, the GB10's shared memory, HMM): in
    /// place, as before.
    InPlace,
    /// No pageable access: the expert ranges are registered with the driver
    /// (`cuMemHostRegister`, device-mapped), and read through the mapped pointer.
    /// `read_only`: with `CU_MEMHOSTREGISTER_READ_ONLY`, so the file's pages are pinned as they
    /// are rather than copied (the mapping is private).
    Register { read_only: bool },
    /// Neither: the load must stop with this reason.
    Refuse(&'static str),
}

/// [`HostExpertRead`] for GPUs that all read pageable memory (`pageable`) or not, can register
/// host memory (`register`, `CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED`) and register it
/// read-only (`read_only`, `CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED`).
pub fn host_expert_read(pageable: bool, register: bool, read_only: bool) -> HostExpertRead {
    if pageable {
        HostExpertRead::InPlace
    } else if register {
        HostExpertRead::Register { read_only }
    } else {
        HostExpertRead::Refuse(
            "this GPU can neither read pageable host memory (no ATS or HMM) nor register host \
             memory with the driver",
        )
    }
}

/// The byte ranges `(start, len)` widened to whole pages of `page` bytes and merged where they
/// touch or overlap, sorted: what `cuMemHostRegister` takes once each (a page registered twice
/// fails). Empty ranges are dropped.
pub fn page_spans(ranges: &[(usize, usize)], page: usize) -> Vec<(usize, usize)> {
    let page = page.max(1);
    let mut spans: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.1 > 0)
        .map(|&(start, len)| {
            let lo = start / page * page;
            let hi = (start + len).div_ceil(page) * page;
            (lo, hi)
        })
        .collect();
    spans.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (lo, hi) in spans {
        match out.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out.into_iter().map(|(lo, hi)| (lo, hi - lo)).collect()
}

/// The `(peer, device)` pairs of `--prefill-peer-stage` (per stage GPU `device`, the peer that
/// writes into its staged ranges, `pairs[device]`) where the peer cannot access the device's
/// memory (`can_access(peer, device)`, `cudaDeviceCanAccessPeer`).
pub fn missing_peer_access(
    pairs: &[Option<usize>],
    can_access: impl Fn(usize, usize) -> bool,
) -> Vec<(usize, usize)> {
    pairs
        .iter()
        .enumerate()
        .filter_map(|(d, p)| p.map(|p| (p, d)))
        .filter(|&(p, d)| !can_access(p, d))
        .collect()
}

/// Whether `--numa auto`'s one-node binding of the process's memory should be a preference
/// instead (`MPOL_PREFERRED`): on a discrete PCIe host (a DGX-1-style x86 box), where a model
/// bigger than one socket's free memory would otherwise fail to load. A coherent NVLink host (an
/// AC922), Grace and the GB10 keep the binding.
pub fn numa_auto_prefers(platform: super::Platform) -> bool {
    platform == super::Platform::DiscretePcie
}

#[cfg(test)]
mod tests;
