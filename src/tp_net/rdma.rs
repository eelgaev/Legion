//! The all-reduce's RDMA transport (RoCE v2 on the NIC the data connection runs over; #619
//! step 3): one reliable-connected queue pair per peer. A writer posts its partial as an RDMA
//! write with immediate (the all-reduce's sequence number) straight from the pinned send
//! buffer into the peer's pinned receive buffer; the peer's reader polls its completion queue,
//! with no interrupt and no kernel on the path, and raises the GPU's flag. Measured on two DGX
//! Sparks over their ConnectX-7 link: about 5 µs one way for a decode step's 21 KB partial,
//! against 130-600 µs for TCP once the cores idle between all-reduces.
//!
//! libibverbs (rdma-core, dual GPL-2/BSD-2: used under BSD) is loaded at run time, so the
//! build needs neither its headers nor its library, and a box without it falls back to TCP.
//! The structs below are the parts of `<infiniband/verbs.h>` this module reads or fills, with
//! their sizes checked at compile time against that header's (rdma-core 50, LP64). The
//! data-path calls (`ibv_post_send`, `ibv_post_recv`, `ibv_poll_cq`) are inline functions in
//! the header: they call the context's ops table, as here.
//!
//! The device, port and GID come from sysfs: the RoCE v2 GID that holds the local address of
//! the peer's TCP data connection ([`find_gid`]). The queue pairs are set up over that
//! connection ([`Endpoint::connect`]), and so are the memory regions' keys
//! ([`super::Comm::register_region`]).

#[cfg(test)]
mod tests;

use std::ffi::{CStr, c_void};
use std::io::{Read, Write};
use std::net::{IpAddr, TcpStream};
use std::os::raw::{c_char, c_int, c_uint};
use std::path::Path;
use std::sync::{Arc, OnceLock};

// The constants of verbs.h used here.
const IBV_QPT_RC: c_uint = 2;
const IBV_QPS_INIT: c_uint = 1;
const IBV_QPS_RTR: c_uint = 2;
const IBV_QPS_RTS: c_uint = 3;
const IBV_ACCESS_LOCAL_WRITE: c_int = 1;
const IBV_ACCESS_REMOTE_WRITE: c_int = 1 << 1;
const IBV_QP_STATE: c_int = 1;
const IBV_QP_ACCESS_FLAGS: c_int = 1 << 3;
const IBV_QP_PKEY_INDEX: c_int = 1 << 4;
const IBV_QP_PORT: c_int = 1 << 5;
const IBV_QP_AV: c_int = 1 << 7;
const IBV_QP_PATH_MTU: c_int = 1 << 8;
const IBV_QP_TIMEOUT: c_int = 1 << 9;
const IBV_QP_RETRY_CNT: c_int = 1 << 10;
const IBV_QP_RNR_RETRY: c_int = 1 << 11;
const IBV_QP_RQ_PSN: c_int = 1 << 12;
const IBV_QP_MAX_QP_RD_ATOMIC: c_int = 1 << 13;
const IBV_QP_MIN_RNR_TIMER: c_int = 1 << 15;
const IBV_QP_SQ_PSN: c_int = 1 << 16;
const IBV_QP_MAX_DEST_RD_ATOMIC: c_int = 1 << 17;
const IBV_QP_DEST_QPN: c_int = 1 << 20;
const IBV_WR_RDMA_WRITE_WITH_IMM: c_uint = 1;
const IBV_SEND_SIGNALED: c_uint = 1 << 1;
const IBV_WC_SUCCESS: c_uint = 0;
const IBV_WC_RECV_RDMA_WITH_IMM: c_uint = 129;
const IBV_WC_WITH_IMM: c_uint = 1 << 1;
/// `IBV_MTU_256` .. `IBV_MTU_4096`.
const IBV_MTU_MAX: u32 = 5;

/// Receive requests kept posted per queue pair (a write with immediate takes one each).
const RECV_DEPTH: u32 = 64;
const SEND_DEPTH: u32 = 16;

#[repr(C)]
struct IbvSge {
    addr: u64,
    length: u32,
    lkey: u32,
}

/// `struct ibv_send_wr`: the RDMA member of its `wr` union; the rest is zeros.
#[repr(C)]
struct IbvSendWr {
    wr_id: u64,
    next: *mut IbvSendWr,
    sg_list: *mut IbvSge,
    num_sge: c_int,
    opcode: c_uint,
    send_flags: c_uint,
    imm_data: u32,
    remote_addr: u64,
    rkey: u32,
    _rest: [u8; 76],
}

#[repr(C)]
struct IbvRecvWr {
    wr_id: u64,
    next: *mut IbvRecvWr,
    sg_list: *mut IbvSge,
    num_sge: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct IbvWc {
    wr_id: u64,
    status: c_uint,
    opcode: c_uint,
    vendor_err: u32,
    byte_len: u32,
    imm_data: u32,
    qp_num: u32,
    src_qp: u32,
    wc_flags: c_uint,
    pkey_index: u16,
    slid: u16,
    sl: u8,
    dlid_path_bits: u8,
}

/// The head of `struct ibv_context`: its ops table (27 function pointers).
#[repr(C)]
struct IbvContext {
    device: *mut c_void,
    ops: [usize; 27],
}
const OP_POLL_CQ: usize = 11;
const OP_POST_SEND: usize = 25;
const OP_POST_RECV: usize = 26;

/// The head of `struct ibv_qp`.
#[repr(C)]
struct IbvQp {
    context: *mut IbvContext,
    _ptrs: [*mut c_void; 5],
    handle: u32,
    qp_num: u32,
}

/// The head of `struct ibv_cq`.
#[repr(C)]
struct IbvCq {
    context: *mut IbvContext,
}

/// The head of `struct ibv_mr`.
#[repr(C)]
struct IbvMr {
    context: *mut IbvContext,
    pd: *mut c_void,
    addr: *mut c_void,
    length: usize,
    handle: u32,
    lkey: u32,
    rkey: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IbvQpCap {
    max_send_wr: u32,
    max_recv_wr: u32,
    max_send_sge: u32,
    max_recv_sge: u32,
    max_inline_data: u32,
}

#[repr(C)]
struct IbvQpInitAttr {
    qp_context: *mut c_void,
    send_cq: *mut IbvCq,
    recv_cq: *mut IbvCq,
    srq: *mut c_void,
    cap: IbvQpCap,
    qp_type: c_uint,
    sq_sig_all: c_int,
}

/// `union ibv_gid` (8-byte aligned by its `global` member).
#[repr(C, align(8))]
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct Gid(pub [u8; 16]);

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IbvGlobalRoute {
    dgid: Gid,
    flow_label: u32,
    sgid_index: u8,
    hop_limit: u8,
    traffic_class: u8,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IbvAhAttr {
    grh: IbvGlobalRoute,
    dlid: u16,
    sl: u8,
    src_path_bits: u8,
    static_rate: u8,
    is_global: u8,
    port_num: u8,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct IbvQpAttr {
    qp_state: c_uint,
    cur_qp_state: c_uint,
    path_mtu: c_uint,
    path_mig_state: c_uint,
    qkey: u32,
    rq_psn: u32,
    sq_psn: u32,
    dest_qp_num: u32,
    qp_access_flags: c_uint,
    cap: IbvQpCap,
    ah_attr: IbvAhAttr,
    alt_ah_attr: IbvAhAttr,
    pkey_index: u16,
    alt_pkey_index: u16,
    en_sqd_async_notify: u8,
    sq_draining: u8,
    max_rd_atomic: u8,
    max_dest_rd_atomic: u8,
    min_rnr_timer: u8,
    port_num: u8,
    timeout: u8,
    retry_cnt: u8,
    rnr_retry: u8,
    alt_port_num: u8,
    alt_timeout: u8,
    rate_limit: u32,
}

/// `struct ibv_port_attr` up to `active_mtu`, padded to its size.
#[repr(C)]
#[derive(Default)]
struct IbvPortAttr {
    state: c_uint,
    max_mtu: c_uint,
    active_mtu: c_uint,
    _rest: [u32; 11],
}

// The header's sizes (rdma-core 50, LP64), so a wrong field shows at build time.
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<IbvSge>() == 16);
    assert!(size_of::<IbvSendWr>() == 128);
    assert!(offset_of!(IbvSendWr, imm_data) == 36);
    assert!(offset_of!(IbvSendWr, remote_addr) == 40);
    assert!(offset_of!(IbvSendWr, rkey) == 48);
    assert!(size_of::<IbvRecvWr>() == 32);
    assert!(size_of::<IbvWc>() == 48);
    assert!(offset_of!(IbvWc, byte_len) == 20);
    assert!(offset_of!(IbvWc, wc_flags) == 36);
    assert!(offset_of!(IbvQp, qp_num) == 52);
    assert!(offset_of!(IbvMr, lkey) == 36);
    assert!(offset_of!(IbvMr, rkey) == 40);
    assert!(size_of::<IbvQpInitAttr>() == 64);
    assert!(offset_of!(IbvQpInitAttr, qp_type) == 52);
    assert!(size_of::<IbvQpAttr>() == 144);
    assert!(offset_of!(IbvQpAttr, ah_attr) == 56);
    assert!(offset_of!(IbvQpAttr, pkey_index) == 120);
    assert!(offset_of!(IbvQpAttr, max_rd_atomic) == 126);
    assert!(offset_of!(IbvQpAttr, rnr_retry) == 132);
    assert!(offset_of!(IbvQpAttr, rate_limit) == 136);
    assert!(size_of::<IbvAhAttr>() == 32);
    assert!(offset_of!(IbvAhAttr, is_global) == 29);
    assert!(offset_of!(IbvGlobalRoute, sgid_index) == 20);
    assert!(size_of::<IbvPortAttr>() == 56);
    assert!(offset_of!(IbvContext, ops) + OP_POLL_CQ * 8 == 96);
    assert!(offset_of!(IbvContext, ops) + OP_POST_SEND * 8 == 208);
    assert!(offset_of!(IbvContext, ops) + OP_POST_RECV * 8 == 216);
};

type PollCq = unsafe extern "C" fn(*mut IbvCq, c_int, *mut IbvWc) -> c_int;
type PostSend = unsafe extern "C" fn(*mut IbvQp, *mut IbvSendWr, *mut *mut IbvSendWr) -> c_int;
type PostRecv = unsafe extern "C" fn(*mut IbvQp, *mut IbvRecvWr, *mut *mut IbvRecvWr) -> c_int;

/// The libibverbs entry points, from `dlopen`.
struct Verbs {
    get_device_list: unsafe extern "C" fn(*mut c_int) -> *mut *mut c_void,
    free_device_list: unsafe extern "C" fn(*mut *mut c_void),
    get_device_name: unsafe extern "C" fn(*mut c_void) -> *const c_char,
    open_device: unsafe extern "C" fn(*mut c_void) -> *mut IbvContext,
    close_device: unsafe extern "C" fn(*mut IbvContext) -> c_int,
    alloc_pd: unsafe extern "C" fn(*mut IbvContext) -> *mut c_void,
    dealloc_pd: unsafe extern "C" fn(*mut c_void) -> c_int,
    reg_mr: unsafe extern "C" fn(*mut c_void, *mut c_void, usize, c_int) -> *mut IbvMr,
    dereg_mr: unsafe extern "C" fn(*mut IbvMr) -> c_int,
    create_cq:
        unsafe extern "C" fn(*mut IbvContext, c_int, *mut c_void, *mut c_void, c_int) -> *mut IbvCq,
    destroy_cq: unsafe extern "C" fn(*mut IbvCq) -> c_int,
    create_qp: unsafe extern "C" fn(*mut c_void, *mut IbvQpInitAttr) -> *mut IbvQp,
    destroy_qp: unsafe extern "C" fn(*mut IbvQp) -> c_int,
    modify_qp: unsafe extern "C" fn(*mut IbvQp, *mut IbvQpAttr, c_int) -> c_int,
    query_port: unsafe extern "C" fn(*mut IbvContext, u8, *mut IbvPortAttr) -> c_int,
}

/// libibverbs, loaded once (an error if the box has none).
fn verbs() -> Result<&'static Verbs, String> {
    static V: OnceLock<Result<Verbs, String>> = OnceLock::new();
    V.get_or_init(load).as_ref().map_err(Clone::clone)
}

// Each symbol is transmuted to the type of the field it fills (`sym!`).
#[allow(clippy::missing_transmute_annotations)]
fn load() -> Result<Verbs, String> {
    // SAFETY: dlopen/dlsym with NUL-terminated names; each symbol is cast to its C signature
    // in verbs.h. The library stays loaded for the process.
    unsafe {
        let lib = libc::dlopen(c"libibverbs.so.1".as_ptr(), libc::RTLD_NOW);
        if lib.is_null() {
            return Err("libibverbs.so.1 is not installed".into());
        }
        macro_rules! sym {
            ($name:literal) => {{
                let p = libc::dlsym(lib, concat!($name, "\0").as_ptr() as *const c_char);
                if p.is_null() {
                    return Err(format!("libibverbs has no {}", $name));
                }
                std::mem::transmute::<*mut c_void, _>(p)
            }};
        }
        Ok(Verbs {
            get_device_list: sym!("ibv_get_device_list"),
            free_device_list: sym!("ibv_free_device_list"),
            get_device_name: sym!("ibv_get_device_name"),
            open_device: sym!("ibv_open_device"),
            close_device: sym!("ibv_close_device"),
            alloc_pd: sym!("ibv_alloc_pd"),
            dealloc_pd: sym!("ibv_dealloc_pd"),
            reg_mr: sym!("ibv_reg_mr"),
            dereg_mr: sym!("ibv_dereg_mr"),
            create_cq: sym!("ibv_create_cq"),
            destroy_cq: sym!("ibv_destroy_cq"),
            create_qp: sym!("ibv_create_qp"),
            destroy_qp: sym!("ibv_destroy_qp"),
            modify_qp: sym!("ibv_modify_qp"),
            query_port: sym!("ibv_query_port"),
        })
    }
}

fn os_err(what: &str) -> String {
    format!("{what}: {}", std::io::Error::last_os_error())
}

/// The GID of a local address, as the kernel lists it: an IPv4 address mapped into IPv6.
pub(super) fn gid_of(ip: IpAddr) -> Gid {
    Gid(match ip {
        IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
        IpAddr::V6(v6) => v6.octets(),
    })
}

/// A GID as sysfs prints it, `0000:0000:0000:0000:0000:ffff:c000:0201`.
pub(super) fn parse_gid(s: &str) -> Option<Gid> {
    let mut g = [0u8; 16];
    let mut n = 0;
    for part in s.trim().split(':') {
        if n >= 8 || part.len() != 4 {
            return None;
        }
        let v = u16::from_str_radix(part, 16).ok()?;
        g[2 * n..2 * n + 2].copy_from_slice(&v.to_be_bytes());
        n += 1;
    }
    (n == 8).then_some(Gid(g))
}

/// Where a local address is on an RDMA device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct GidPlace {
    pub dev: String,
    pub port: u8,
    pub index: u8,
    pub gid: Gid,
}

/// The RoCE v2 GID of local address `ip` under `root` (`/sys/class/infiniband`): the device,
/// port and GID index RDMA to a peer reached from `ip` goes through. `None` if no device has
/// it (no RDMA NIC, or not RoCE v2).
pub(super) fn find_gid(root: &Path, ip: IpAddr) -> Option<GidPlace> {
    let want = gid_of(ip);
    let mut devs: Vec<_> = std::fs::read_dir(root).ok()?.flatten().collect();
    devs.sort_by_key(|d| d.file_name());
    for dev in devs {
        let ports = dev.path().join("ports");
        let Ok(rd) = std::fs::read_dir(&ports) else {
            continue;
        };
        let mut ports: Vec<_> = rd.flatten().collect();
        ports.sort_by_key(|p| p.file_name());
        for port in ports {
            let Ok(port_num) = port.file_name().to_string_lossy().parse::<u8>() else {
                continue;
            };
            let Ok(rd) = std::fs::read_dir(port.path().join("gids")) else {
                continue;
            };
            let mut idx: Vec<u8> = rd
                .flatten()
                .filter_map(|e| e.file_name().to_string_lossy().parse().ok())
                .collect();
            idx.sort_unstable();
            for i in idx {
                let read =
                    |p: &str| std::fs::read_to_string(port.path().join(p).join(i.to_string()));
                let Some(gid) = read("gids").ok().as_deref().and_then(parse_gid) else {
                    continue;
                };
                let v2 = read("gid_attrs/types").is_ok_and(|t| t.trim() == "RoCE v2");
                if gid == want && v2 {
                    return Some(GidPlace {
                        dev: dev.file_name().to_string_lossy().into_owned(),
                        port: port_num,
                        index: i,
                        gid,
                    });
                }
            }
        }
    }
    None
}

/// An opened device and its protection domain.
pub(super) struct Device {
    v: &'static Verbs,
    ctx: *mut IbvContext,
    pd: *mut c_void,
    place: GidPlace,
    /// The port's active MTU (`IBV_MTU_*`).
    mtu: u32,
}

// SAFETY: libibverbs objects may be used from any thread; each queue pair is used by one
// writer and one reader thread, each on its own queue (see `Endpoint`).
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    fn open(place: GidPlace) -> Result<Arc<Self>, String> {
        let v = verbs()?;
        // SAFETY: libibverbs calls with valid arguments; the list is freed after the open.
        unsafe {
            let mut n = 0;
            let list = (v.get_device_list)(&mut n);
            if list.is_null() {
                return Err(os_err("ibv_get_device_list"));
            }
            let devs = std::slice::from_raw_parts(list, n.max(0) as usize);
            let dev = devs.iter().copied().find(|&d| {
                let name = (v.get_device_name)(d);
                !name.is_null() && CStr::from_ptr(name).to_string_lossy() == place.dev
            });
            let ctx = dev.map_or(std::ptr::null_mut(), |d| (v.open_device)(d));
            (v.free_device_list)(list);
            if ctx.is_null() {
                return Err(format!("cannot open RDMA device {}", place.dev));
            }
            let pd = (v.alloc_pd)(ctx);
            if pd.is_null() {
                let e = os_err("ibv_alloc_pd");
                (v.close_device)(ctx);
                return Err(e);
            }
            let mut attr = IbvPortAttr::default();
            let mtu = match (v.query_port)(ctx, place.port, &mut attr) {
                0 => attr.active_mtu.clamp(1, IBV_MTU_MAX),
                _ => 3,
            };
            Ok(Arc::new(Self {
                v,
                ctx,
                pd,
                place,
                mtu,
            }))
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: the domain and context opened in `open`, freed once, after every queue pair
        // and memory region (which hold this device).
        unsafe {
            (self.v.dealloc_pd)(self.pd);
            (self.v.close_device)(self.ctx);
        }
    }
}

/// A registered memory region.
pub(crate) struct Mr {
    dev: Arc<Device>,
    mr: *mut IbvMr,
}

// SAFETY: as for `Device`.
unsafe impl Send for Mr {}
unsafe impl Sync for Mr {}

impl Mr {
    pub(super) fn lkey(&self) -> u32 {
        // SAFETY: a live region from `ibv_reg_mr`.
        unsafe { (*self.mr).lkey }
    }

    pub(super) fn rkey(&self) -> u32 {
        // SAFETY: as above.
        unsafe { (*self.mr).rkey }
    }
}

impl Drop for Mr {
    fn drop(&mut self) {
        // SAFETY: registered in `Endpoint::register`, deregistered once.
        unsafe {
            (self.dev.v.dereg_mr)(self.mr);
        }
    }
}

/// What each side of a queue pair tells the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Offer {
    /// False: this side has no RDMA to this peer (the pair uses TCP).
    pub rdma: bool,
    pub qpn: u32,
    pub psn: u32,
    pub mtu: u32,
    pub gid: Gid,
}

const OFFER_MAGIC: u32 = u32::from_le_bytes(*b"LgRd");
const REGION_MAGIC: u32 = u32::from_le_bytes(*b"LgRm");

impl Offer {
    pub(super) fn encode(&self) -> [u8; 36] {
        let mut b = [0u8; 36];
        b[0..4].copy_from_slice(&OFFER_MAGIC.to_le_bytes());
        b[4..8].copy_from_slice(&(self.rdma as u32).to_le_bytes());
        b[8..12].copy_from_slice(&self.qpn.to_le_bytes());
        b[12..16].copy_from_slice(&self.psn.to_le_bytes());
        b[16..20].copy_from_slice(&self.mtu.to_le_bytes());
        b[20..36].copy_from_slice(&self.gid.0);
        b
    }

    pub(super) fn decode(b: &[u8; 36]) -> Result<Self, String> {
        let u = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes"));
        if u(0) != OFFER_MAGIC {
            return Err("a peer did not answer the transport offer: run the same legion".into());
        }
        Ok(Self {
            rdma: u(4) != 0,
            qpn: u(8),
            psn: u(12),
            mtu: u(16),
            gid: Gid(b[20..36].try_into().expect("16 bytes")),
        })
    }
}

/// Sends this side's offer on `s` and reads the peer's (both send first, so neither waits).
pub(super) fn swap_offers(s: &mut TcpStream, mine: &Offer) -> Result<Offer, String> {
    let io = |e: std::io::Error| format!("transport offer: {e}");
    s.write_all(&mine.encode()).map_err(io)?;
    let mut b = [0u8; 36];
    s.read_exact(&mut b).map_err(io)?;
    Offer::decode(&b)
}

/// A memory region's address and key, as the peer writes into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RemoteRegion {
    pub addr: u64,
    pub rkey: u32,
    pub len: u64,
}

impl RemoteRegion {
    pub(super) fn encode(&self, seq: u32) -> [u8; 28] {
        let mut b = [0u8; 28];
        b[0..4].copy_from_slice(&REGION_MAGIC.to_le_bytes());
        b[4..8].copy_from_slice(&seq.to_le_bytes());
        b[8..16].copy_from_slice(&self.addr.to_le_bytes());
        b[16..20].copy_from_slice(&self.rkey.to_le_bytes());
        b[20..28].copy_from_slice(&self.len.to_le_bytes());
        b
    }

    /// The region of a peer's message for all-reduce `seq`.
    pub(super) fn decode(b: &[u8; 28], seq: u32) -> Result<Self, String> {
        let u = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().expect("4 bytes"));
        let q = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"));
        if u(0) != REGION_MAGIC || u(4) != seq {
            return Err(format!(
                "a peer's memory region is not for all-reduce {seq}: the ranks are out of step"
            ));
        }
        Ok(Self {
            addr: q(8),
            rkey: u(16),
            len: q(20),
        })
    }
}

/// One queue pair to a peer, with a completion queue for its sends and one for its receives.
pub(crate) struct Endpoint {
    dev: Arc<Device>,
    qp: *mut IbvQp,
    send_cq: *mut IbvCq,
    recv_cq: *mut IbvCq,
    psn: u32,
}

// SAFETY: as for `Device`: the writer thread alone posts sends and polls the send queue, the
// reader alone polls the receive queue and posts receives.
unsafe impl Send for Endpoint {}
unsafe impl Sync for Endpoint {}

impl Endpoint {
    /// A queue pair on the device that holds `local` (the TCP connection's local address), in
    /// the INIT state with its receives posted; `None` if no device does.
    pub(super) fn open(local: IpAddr, rank: usize) -> Result<Option<Self>, String> {
        let Some(place) = find_gid(Path::new("/sys/class/infiniband"), local) else {
            return Ok(None);
        };
        let dev = Device::open(place)?;
        let v = dev.v;
        // SAFETY: libibverbs calls on the live device; everything made here is freed in Drop
        // (or below on failure).
        unsafe {
            let send_cq = (v.create_cq)(
                dev.ctx,
                SEND_DEPTH as c_int,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            let recv_cq = (v.create_cq)(
                dev.ctx,
                RECV_DEPTH as c_int,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            );
            let mut init = IbvQpInitAttr {
                qp_context: std::ptr::null_mut(),
                send_cq,
                recv_cq,
                srq: std::ptr::null_mut(),
                cap: IbvQpCap {
                    max_send_wr: SEND_DEPTH,
                    max_recv_wr: RECV_DEPTH,
                    max_send_sge: 1,
                    max_recv_sge: 1,
                    max_inline_data: 0,
                },
                qp_type: IBV_QPT_RC,
                sq_sig_all: 0,
            };
            let qp = if send_cq.is_null() || recv_cq.is_null() {
                std::ptr::null_mut()
            } else {
                (v.create_qp)(dev.pd, &mut init)
            };
            // A start PSN that differs per rank and process.
            let psn = (std::process::id() ^ (rank as u32).wrapping_mul(0x9e37_79b9)) & 0xff_ffff;
            let ep = Self {
                dev,
                qp,
                send_cq,
                recv_cq,
                psn,
            };
            if qp.is_null() {
                return Err(os_err("creating an RDMA queue pair"));
            }
            let mut a = IbvQpAttr {
                qp_state: IBV_QPS_INIT,
                port_num: ep.dev.place.port,
                qp_access_flags: (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) as c_uint,
                ..Default::default()
            };
            ep.modify(
                &mut a,
                IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS,
                "INIT",
            )?;
            for _ in 0..RECV_DEPTH {
                ep.post_recv()?;
            }
            Ok(Some(ep))
        }
    }

    fn modify(&self, a: &mut IbvQpAttr, mask: c_int, what: &str) -> Result<(), String> {
        // SAFETY: the live queue pair and an attribute struct of verbs.h's layout.
        match unsafe { (self.dev.v.modify_qp)(self.qp, a, mask) } {
            0 => Ok(()),
            e => Err(format!(
                "RDMA queue pair to {what}: {}",
                std::io::Error::from_raw_os_error(e)
            )),
        }
    }

    /// This side's offer.
    pub(super) fn offer(&self) -> Offer {
        Offer {
            rdma: true,
            // SAFETY: the live queue pair.
            qpn: unsafe { (*self.qp).qp_num },
            psn: self.psn,
            mtu: self.dev.mtu,
            gid: self.dev.place.gid,
        }
    }

    /// Connects the queue pair to the peer's `far` (INIT to RTR to RTS).
    pub(super) fn connect(&self, far: &Offer) -> Result<(), String> {
        let mut a = IbvQpAttr {
            qp_state: IBV_QPS_RTR,
            path_mtu: self.dev.mtu.min(far.mtu).clamp(1, IBV_MTU_MAX),
            dest_qp_num: far.qpn,
            rq_psn: far.psn,
            max_dest_rd_atomic: 1,
            min_rnr_timer: 12,
            ah_attr: IbvAhAttr {
                grh: IbvGlobalRoute {
                    dgid: far.gid,
                    sgid_index: self.dev.place.index,
                    hop_limit: 64,
                    ..Default::default()
                },
                is_global: 1,
                port_num: self.dev.place.port,
                ..Default::default()
            },
            ..Default::default()
        };
        self.modify(
            &mut a,
            IBV_QP_STATE
                | IBV_QP_AV
                | IBV_QP_PATH_MTU
                | IBV_QP_DEST_QPN
                | IBV_QP_RQ_PSN
                | IBV_QP_MAX_DEST_RD_ATOMIC
                | IBV_QP_MIN_RNR_TIMER,
            "RTR",
        )?;
        let mut a = IbvQpAttr {
            qp_state: IBV_QPS_RTS,
            timeout: 14,
            retry_cnt: 7,
            rnr_retry: 7,
            sq_psn: self.psn,
            max_rd_atomic: 1,
            ..Default::default()
        };
        self.modify(
            &mut a,
            IBV_QP_STATE
                | IBV_QP_TIMEOUT
                | IBV_QP_RETRY_CNT
                | IBV_QP_RNR_RETRY
                | IBV_QP_SQ_PSN
                | IBV_QP_MAX_QP_RD_ATOMIC,
            "RTS",
        )
    }

    /// The device this endpoint's queue pair is on.
    pub(super) fn device_name(&self) -> &str {
        &self.dev.place.dev
    }

    /// Registers `len` bytes at host address `addr` for local reads and remote writes.
    pub(super) fn register(&self, addr: usize, len: usize) -> Result<Mr, String> {
        let access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        // SAFETY: the caller's memory, which outlives the region (`Mr` is dropped first).
        let mr = unsafe { (self.dev.v.reg_mr)(self.dev.pd, addr as *mut c_void, len, access) };
        if mr.is_null() {
            return Err(os_err("registering RDMA memory"));
        }
        Ok(Mr {
            dev: self.dev.clone(),
            mr,
        })
    }

    fn post_recv(&self) -> Result<(), String> {
        let mut wr = IbvRecvWr {
            wr_id: 0,
            next: std::ptr::null_mut(),
            sg_list: std::ptr::null_mut(),
            num_sge: 0,
        };
        let mut bad = std::ptr::null_mut();
        // SAFETY: the context's post_recv op on its own queue pair; a receive without
        // buffers (a write with immediate carries its data to the address it names).
        let r = unsafe {
            let f: PostRecv = std::mem::transmute((*self.dev.ctx).ops[OP_POST_RECV]);
            f(self.qp, &mut wr, &mut bad)
        };
        match r {
            0 => Ok(()),
            e => Err(format!(
                "posting an RDMA receive: {}",
                std::io::Error::from_raw_os_error(e)
            )),
        }
    }

    /// Posts an RDMA write with immediate `imm` of `len` bytes at `addr` (in `lkey`'s region)
    /// to the peer's `raddr` (in `rkey`'s), signaled.
    pub(super) fn post_write(
        &self,
        (addr, len, lkey): (usize, usize, u32),
        (raddr, rkey): (u64, u32),
        imm: u32,
    ) -> Result<(), String> {
        let mut sge = IbvSge {
            addr: addr as u64,
            length: u32::try_from(len).map_err(|_| "an RDMA write over 4 GiB".to_string())?,
            lkey,
        };
        let mut wr = IbvSendWr {
            wr_id: imm as u64,
            next: std::ptr::null_mut(),
            sg_list: &mut sge,
            num_sge: 1,
            opcode: IBV_WR_RDMA_WRITE_WITH_IMM,
            send_flags: IBV_SEND_SIGNALED,
            imm_data: imm.to_be(),
            remote_addr: raddr,
            rkey,
            _rest: [0; 76],
        };
        let mut bad = std::ptr::null_mut();
        // SAFETY: the context's post_send op on its own queue pair; the request and its
        // scatter entry live through the call (the provider copies them).
        let r = unsafe {
            let f: PostSend = std::mem::transmute((*self.dev.ctx).ops[OP_POST_SEND]);
            f(self.qp, &mut wr, &mut bad)
        };
        match r {
            0 => Ok(()),
            e => Err(format!(
                "posting an RDMA write: {}",
                std::io::Error::from_raw_os_error(e)
            )),
        }
    }

    fn poll(&self, cq: *mut IbvCq) -> Result<Option<IbvWc>, String> {
        let mut wc = IbvWc::default();
        // SAFETY: the context's poll_cq op on one of this endpoint's queues.
        let n = unsafe {
            let f: PollCq = std::mem::transmute((*self.dev.ctx).ops[OP_POLL_CQ]);
            f(cq, 1, &mut wc)
        };
        match n {
            0 => Ok(None),
            1 if wc.status == IBV_WC_SUCCESS => Ok(Some(wc)),
            1 => Err(format!(
                "an RDMA completion failed (status {}, vendor error {:#x})",
                wc.status, wc.vendor_err
            )),
            _ => Err("polling an RDMA completion queue failed".into()),
        }
    }

    /// The next send completion, if there is one: the immediate of the write it completes.
    pub(super) fn poll_send(&self) -> Result<Option<u32>, String> {
        Ok(self.poll(self.send_cq)?.map(|wc| wc.wr_id as u32))
    }

    /// The next received write, if there is one: its immediate and length; the receive it
    /// took is posted again.
    pub(super) fn poll_recv(&self) -> Result<Option<(u32, usize)>, String> {
        let Some(wc) = self.poll(self.recv_cq)? else {
            return Ok(None);
        };
        if wc.opcode != IBV_WC_RECV_RDMA_WITH_IMM || wc.wc_flags & IBV_WC_WITH_IMM == 0 {
            return Err(format!(
                "an RDMA receive completed as opcode {}, not a write with immediate",
                wc.opcode
            ));
        }
        self.post_recv()?;
        Ok(Some((u32::from_be(wc.imm_data), wc.byte_len as usize)))
    }
}

impl Drop for Endpoint {
    fn drop(&mut self) {
        // SAFETY: made in `open`, destroyed once (the queue pair before its queues).
        unsafe {
            if !self.qp.is_null() {
                (self.dev.v.destroy_qp)(self.qp);
            }
            if !self.send_cq.is_null() {
                (self.dev.v.destroy_cq)(self.send_cq);
            }
            if !self.recv_cq.is_null() {
                (self.dev.v.destroy_cq)(self.recv_cq);
            }
        }
    }
}
