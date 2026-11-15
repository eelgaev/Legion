//! The all-reduce over loopback connections (two and three ranks, threads for machines), its
//! graph op on the CPU, the connection setup, failures, and the shards of a dense tiny Gemma 4.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::*;
use crate::ggml::{Backend, ComputeAllocator, Context, types};
use crate::gguf::{GgufFile, Recipe};

/// Fully connected loopback streams: `out[r][p]` is rank `r`'s end of its connection to `p`.
fn loopback(n: usize) -> Vec<Vec<Option<TcpStream>>> {
    let mut out: Vec<Vec<Option<TcpStream>>> =
        (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
        .collect();
    for (a, b) in pairs {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let s = TcpStream::connect(l.local_addr().unwrap()).unwrap();
        let (t, _) = l.accept().unwrap();
        s.set_nodelay(true).unwrap();
        t.set_nodelay(true).unwrap();
        out[a][b] = Some(s);
        out[b][a] = Some(t);
    }
    out
}

/// The ranks' communicators, made at once (each swaps transport offers with the others).
fn comms_over(
    n: usize,
    timeout: Duration,
    transport: impl Fn(usize) -> Transport,
) -> Vec<Arc<Comm>> {
    let made: Vec<_> = loopback(n)
        .into_iter()
        .enumerate()
        .map(|(r, data)| {
            let t = transport(r);
            std::thread::spawn(move || Comm::with_transport(r, data, timeout, t))
        })
        .collect();
    made.into_iter()
        .map(|t| t.join().unwrap().unwrap())
        .collect()
}

pub(super) fn comms(n: usize, timeout: Duration) -> Vec<Arc<Comm>> {
    comms_over(n, timeout, |_| Transport::Tcp)
}

/// Rank `r`'s partial of all-reduce `k`: values whose sum depends on the order of the adds.
fn partial(r: usize, k: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i * 7 + k * 13 + r * 31) % 97) as f32;
            if r.is_multiple_of(2) {
                x * 1e7
            } else {
                x * 0.3 + 1e-3
            }
        })
        .collect()
}

/// The sum every rank must get: in rank order, one rounding at a time.
fn rank_order_sum(n: usize, k: usize, len: usize) -> Vec<f32> {
    let parts: Vec<Vec<f32>> = (0..n).map(|r| partial(r, k, len)).collect();
    (0..len)
        .map(|i| {
            let mut acc = parts[0][i];
            for p in &parts[1..] {
                acc += p[i];
            }
            acc
        })
        .collect()
}

/// Lengths from one float to several 64 KiB frames.
const LENS: [usize; 5] = [1, 5, 4096, 40_000, 3];

fn all_reduces_match(n: usize) {
    let cs = comms(n, Duration::from_secs(20));
    let threads: Vec<_> = cs
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let mut got = Vec::new();
                for (k, &len) in LENS.iter().enumerate() {
                    let src = partial(r, k, len);
                    let mut dst = vec![0f32; len];
                    c.all_reduce_host(&src, &mut dst).unwrap();
                    got.push(dst);
                }
                assert_eq!(c.all_reduces(), LENS.len() as u32);
                got
            })
        })
        .collect();
    let results: Vec<Vec<Vec<f32>>> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for (k, &len) in LENS.iter().enumerate() {
        let want = rank_order_sum(n, k, len);
        for (r, res) in results.iter().enumerate() {
            let same = res[k]
                .iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{n} ranks, all-reduce {k}: rank {r} differs");
        }
    }
}

#[test]
fn two_ranks_sum_bit_identically() {
    all_reduces_match(2);
}

#[test]
fn three_ranks_sum_bit_identically() {
    all_reduces_match(3);
}

#[test]
fn a_missing_rank_fails_the_others_in_time() {
    let mut cs = comms(3, Duration::from_millis(500));
    let gone = cs.pop().unwrap();
    gone.shutdown();
    drop(gone);
    for c in cs {
        let src = [1.0f32, 2.0];
        let mut dst = [0f32; 2];
        let t = std::time::Instant::now();
        let e = c.all_reduce_host(&src, &mut dst).unwrap_err();
        assert!(t.elapsed() < Duration::from_secs(10), "{e}");
        assert_eq!(dst, src, "a failed all-reduce leaves the partial");
        assert!(c.error().is_some());
        // Final: the next one is refused at once.
        assert!(c.all_reduce_host(&src, &mut dst).is_err());
    }
}

#[test]
fn ranks_out_of_step_fail() {
    let cs = comms(2, Duration::from_secs(5));
    let (a, b) = (cs[0].clone(), cs[1].clone());
    let t = std::thread::spawn(move || {
        let mut dst = [0f32; 4];
        b.all_reduce_host(&[1.0; 4], &mut dst).unwrap();
        // Rank 1 runs a different (longer) all-reduce next.
        let mut dst = [0f32; 8];
        b.all_reduce_host(&[1.0; 8], &mut dst)
    });
    let mut dst = [0f32; 4];
    a.all_reduce_host(&[2.0; 4], &mut dst).unwrap();
    assert_eq!(dst, [3.0; 4]);
    assert!(a.all_reduce_host(&[1.0; 4], &mut dst).is_err());
    assert!(t.join().unwrap().is_err());
}

/// Caps a socket's kernel buffer (`SO_SNDBUF` or `SO_RCVBUF`) at `bytes`, so a big frame stays
/// in the writer until the peer reads it.
fn small_buffer(s: &TcpStream, opt: libc::c_int, bytes: libc::c_int) {
    // SAFETY: an int option of an open socket.
    let r = unsafe {
        libc::setsockopt(
            s.as_raw_fd(),
            libc::SOL_SOCKET,
            opt,
            (&bytes as *const libc::c_int).cast(),
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    };
    assert_eq!(r, 0, "setsockopt");
}

/// Rank 0's communicator over a loopback connection with small buffers, and the other end,
/// past the transport offers: the test plays rank 1 by hand.
fn comm_and_raw_peer(timeout: Duration) -> (Arc<Comm>, TcpStream) {
    let mut ends = loopback(2);
    let mut raw = ends[1][0].take().unwrap();
    small_buffer(ends[0][1].as_ref().unwrap(), libc::SO_SNDBUF, 64 << 10);
    small_buffer(&raw, libc::SO_RCVBUF, 64 << 10);
    let data = std::mem::take(&mut ends[0]);
    let made = std::thread::spawn(move || Comm::new(0, data, timeout));
    super::rdma::swap_offers(&mut raw, &Default::default()).unwrap();
    (made.join().unwrap().unwrap(), raw)
}

/// Rank 1's all-reduce 1 of `src` on `raw`, written from another thread (it is bigger than the
/// buffers: rank 0 reads it in its all-reduce).
fn send_frame(raw: &TcpStream, src: &[f32]) -> std::thread::JoinHandle<()> {
    let mut frame = Vec::with_capacity(12 + 4 * src.len());
    frame.extend_from_slice(&1u32.to_le_bytes());
    frame.extend_from_slice(&(4 * src.len() as u64).to_le_bytes());
    frame.extend(src.iter().flat_map(|x| x.to_le_bytes()));
    let mut s = raw.try_clone().unwrap();
    std::thread::spawn(move || s.write_all(&frame).unwrap())
}

/// Floats in the frames of the shutdown tests: far more than the connection's buffers hold.
const BIG: usize = 1 << 20;

/// #673: an all-reduce ends once the peers' partials arrived, maybe before this rank's own
/// went out; shutting down right then must still send it whole (the connections used to close
/// under the writer: the peer read "failed to fill whole buffer").
#[test]
fn shutdown_sends_the_last_partial_first() {
    let (c, mut raw) = comm_and_raw_peer(Duration::from_secs(20));
    let sent = send_frame(&raw, &partial(1, 0, BIG));
    let src = partial(0, 0, BIG);
    let mut dst = vec![0f32; BIG];
    c.all_reduce_host(&src, &mut dst).unwrap();
    sent.join().unwrap();
    assert_eq!(dst, rank_order_sum(2, 0, BIG));
    // Rank 1 reads rank 0's partial only once rank 0 is shutting down.
    let read = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        let mut h = [0u8; 12];
        raw.read_exact(&mut h).map_err(|e| e.to_string())?;
        let mut b = vec![0u8; 4 * BIG];
        raw.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok::<_, String>((h, b))
    });
    c.shutdown();
    let (h, b) = read.join().unwrap().expect("rank 0's partial, whole");
    assert_eq!(h[..4], 1u32.to_le_bytes());
    assert_eq!(h[4..], (4 * BIG as u64).to_le_bytes());
    let want: Vec<u8> = src.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert!(b == want, "rank 0's partial differs");
    assert!(c.error().is_none(), "{:?}", c.error());
}

/// A peer that never reads holds a shutdown up for a bounded time only.
#[test]
fn shutdown_gives_up_on_a_peer_that_does_not_read() {
    let (c, raw) = comm_and_raw_peer(Duration::from_secs(2));
    let sent = send_frame(&raw, &partial(1, 0, BIG));
    let mut dst = vec![0f32; BIG];
    c.all_reduce_host(&partial(0, 0, BIG), &mut dst).unwrap();
    sent.join().unwrap();
    let t = Instant::now();
    c.shutdown();
    assert!(t.elapsed() < Duration::from_secs(10), "{:?}", t.elapsed());
    drop(raw);
}

#[test]
fn auto_and_tcp_ranks_agree_on_tcp_without_rdma() {
    // Loopback has no RDMA device: `Auto` takes TCP, and meets a `Tcp` rank's offer.
    let cs = comms_over(3, Duration::from_secs(20), |r| {
        if r == 1 {
            Transport::Tcp
        } else {
            Transport::Auto
        }
    });
    for c in &cs {
        assert!(!c.has_rdma());
        assert!(!c.transports().contains("rdma"), "{}", c.transports());
    }
    let threads: Vec<_> = cs
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let mut dst = [0f32; 2];
                c.all_reduce_host(&[r as f32 + 1.0; 2], &mut dst).unwrap();
                dst
            })
        })
        .collect();
    for t in threads {
        assert_eq!(t.join().unwrap(), [6.0; 2]);
    }
}

#[test]
fn rdma_forced_without_a_device_fails_to_start() {
    let made: Vec<_> = loopback(2)
        .into_iter()
        .enumerate()
        .map(|(r, data)| {
            std::thread::spawn(move || {
                Comm::with_transport(r, data, Duration::from_secs(5), Transport::Rdma).err()
            })
        })
        .collect();
    for t in made {
        let e = t.join().unwrap().expect("refused").to_string();
        assert!(e.contains("no RDMA device"), "{e}");
    }
}

#[test]
fn stats_count_the_all_reduces_and_the_steps_first() {
    let cs = comms(2, Duration::from_secs(20));
    let threads: Vec<_> = cs
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let mut dst = [0f32; 16];
                for step in [3, 2] {
                    c.mark_step();
                    for _ in 0..step {
                        c.all_reduce_host(&[r as f32; 16], &mut dst).unwrap();
                    }
                }
                c
            })
        })
        .collect();
    let cs: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for c in &cs {
        let s = c.stats();
        assert_eq!(s.count, 5, "{s:?}");
        assert_eq!(s.firsts, 2, "{s:?}");
        assert!(s.stall_ns as i64 >= s.wait_ns, "{s:?}");
        assert!(s.first_stall_ns <= s.stall_ns, "{s:?}");
    }
}

#[test]
fn step_marks_fall_before_the_first_and_after_the_last_all_reduce() {
    let cs = comms(2, Duration::from_secs(20));
    let threads: Vec<_> = cs
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let mut dst = [0f32; 16];
                assert_eq!(c.step_marks().before_first, None);
                c.mark_step();
                std::thread::sleep(Duration::from_millis(20));
                for _ in 0..3 {
                    c.all_reduce_host(&[r as f32; 16], &mut dst).unwrap();
                }
                std::thread::sleep(Duration::from_millis(20));
                c.step_marks()
            })
        })
        .collect();
    for t in threads {
        let m = t.join().unwrap();
        let (a, b) = (
            m.before_first.expect("a first"),
            m.after_last.expect("a last"),
        );
        assert!(a >= Duration::from_millis(20), "{m:?}");
        assert!(b >= Duration::from_millis(20), "{m:?}");
        assert!(m.tx_cpu.is_some() && m.rx_cpu.is_some(), "{m:?}");
    }
}

/// Free loopback ports (bound and released: another process could take one meanwhile, which
/// these tests accept).
fn free_ports(n: usize) -> Vec<String> {
    let ls: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    ls.iter()
        .map(|l| l.local_addr().unwrap().to_string())
        .collect()
}

#[test]
fn connect_makes_a_mesh_and_control_connections() {
    let peers = free_ports(3);
    let cfgs: Vec<TpNetConfig> = (0..3)
        .map(|rank| TpNetConfig {
            rank,
            peers: peers.clone(),
            timeout: Duration::from_secs(10),
            ..Default::default()
        })
        .collect();
    // Started in reverse, as machines come up in any order.
    let threads: Vec<_> = cfgs
        .iter()
        .rev()
        .cloned()
        .map(|cfg| {
            std::thread::spawn(move || {
                let mut mesh = connect(&cfg, Duration::from_secs(20)).unwrap();
                let comm =
                    Comm::new(cfg.rank, std::mem::take(&mut mesh.data), cfg.timeout).unwrap();
                let mut dst = [0f32; 3];
                comm.all_reduce_host(&[cfg.rank as f32 + 1.0; 3], &mut dst)
                    .unwrap();
                assert_eq!(dst, [6.0; 3]);
                // Rank 0 dials each other rank's control listener, which answers.
                if cfg.rank == 0 {
                    for to in 1..3 {
                        dial_control(&cfg, to, Duration::from_secs(20)).unwrap();
                    }
                } else {
                    mesh.control.as_mut().unwrap().accept().unwrap();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
}

#[test]
fn a_rank_listed_wrong_is_refused() {
    let peers = free_ports(2);
    let mut swapped = peers.clone();
    swapped.swap(0, 1);
    let one = TpNetConfig {
        rank: 1,
        peers: peers.clone(),
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    let t = std::thread::spawn(move || connect(&one, Duration::from_secs(5)));
    // Rank 0 thinks it is talking to 3 ranks: the preamble's count differs.
    let zero = TpNetConfig {
        rank: 0,
        peers: vec![peers[0].clone(), peers[1].clone(), "127.0.0.1:1".into()],
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    assert!(connect(&zero, Duration::from_secs(5)).is_err());
    drop(t);
    assert!(
        TpNetConfig {
            rank: 2,
            peers: swapped,
            timeout: Duration::from_secs(1),
            ..Default::default()
        }
        .check()
        .is_err()
    );
}

/// The all-reduce op through a CPU graph on every rank: `x * 2` summed over the ranks.
#[test]
fn the_graph_op_sums_on_the_cpu() {
    let cs = comms(2, Duration::from_secs(20));
    let threads: Vec<_> = cs
        .into_iter()
        .enumerate()
        .map(|(r, c)| {
            std::thread::spawn(move || {
                let handle = register(c);
                let mut backend = Backend::cpu().unwrap();
                backend.set_threads(3);
                let ctx = Context::new(16, 16);
                let x = ctx.new_tensor(types::F32, &[6, 5]).as_input();
                let y = all_reduce(&ctx, handle.id(), ctx.scale(x, 2.0)).as_output();
                let mut graph = ctx.new_graph(16);
                graph.expand(y);
                ctx.alloc_graph(&graph, ComputeAllocator::new(backend.buffer_type()))
                    .unwrap();
                let mut out = Vec::new();
                for k in 0..3 {
                    x.set_f32(&partial(r, k, 30));
                    backend.compute(&graph).unwrap();
                    out.push(y.to_f32());
                }
                assert!(handle.comm().error().is_none());
                out
            })
        })
        .collect();
    let results: Vec<Vec<Vec<f32>>> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for (k, (r0, r1)) in results[0].iter().zip(&results[1]).enumerate() {
        let want: Vec<f32> = partial(0, k, 30)
            .iter()
            .zip(partial(1, k, 30))
            .map(|(a, b)| a * 2.0 + b * 2.0)
            .collect();
        assert_eq!(r0, &want);
        assert_eq!(r1, &want);
    }
}

#[test]
fn handles_unregister() {
    let cs = comms(2, Duration::from_secs(5));
    let h = register(cs[0].clone());
    let id = h.id();
    assert!(comm(id).is_some());
    drop(h);
    assert!(comm(id).is_none());
}

/// Every byte of `t` as the loader reads it from `g` (its recipe, or its file range).
fn tensor_bytes(g: &GgufFile, name: &str) -> Vec<u8> {
    let t = g.tensor(name).unwrap();
    match g.recipe(name) {
        Some(Recipe::Computed(f)) => f().unwrap(),
        Some(Recipe::Gather(_)) => panic!("{name}: a gather"),
        None => {
            use std::os::unix::fs::FileExt;
            let path = g.shard_paths().nth(t.file_index).unwrap().to_path_buf();
            let mut b = vec![0u8; t.size as usize];
            std::fs::File::open(path)
                .unwrap()
                .read_exact_at(&mut b, t.offset)
                .unwrap();
            b
        }
    }
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

#[test]
fn shards_of_a_dense_gemma4_cut_heads_and_ffn_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.gguf");
    crate::model::gemma4::tiny::write_dense(&path, 5).unwrap();
    let whole = GgufFile::open(&path).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&whole).unwrap();
    let g4 = crate::model::gemma4::params(&hp).unwrap().clone();
    for rank in 0..2 {
        let share = shard::view(&whole, rank, 2).unwrap();
        let sp = crate::model::core::Hparams::from_gguf(&share).unwrap();
        let s4 = crate::model::gemma4::params(&sp).unwrap();
        assert_eq!(sp.n_head * 2, hp.n_head);
        assert_eq!(sp.n_embd, hp.n_embd);
        for il in 0..hp.n_layer {
            assert_eq!(s4.n_head_kv[il] * 2, g4.n_head_kv[il]);
            assert_eq!(s4.n_ff[il] * 2, g4.n_ff[il]);
        }
        // Mirrored tensors are the same bytes.
        for name in [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.1.attn_q_norm.weight",
        ] {
            assert_eq!(
                tensor_bytes(&share, name),
                tensor_bytes(&whole, name),
                "{name}"
            );
        }
        for il in [0usize, 1] {
            let (d, kv) = g4.kv_dims(il);
            let (d, kv, e) = (d as usize, kv as usize, hp.n_embd as usize);
            let n_head = hp.n_head as usize;
            let ff = g4.n_ff[il] as usize;
            let t = |s: &str| format!("blk.{il}.{s}");
            // Rows: this rank's heads (and FFN rows) in order.
            let rows = |name: &str, per: usize| {
                let all = f32s(&tensor_bytes(&whole, &t(name)));
                let mine = f32s(&tensor_bytes(&share, &t(name)));
                assert_eq!(mine.len() * 2, all.len(), "{name}");
                let start = rank * per * e;
                assert_eq!(mine, all[start..start + per * e], "{name}");
            };
            rows("attn_q.weight", n_head / 2 * d);
            rows("attn_k.weight", kv / 2 * d);
            rows("ffn_gate.weight", ff / 2);
            rows("ffn_up.weight", ff / 2);
            // Columns: every row's share of the inputs.
            let cols = |name: &str, n_in: usize| {
                let all = f32s(&tensor_bytes(&whole, &t(name)));
                let mine = f32s(&tensor_bytes(&share, &t(name)));
                let per = n_in / 2;
                assert_eq!(
                    share.tensor(&t(name)).unwrap().shape,
                    [per as u64, e as u64]
                );
                for row in 0..e {
                    assert_eq!(
                        mine[row * per..(row + 1) * per],
                        all[row * n_in + rank * per..row * n_in + (rank + 1) * per],
                        "{name} row {row}"
                    );
                }
            };
            cols("attn_output.weight", n_head * d);
            cols("ffn_down.weight", ff);
        }
    }
}

#[test]
fn shards_refuse_what_they_cannot_cut() {
    let refused = |g: &GgufFile, n: usize| match shard::view(g, 0, n) {
        Ok(_) => panic!("a share of what cannot be cut"),
        Err(e) => e,
    };
    let dir = tempfile::tempdir().unwrap();
    let dense = dir.path().join("dense.gguf");
    crate::model::gemma4::tiny::write_dense(&dense, 5).unwrap();
    let g = GgufFile::open(&dense).unwrap();
    // Four query heads and two kv heads per layer do not divide among three ranks.
    let e = refused(&g, 3);
    assert!(e.contains("divide"), "{e}");
    // Experts and per-layer embeddings.
    let moe = dir.path().join("moe.gguf");
    crate::model::gemma4::tiny::write(&moe, 5).unwrap();
    let e = refused(&GgufFile::open(&moe).unwrap(), 2);
    assert!(e.contains("experts"), "{e}");
    let qwen = dir.path().join("qwen.gguf");
    crate::model::qwen35::tiny::write(&qwen, &Default::default()).unwrap();
    let e = refused(&GgufFile::open(&qwen).unwrap(), 2);
    assert!(e.contains("Gemma 4"), "{e}");
}

#[test]
fn shards_of_a_gemma4_assistant_cut_its_queries_by_the_targets_kv_heads() {
    let dir = tempfile::tempdir().unwrap();
    let dense = dir.path().join("dense.gguf");
    crate::model::gemma4::tiny::write_dense(&dense, 5).unwrap();
    let target = GgufFile::open(&dense).unwrap();
    let hp = crate::model::core::Hparams::from_gguf(&target).unwrap();
    let g4 = crate::model::gemma4::params(&hp).unwrap().clone();
    let kv = |il: usize| {
        let (d, kv) = g4.kv_dims(il);
        (d as u64, kv as u64)
    };
    let (swa, global) = (kv(hp.n_layer - 2), kv(hp.n_layer - 1));
    let path = dir.path().join("assistant.gguf");
    crate::model::gemma4::tiny::write_assistant(&path, 5).unwrap();
    let whole = GgufFile::open(&path).unwrap();
    let u = |g: &GgufFile, k: &str| g.get_u64(&format!("gemma4-assistant.{k}")).unwrap();
    let ffn = |g: &GgufFile| -> Vec<u64> {
        let v = g.get("gemma4-assistant.feed_forward_length").unwrap();
        match v.as_array() {
            Some(a) => a.iter().map(|x| x.as_u64().unwrap()).collect(),
            None => vec![v.as_u64().unwrap(); 2],
        }
    };
    let e = u(&whole, "embedding_length") as usize;
    let n_head = u(&whole, "attention.head_count") as usize;
    for rank in 0..2 {
        let share = shard::assistant_view(&whole, swa, global, rank, 2).unwrap();
        assert_eq!(u(&share, "attention.head_count") as usize * 2, n_head);
        assert_eq!(
            ffn(&share).iter().map(|f| f * 2).collect::<Vec<_>>(),
            ffn(&whole)
        );
        for name in [
            "token_embd.weight",
            "nextn.pre_projection.weight",
            "nextn.post_projection.weight",
            "blk.0.attn_q_norm.weight",
        ] {
            assert_eq!(
                tensor_bytes(&share, name),
                tensor_bytes(&whole, name),
                "{name}"
            );
        }
        // Layer 0 reads the sliding-window cache, layer 1 the global one: each its head size.
        for (il, d) in [(0usize, swa.0 as usize), (1, global.0 as usize)] {
            let t = |s: &str| format!("blk.{il}.{s}");
            let all = f32s(&tensor_bytes(&whole, &t("attn_q.weight")));
            let mine = f32s(&tensor_bytes(&share, &t("attn_q.weight")));
            let per = n_head / 2 * d * e;
            assert_eq!(
                mine,
                all[rank * per..(rank + 1) * per],
                "layer {il} queries"
            );
            let n_in = n_head * d;
            let all = f32s(&tensor_bytes(&whole, &t("attn_output.weight")));
            let mine = f32s(&tensor_bytes(&share, &t("attn_output.weight")));
            let per = n_in / 2;
            for row in 0..e {
                assert_eq!(
                    mine[row * per..(row + 1) * per],
                    all[row * n_in + rank * per..row * n_in + (rank + 1) * per],
                    "layer {il} output row {row}"
                );
            }
        }
    }
    // Not an assistant; and kv heads that do not divide among the ranks.
    assert!(shard::assistant_view(&target, swa, global, 0, 2).is_err());
    let Err(e) = shard::assistant_view(&whole, (swa.0, 1), global, 0, 2) else {
        panic!("a share of queries over one kv head on two ranks");
    };
    assert!(e.contains("divide"), "{e}");
}

/// `whole` as a safetensors checkpoint would open: synthetic, its layer weights made by
/// recipes (computed, as NVFP4 repacks are, and gathered from two ranges), plus `extra`.
fn synthetic_of(whole: &GgufFile, extra: Vec<(crate::gguf::TensorInfo, Recipe)>) -> GgufFile {
    let mut meta = crate::gguf::GgufWriter::new();
    for (k, v) in whole.metadata() {
        meta.set(&k, &v).unwrap();
    }
    let paths: Vec<_> = whole.shard_paths().map(|p| p.to_path_buf()).collect();
    let mut tensors = whole.tensors();
    let mut recipes = std::collections::HashMap::new();
    for t in &tensors {
        if !t.name.starts_with("blk.") || t.shape.len() < 2 {
            continue; // a plain range, as most of a checkpoint's tensors are
        }
        let r = if t.name.contains("attn_k") || t.name.contains("ffn_up") {
            let half = t.size / 2;
            Recipe::Gather(vec![
                crate::gguf::FileRange {
                    file: t.file_index,
                    offset: t.offset,
                    len: half,
                },
                crate::gguf::FileRange {
                    file: t.file_index,
                    offset: t.offset + half,
                    len: t.size - half,
                },
            ])
        } else {
            let b = tensor_bytes(whole, &t.name);
            Recipe::Computed(Arc::new(move || Ok(b.clone())))
        };
        recipes.insert(t.name.clone(), r);
    }
    for (t, r) in extra {
        recipes.insert(t.name.clone(), r);
        tensors.push(t);
    }
    GgufFile::synthetic(meta, whole.path().to_path_buf(), paths, tensors, recipes).unwrap()
}

/// A safetensors checkpoint's shares (an NVFP4 one's repacked matrices are computed tensors)
/// are the GGUF's, byte for byte, and a per-tensor scale goes whole to every rank.
#[test]
fn shards_of_a_synthetic_gemma4_cut_its_recipes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.gguf");
    crate::model::gemma4::tiny::write_dense(&path, 5).unwrap();
    let whole = GgufFile::open(&path).unwrap();
    let scale = |name: &str, n: usize| {
        let v: Vec<u8> = (0..n)
            .flat_map(|i| (0.5 + i as f32).to_le_bytes())
            .collect();
        let t = crate::gguf::TensorInfo {
            name: name.into(),
            ggml_type: types::F32,
            shape: vec![n as u64],
            file_index: 0,
            offset: 0,
            size: 4 * n as u64,
        };
        (t, Recipe::Computed(Arc::new(move || Ok(v.clone()))))
    };
    let synth = synthetic_of(&whole, vec![scale("blk.0.ffn_down.scale", 1)]);
    for rank in 0..2 {
        let want = shard::view(&whole, rank, 2).unwrap();
        let got = shard::view(&synth, rank, 2).unwrap();
        for t in want.tensors() {
            let g = got.tensor(&t.name).unwrap();
            assert_eq!((&g.shape, g.size), (&t.shape, t.size), "{}", t.name);
            assert_eq!(
                tensor_bytes(&got, &t.name),
                tensor_bytes(&want, &t.name),
                "rank {rank}: {}",
                t.name
            );
        }
        assert_eq!(
            tensor_bytes(&got, "blk.0.ffn_down.scale"),
            0.5f32.to_le_bytes()
        );
    }
    // One scale per row (channel-wise FP8) would need cutting with its rows.
    let synth = synthetic_of(&whole, vec![scale("blk.0.attn_q.scale", 8)]);
    let Err(e) = shard::view(&synth, 0, 2) else {
        panic!("a share of per-row scales");
    };
    assert!(e.contains("per-row"), "{e}");
}

#[test]
fn ffn_rows_prefer_whole_granules() {
    // The 31B's FFN over two ranks: 10752 rows each, in 256s.
    assert_eq!(shard::ffn_rows(21504, 32, 2, 1).unwrap(), (10752, 10752));
    // Not a multiple of 256 per rank: whole quantization blocks.
    assert_eq!(shard::ffn_rows(96, 32, 2, 0).unwrap(), (0, 32));
    assert_eq!(shard::ffn_rows(96, 32, 2, 1).unwrap(), (32, 64));
    assert!(shard::ffn_rows(32, 32, 2, 0).is_err());
}
