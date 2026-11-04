use super::*;

fn host_tensors(ctx: &Context) -> (Tensor<'_>, Tensor<'_>) {
    let a = ctx.new_tensor(types::I32, &[5]).named("a");
    let b = ctx.new_tensor(types::F32, &[6]).named("b");
    ctx.alloc_tensors(Backend::cpu().unwrap().buffer_type())
        .unwrap();
    (a, b)
}

/// On the CPU (no pinned memory, no queued copies) every transfer is the plain synchronous
/// one: writes land at once, reads are taken after `finish`.
#[test]
fn cpu_transfers_fall_back_to_plain_copies() {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let (a, b) = host_tensors(&ctx);
    let mut io = PinnedIo::default();
    for round in 0..3 {
        io.begin(&mut backend, 1 << 20);
        let ints: Vec<i32> = (0..5).map(|i| i * 7 - round).collect();
        io.write_i32(&mut backend, a, &ints);
        io.write_f32_parts(
            &mut backend,
            b,
            &[&[1.5, -2.0], &[], &[3.25, 4.0, 5.0, round as f32]],
        );
        assert_eq!(a.to_i32(), ints, "written at once");
        let ra = io.read(&mut backend, a);
        let rb = io.read(&mut backend, b);
        assert!(matches!(ra, Read::Later(_)) && matches!(rb, Read::Later(_)));
        io.finish(&mut backend);
        assert_eq!(io.take_i32(ra), ints);
        assert_eq!(
            io.take_f32(rb),
            vec![1.5, -2.0, 3.25, 4.0, 5.0, round as f32]
        );
        // Nothing queueable asked for pinned memory.
        assert_eq!(io.capacity(), 0);
    }
}

#[test]
#[should_panic(expected = "size")]
fn a_write_must_cover_the_whole_tensor() {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let (_, b) = host_tensors(&ctx);
    let mut io = PinnedIo::default();
    io.begin(&mut backend, 1 << 20);
    io.write_f32(&mut backend, b, &[1.0; 5]);
}

#[test]
#[should_panic(expected = "non-I32")]
fn types_are_checked() {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let (_, b) = host_tensors(&ctx);
    let mut io = PinnedIo::default();
    io.begin(&mut backend, 1 << 20);
    io.write_i32(&mut backend, b, &[0; 6]);
}

#[test]
fn bytes_round_trip_in_memory_order() {
    let v = [1.0f32, -0.5, f32::MIN_POSITIVE, 1e30];
    assert_eq!(from_bytes(as_bytes(&v), f32::from_ne_bytes), v);
    let w = [i32::MIN, -1, 0, 7, i32::MAX];
    assert_eq!(from_bytes(as_bytes(&w), i32::from_ne_bytes), w);
}

/// On a GPU: inputs queued through pinned memory reach the device, reads queued after them
/// come back after `finish`, and the buffer grows to what a pass wanted (the first pass
/// copies synchronously, later ones through pinned memory), up to the cap.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn gpu_transfers_queue_through_pinned_memory() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let dev = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU");
    let mut backend = dev.init().unwrap();
    let ctx = Context::new(2, 0);
    let a = ctx.new_tensor(types::I32, &[1000]).named("a");
    let b = ctx.new_tensor(types::F32, &[3000]).named("b");
    ctx.alloc_tensors(dev.buffer_type()).unwrap();
    assert!(backend.queues_copies_of(a));
    let mut io = PinnedIo::default();
    for round in 0..4 {
        io.begin(&mut backend, 1 << 20);
        if round > 0 {
            assert!(io.capacity() >= 4000 + 12000, "round {round}");
        }
        let ints: Vec<i32> = (0..1000).map(|i| i * 3 - round).collect();
        let floats: Vec<f32> = (0..3000).map(|i| i as f32 * 0.5 + round as f32).collect();
        io.write_i32(&mut backend, a, &ints);
        io.write_f32_parts(&mut backend, b, &[&floats[..1000], &floats[1000..]]);
        let ra = io.read(&mut backend, a);
        // Read back after the queued writes, on the same stream.
        let rc = io.read(&mut backend, b);
        assert_eq!(
            matches!(ra, Read::Queued { .. }),
            round > 0,
            "round {round}"
        );
        io.finish(&mut backend);
        assert_eq!(io.take_i32(ra), ints);
        assert_eq!(io.take_f32(rc), floats);
    }
    // The cap holds.
    io.begin(&mut backend, 1 << 20);
    let big = ctx_big(&dev);
    let t = big.tensor("big").unwrap();
    io.write_f32(&mut backend, t, &vec![1.0; t.elements()]);
    io.begin(&mut backend, 8192);
    assert!(io.capacity() < 4 << 20);
    // A reserve grows it at once to what is asked; passes within it then never regrow it.
    io.reserve(&mut backend, 2 << 20);
    let reserved = io.capacity();
    assert!(reserved >= 2 << 20);
    for _ in 0..2 {
        io.begin(&mut backend, 2 << 20);
        assert_eq!(io.capacity(), reserved);
        io.write_f32(&mut backend, t, &vec![2.0; t.elements()]);
        io.finish(&mut backend);
    }
    io.begin(&mut backend, 2 << 20);
    assert_eq!(io.capacity(), reserved);
}

/// Without pinned memory (the CPU) a reserve is a no-op and transfers stay plain copies.
#[test]
fn cpu_reserve_keeps_plain_copies() {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let (a, _) = host_tensors(&ctx);
    let mut io = PinnedIo::default();
    io.reserve(&mut backend, 1 << 20);
    assert_eq!(io.capacity(), 0);
    io.begin(&mut backend, 1 << 20);
    io.write_i32(&mut backend, a, &[1, 2, 3, 4, 5]);
    assert_eq!(a.to_i32(), vec![1, 2, 3, 4, 5]);
    io.finish(&mut backend);
}

/// Ordered writes that cannot be queued still land at once (after the backend's queued
/// work), and a settled buffer starts its next pass without waiting.
#[test]
fn ordered_fallback_and_settle() {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let (a, b) = host_tensors(&ctx);
    let mut io = PinnedIo::default();
    io.set_ordered(true);
    io.begin(&mut backend, 1 << 20);
    io.write_i32(&mut backend, a, &[5, 4, 3, 2, 1]);
    io.write_f32(&mut backend, b, &[0.5; 6]);
    assert_eq!(a.to_i32(), vec![5, 4, 3, 2, 1]);
    assert_eq!(b.to_f32(), vec![0.5; 6]);
    assert!(!io.pending);
    io.set_ordered(false);
    io.pending = true;
    io.settle();
    assert!(!io.pending);
    io.begin(&mut backend, 1 << 20);
    io.write_i32(&mut backend, a, &[1, 1, 2, 3, 5]);
    assert_eq!(a.to_i32(), vec![1, 1, 2, 3, 5]);
}

#[cfg(feature = "cuda")]
fn ctx_big(dev: &crate::ggml::Device) -> Context {
    let ctx = Context::new(1, 0);
    ctx.new_tensor(types::F32, &[1 << 20]).named("big");
    ctx.alloc_tensors(dev.buffer_type()).unwrap();
    ctx
}

/// `has_room`: what is left since `begin`, aligned per transfer; asking for more than that
/// grows the buffer at the next `begin` (here none: the CPU has no pinned memory).
#[test]
fn room_is_what_is_left_aligned() {
    let mut io = PinnedIo::default();
    assert!(io.has_room(&[]));
    assert!(!io.has_room(&[4, 4]));
    assert_eq!(io.wanted, 2 * ALIGN);
    let mut backend = Backend::cpu().unwrap();
    io.begin(&mut backend, 1 << 20);
    assert_eq!((io.wanted, io.capacity()), (0, 0));
}

#[test]
fn a_fallback_read_is_not_queued() {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let (a, _) = host_tensors(&ctx);
    let mut io = PinnedIo::default();
    io.begin(&mut backend, 1 << 20);
    let r = io.read(&mut backend, a);
    assert!(!r.is_queued());
    assert!(Read::Queued { at: 0, len: 4 }.is_queued());
    io.finish(&mut backend);
    let _ = io.take(r);
}

/// On the CPU a packed pass is a plain one: nothing held, writes land at once, the flush does
/// nothing.
#[test]
fn cpu_packed_pass_is_a_plain_one() {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let (a, b) = host_tensors(&ctx);
    let mut io = PinnedIo::default();
    io.begin_packed(&mut backend, 1 << 20);
    assert!(!io.is_packing());
    io.write_i32(&mut backend, a, &[9, 8, 7, 6, 5]);
    io.write_f32(&mut backend, b, &[0.25; 6]);
    assert_eq!(a.to_i32(), vec![9, 8, 7, 6, 5]);
    io.flush(&mut backend);
    io.finish(&mut backend);
    assert_eq!(b.to_f32(), vec![0.25; 6]);
}

/// Held writes left unflushed are a caller bug, caught at the next `begin`.
#[test]
#[should_panic(expected = "never flushed")]
fn held_writes_must_be_flushed() {
    let mut io = PinnedIo::default();
    let mut held = pack::Pending::default();
    held.push(0x1000, 0, 4);
    io.pack = Some((0, held));
    let mut backend = Backend::cpu().unwrap();
    io.begin(&mut backend, 1 << 20);
}

/// On a GPU: a packed pass's writes (small ones, one over many kernel pieces, more pieces than
/// one launch takes, a tensor written twice) land with the same bytes as plain queued copies,
/// once flushed ahead of the reads.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn gpu_packed_writes_land_as_copies() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let dev = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU");
    let mut backend = dev.init().unwrap();
    let ctx = Context::new(5, 0);
    let a = ctx.new_tensor(types::I32, &[3]).named("a");
    let b = ctx.new_tensor(types::F32, &[3001]).named("b");
    // More than `MAX_PIECES` pieces of `CHUNK` bytes.
    let n_big = (pack::MAX_PIECES + 7) * pack::CHUNK / 4 + 5;
    let c = ctx.new_tensor(types::F32, &[n_big as i64]).named("c");
    let v = ctx.new_tensor(types::I32, &[16]).named("v");
    ctx.alloc_tensors(dev.buffer_type()).unwrap();
    // A view at an odd 4-byte offset: the kernel's unaligned path.
    let d = ctx.view_1d(v, 5, 4);
    let mut io = PinnedIo::default();
    v.set_i32(&[0; 16]);
    for round in 0..3 {
        io.begin_packed(&mut backend, 8 << 20);
        // The first pass grows the buffer (plain copies); later ones pack.
        assert_eq!(io.is_packing(), round > 0, "round {round}");
        let ints = [round, -round, 7 * round];
        let floats: Vec<f32> = (0..3001).map(|i| i as f32 * 0.25 - round as f32).collect();
        let big: Vec<f32> = (0..n_big)
            .map(|i| (i % 977) as f32 + round as f32)
            .collect();
        let five: Vec<i32> = (0..5).map(|i| 100 * round + i).collect();
        io.write_i32(&mut backend, a, &[-1, -1, -1]);
        io.write_f32(&mut backend, b, &floats);
        io.write_f32(&mut backend, c, &big);
        io.write_i32(&mut backend, d, &five);
        // Again: after the held write, as plain copies were.
        io.write_i32(&mut backend, a, &ints);
        io.flush(&mut backend);
        assert!(!io.is_packing());
        let ra = io.read(&mut backend, a);
        let rb = io.read(&mut backend, b);
        io.finish(&mut backend);
        assert_eq!(io.take_i32(ra), ints);
        assert_eq!(io.take_f32(rb), floats);
        assert_eq!(c.to_f32(), big);
        let mut want = vec![0; 16];
        want[1..6].copy_from_slice(&five);
        assert_eq!(v.to_i32(), want);
    }
}

/// Without events (the CPU) there is no fence: the caller waits with `finish`. A read made at
/// `take` does not detach from its tensor; a queued one does.
#[test]
fn cpu_has_no_fence() {
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let (a, _) = host_tensors(&ctx);
    let mut io = PinnedIo::default();
    io.begin(&mut backend, 1 << 20);
    assert!(io.fence(&mut backend).is_none());
    assert!(io.read(&mut backend, a).detach().is_none());
    assert!(matches!(
        Read::Queued { at: 256, len: 8 }.detach(),
        Some(Read::Queued { at: 256, len: 8 })
    ));
    io.finish(&mut backend);
}

/// A queued read past what the fences waited for cannot be taken while copies may run.
#[test]
#[should_panic(expected = "before finish")]
fn a_read_past_the_fences_waits_for_finish() {
    let mut io = PinnedIo {
        pending: true,
        landed: 256,
        ..Default::default()
    };
    let _ = io.take(Read::Queued { at: 256, len: 4 });
}

/// On a GPU: after waiting for a fence, the reads queued before it can be taken while the
/// copies queued after it are still pending; the rest after `finish`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn gpu_fence_lands_the_reads_before_it() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let dev = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU");
    let mut backend = dev.init().unwrap();
    let ctx = Context::new(2, 0);
    let a = ctx.new_tensor(types::I32, &[1000]).named("a");
    let b = ctx.new_tensor(types::F32, &[3000]).named("b");
    ctx.alloc_tensors(dev.buffer_type()).unwrap();
    let mut io = PinnedIo::default();
    io.reserve(&mut backend, 1 << 20);
    for round in 0..3 {
        io.begin(&mut backend, 1 << 20);
        let ints: Vec<i32> = (0..1000).map(|i| i * 5 - round).collect();
        let floats: Vec<f32> = (0..3000).map(|i| i as f32 - round as f32).collect();
        io.write_i32(&mut backend, a, &ints);
        let ra = io.read(&mut backend, a).detach().expect("queued");
        let fence = io.fence(&mut backend).expect("CUDA has events");
        io.write_f32(&mut backend, b, &floats);
        let rb = io.read(&mut backend, b);
        io.wait(fence);
        assert_eq!(io.take_i32(ra), ints, "round {round}");
        io.finish(&mut backend);
        assert_eq!(io.take_f32(rb), floats, "round {round}");
    }
}

/// A fence covers the buffer only if nothing was handed out or held for a launch after it.
#[test]
fn a_fence_covers_only_what_was_queued_before_it() {
    assert!(covers_all(0, 0, false));
    assert!(covers_all(512, 512, false));
    assert!(
        !covers_all(256, 512, false),
        "a copy claimed after the fence"
    );
    assert!(
        !covers_all(512, 512, true),
        "a write held for the packed launch"
    );
}

/// On a GPU: a fence after every copy of a pass lets the next pass begin without waiting for
/// the work queued after the fence, and its copies still land in order behind that work.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs a GPU"]
fn gpu_fence_over_all_copies_ends_the_pass() {
    use crate::ggml::{DeviceKind, device_by_kind};
    let dev = device_by_kind(DeviceKind::Gpu)
        .or_else(|| device_by_kind(DeviceKind::IntegratedGpu))
        .expect("no GPU");
    let mut backend = dev.init().unwrap();
    let ctx = Context::new(1, 0);
    let a = ctx.new_tensor(types::I32, &[1000]).named("a");
    ctx.alloc_tensors(dev.buffer_type()).unwrap();
    let mut io = PinnedIo::default();
    io.reserve(&mut backend, 1 << 20);
    for round in 0..3 {
        io.begin(&mut backend, 1 << 20);
        let ints: Vec<i32> = (0..1000).map(|i| i * 3 + round).collect();
        io.write_i32(&mut backend, a, &ints);
        let ra = io.read(&mut backend, a).detach().expect("queued");
        let fence = io.fence(&mut backend).expect("CUDA has events");
        io.wait(fence);
        assert!(
            !io.pending,
            "round {round}: every copy was before the fence"
        );
        assert_eq!(io.take_i32(ra), ints, "round {round}");
    }
    io.finish(&mut backend);
}
