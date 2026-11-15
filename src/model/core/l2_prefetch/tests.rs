//! `l2_prefetch`: the node's shape and parameters, and its switches.

use super::*;

#[test]
fn the_switch_takes_whole_mib() {
    assert_eq!(gdn_z_bytes_from(None), GDN_Z_MIB << 20);
    assert_eq!(gdn_z_bytes_from(Some("0")), 0);
    assert_eq!(gdn_z_bytes_from(Some("4")), 4 << 20);
    assert_eq!(gdn_z_bytes_from(Some(" 8 ")), 8 << 20);
    assert_eq!(gdn_z_bytes_from(Some("65")), GDN_Z_MIB << 20);
    assert_eq!(gdn_z_bytes_from(Some("x")), GDN_Z_MIB << 20);
}

#[test]
// The pace and the delay share one parser.
fn the_pace_takes_nanoseconds_up_to_a_millisecond() {
    assert_eq!(pace_from(None), 0);
    assert_eq!(pace_from(Some("1000")), 1000);
    assert_eq!(pace_from(Some(" 600 ")), 600);
    assert_eq!(pace_from(Some("1000001")), 0);
    assert_eq!(pace_from(Some("-5")), 0);
}

#[test]
fn spans_stay_inside_the_tensor() {
    // 31.5 MB: the 27B's FP8 z projection (5120 x 6144 bytes).
    let z = 5120 * 6144;
    assert_eq!(span(0, 4 << 20, z), (0, 1024));
    // More than the tensor: all of it.
    assert_eq!(span(0, 64 << 20, z), (0, z / CHUNK));
    // An offset near the end keeps at least one chunk and none past the end.
    let (first, n) = span(z - 100, 4 << 20, z);
    assert_eq!((first, n), (z / CHUNK - 1, 1));
    // Under a chunk: nothing; a partial chunk is dropped.
    assert_eq!(span(0, 4096, 4000), (0, 0));
    assert_eq!(span(0, 6000, 3 * CHUNK), (0, 1));
}

#[test]
fn the_node_reads_the_weight_after_its_anchor() {
    let ctx = Context::new(16, 0);
    let w = ctx.new_tensor(types::I8, &[5120, 6144]);
    let x = ctx.new_tensor(types::F32, &[16, 8]);
    let after = ctx.scale(x, 2.0);
    let node = ahead(&ctx, w, 0, 8 << 20, after, false);
    assert_eq!(node.ne(), [1, 1, 1, 1]);
    assert!(reads(node, w));
    assert!(reads(node, after));
    // x through `after`, one level down; an unrelated tensor not at all.
    assert!(reads(node, x));
    assert!(!reads(node, ctx.new_tensor(types::F32, &[4])));
    // SAFETY: a node of the live context.
    let [first, n, side, ..] = external_params(unsafe { &*node.raw() });
    assert_eq!((first, n, side), (0, 2048, 0));
}

#[test]
fn a_side_node_and_its_join() {
    let ctx = Context::new(16, 0);
    let w = ctx.new_tensor(types::I8, &[5120, 6144]);
    let x = ctx.new_tensor(types::F32, &[16, 8]);
    let qkv = ctx.scale(x, 2.0);
    let node = ahead(&ctx, w, 0, 4 << 20, qkv, true);
    // SAFETY: a node of the live context.
    let [first, n, side, ..] = external_params(unsafe { &*node.raw() });
    assert_eq!((first, n, side), (0, 1024, 1));
    let out = ctx.scale(qkv, 0.5);
    let j = join(&ctx, node, out);
    assert_eq!(j.ne(), [1, 1, 1, 1]);
    // The layer's output first (its chain runs before the join), then the prefetch.
    // SAFETY: a node of the live context.
    let src = unsafe { (*j.raw()).src };
    assert!(std::ptr::eq(src[0], out.raw()));
    assert!(std::ptr::eq(src[1], node.raw()));
}

#[test]
fn a_side_join_keeps_the_side_work_and_its_inputs() {
    let ctx = Context::new(16, 0);
    let x = ctx.new_tensor(types::F32, &[16, 8]);
    let z = ctx.scale(x, 3.0);
    let out = ctx.scale(x, 0.5);
    let j = side_join(&ctx, out, &[z, x]);
    assert_eq!(j.ne(), [1, 1, 1, 1]);
    // SAFETY: a node of the live context.
    let src = unsafe { (*j.raw()).src };
    assert!(std::ptr::eq(src[0], out.raw()));
    assert!(std::ptr::eq(src[1], z.raw()));
    assert!(std::ptr::eq(src[2], x.raw()));
    assert!(src[3].is_null());
}
