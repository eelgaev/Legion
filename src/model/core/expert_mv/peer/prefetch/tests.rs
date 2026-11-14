use super::*;

fn shape(stack: usize) -> Shape {
    Shape {
        stacks: [stack, stack + 1, stack + 2],
        nb2: [1 << 20, 1 << 19],
        vram_below: 100,
        n_w: 356,
    }
}

#[test]
fn layout_keeps_the_parts_apart() {
    let l = Layout {
        slot: 3 << 20,
        peer_slots: 8,
    };
    // The peer's slots, then the two halves.
    assert_eq!(l.slot_at(0), Layout::HDR);
    assert_eq!(l.half(0), 8);
    assert_eq!(l.half(1), 8 + SLOTS);
    // The lists start behind the last prefetch slot, 256-byte aligned, and do not overlap.
    let end = l.slot_at(8 + 2 * SLOTS);
    assert!(l.rest() >= end && l.rest().is_multiple_of(256));
    assert_eq!(l.guess(), l.rest() + LIST);
    // The stamps (a u64 per slot and copy CTA), each half's ids, the progress word and the
    // counters, all 256-byte aligned.
    assert_eq!(l.stamps(0), l.guess() + LIST);
    assert!(l.stamps(1) >= l.stamps(0) + SLOTS * CTAS * 8);
    assert!(l.pf_ids(0) >= l.stamps(1) + SLOTS * CTAS * 8);
    assert!(l.pf_ids(1) >= l.pf_ids(0) + SLOTS * 4);
    assert!(l.progress() >= l.pf_ids(1) + SLOTS * 4);
    assert!(l.count() >= l.progress() + 8);
    for off in [
        l.stamps(0),
        l.stamps(1),
        l.pf_ids(0),
        l.pf_ids(1),
        l.progress(),
        l.count(),
    ] {
        assert!(off.is_multiple_of(256));
    }
    // The seven counters fit, then a layer's two times each, and the zeroed words are exactly
    // those.
    assert!(l.timing(0) >= l.count() + 32 + 3 * 8);
    assert_eq!(l.timing(1), l.timing(0) + 16);
    assert!(l.x() >= l.timing(MAX_LAYERS - 1) + 16 && l.x().is_multiple_of(256));
    assert_eq!(l.stamps(0) + l.state_bytes(), l.x());
    assert_eq!(l.logits(), l.x() + MAX_TOKENS * MAX_EMBD * 4);
    assert_eq!(l.bytes(), l.logits() + MAX_TOKENS * MAX_EXPERTS * 4);
}

#[test]
fn a_layer_is_prefetched_once_the_order_is_known() {
    let mut c = Chain::default();
    // Step 1: the order is learned, nothing to prefetch yet for the last layer.
    let v = c.visit(shape(10)).unwrap();
    assert_eq!((v.layer, v.guess), (0, None));
    assert_eq!(c.ahead(0, 1, 8), None);
    for s in [20, 30] {
        let v = c.visit(shape(s)).unwrap();
        assert_eq!(v.guess, None);
        assert_eq!(c.ahead(v.layer, 1, 8), None);
    }
    // Step 2: each layer's FORK queues the layer after it into the half it does not read, and
    // that layer matches the copy's stamp and half.
    let v0 = c.visit(shape(10)).unwrap();
    assert_eq!((v0.layer, v0.guess), (0, None));
    assert_eq!(c.next(0), Some(shape(20)));
    let a = c.ahead(0, 5, 24).unwrap();
    assert_eq!((a.layer, a.shape), (1, shape(20)));
    let v1 = c.visit(shape(20)).unwrap();
    assert_eq!(v1.guess, Some((5, a.half)));
    let b = c.ahead(1, 6, 16).unwrap();
    assert_eq!((b.layer, b.half), (2, 1 - a.half));
    let v2 = c.visit(shape(30)).unwrap();
    assert_eq!(v2.guess, Some((6, b.half)));
    // The last layer does not prefetch the next step's first one (the peer runs its own layers
    // between them).
    assert_eq!(c.next(2), None);
    assert_eq!(c.ahead(2, 7, 8), None);
    assert_eq!(c.visit(shape(10)).unwrap().guess, None);
    // The halves alternate from launch to launch.
    let d = c.ahead(0, 7, 8).unwrap();
    assert_eq!(d.half, 1 - b.half);
}

#[test]
fn a_guess_for_another_layer_is_not_matched() {
    let mut c = Chain::default();
    for _ in 0..2 {
        for s in [10, 20, 30] {
            let v = c.visit(shape(s)).unwrap();
            c.ahead(v.layer, 1, 8);
        }
    }
    // After layer 0 comes layer 2 this time: the prefetch was layer 1's.
    let v = c.visit(shape(10)).unwrap();
    assert_eq!(c.ahead(v.layer, 2, 8).unwrap().layer, 1);
    let v = c.visit(shape(30)).unwrap();
    assert_eq!(v.guess, None);
    // And the order follows: layer 0's next is layer 2 now.
    c.ahead(v.layer, 3, 8);
    let v = c.visit(shape(10)).unwrap();
    assert_eq!(c.ahead(v.layer, 3, 8).unwrap().layer, 2);
}

#[test]
fn a_guess_is_used_once_and_only_with_the_same_bounds() {
    let mut c = Chain::default();
    for _ in 0..2 {
        for s in [10, 20] {
            let v = c.visit(shape(s)).unwrap();
            c.ahead(v.layer, 1, 8);
        }
    }
    // Prefetched for layer 1, then the buffer is new: no guess.
    c.visit(shape(10)).unwrap();
    assert!(c.ahead(0, 2, 8).is_some());
    c.reset();
    let v = c.visit(shape(20)).unwrap();
    assert_eq!(v.guess, None);
    // A layer that queued nothing leaves nothing for the next one.
    assert_eq!(c.visit(shape(10)).unwrap().guess, None);
    assert_eq!(c.ahead(0, 3, 0), None);
    assert_eq!(c.visit(shape(20)).unwrap().guess, None);
    // The cache's slot count changed since the copy: its ids were another table's.
    c.visit(shape(10)).unwrap();
    assert!(c.ahead(0, 3, 8).is_some());
    let moved = Shape {
        vram_below: 90,
        ..shape(20)
    };
    assert_eq!(c.visit(moved).unwrap().guess, None);
    // A layer of other bytes is prefetched with its own.
    let mut c = Chain::default();
    let other = Shape {
        nb2: [1 << 21, 1 << 19],
        ..shape(20)
    };
    for _ in 0..2 {
        c.visit(shape(10)).unwrap();
        c.visit(other).unwrap();
    }
    c.visit(shape(10)).unwrap();
    assert_eq!(c.ahead(0, 1, 8).unwrap().shape, other);
}

#[test]
fn layers_past_the_limit_run_without() {
    let mut c = Chain::default();
    for s in 0..MAX_LAYERS {
        assert!(c.visit(shape(10 * s)).is_some());
    }
    assert_eq!(c.visit(shape(10 * MAX_LAYERS)), None);
}

#[test]
fn routers_are_found_by_their_gate_stack() {
    let r = Router {
        w: 0x1000,
        f16: true,
        n_embd: 2048,
        n_expert: 256,
        table: 0x2000,
    };
    register(0x7001, r);
    assert_eq!(router_of(0x7001), Some(r));
    assert_eq!(router_of(0x7002), None);
    // The last word counts; a router the kernels do not take is none.
    register(0x7001, Router { table: 0x3000, ..r });
    assert_eq!(router_of(0x7001).unwrap().table, 0x3000);
    register(
        0x7001,
        Router {
            n_expert: MAX_EXPERTS + 1,
            ..r
        },
    );
    assert_eq!(router_of(0x7001), None);
}

#[test]
fn on_only_when_wanted_and_allowed() {
    assert!(!enabled());
    set_auto(true);
    assert!(!enabled());
    set_enabled(true);
    assert!(enabled());
    set_auto(false);
    assert!(!enabled());
    set_enabled(false);
}
