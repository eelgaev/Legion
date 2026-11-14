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
    assert_eq!(l.guess(0), l.rest() + LIST);
    assert_eq!(l.guess_slots(0), l.guess(0) + LIST);
    assert_eq!(l.guess(1), l.guess_slots(0) + LIST);
    assert_eq!(l.guess_slots(1), l.guess(1) + LIST);
    assert_eq!(l.count(), l.guess_slots(1) + LIST);
    assert!(l.x() >= l.count() + 32);
    assert_eq!(l.logits(), l.x() + MAX_TOKENS * MAX_EMBD * 4);
    assert_eq!(l.bytes(), l.logits() + MAX_TOKENS * MAX_EXPERTS * 4);
}

#[test]
fn a_layer_is_prefetched_once_the_order_is_known() {
    let mut c = Chain::default();
    // Step 1: the order is learned, nothing to prefetch yet for the last layer.
    let v = c.visit(shape(10)).unwrap();
    assert_eq!((v.layer, v.guess), (0, None));
    assert_eq!(c.ahead(0, 8), None);
    for s in [20, 30] {
        let v = c.visit(shape(s)).unwrap();
        assert_eq!(v.guess, None);
        assert_eq!(c.ahead(v.layer, 8), None);
    }
    // Step 2: each layer's FORK queues the layer after it into the half it does not read, and
    // that layer matches the guess.
    let v0 = c.visit(shape(10)).unwrap();
    assert_eq!((v0.layer, v0.guess), (0, None));
    assert_eq!(c.next(0), Some(shape(20)));
    let a = c.ahead(0, 24).unwrap();
    assert_eq!((a.layer, a.shape), (1, shape(20)));
    let v1 = c.visit(shape(20)).unwrap();
    assert_eq!(v1.guess, Some((24, a.half)));
    let b = c.ahead(1, 16).unwrap();
    assert_eq!((b.layer, b.half), (2, 1 - a.half));
    let v2 = c.visit(shape(30)).unwrap();
    assert_eq!(v2.guess, Some((16, b.half)));
    // The last layer prefetches the next step's first one.
    let w = c.ahead(2, 8).unwrap();
    assert_eq!((w.layer, w.half), (0, 1 - b.half));
    assert_eq!(c.visit(shape(10)).unwrap().guess, Some((8, w.half)));
}

#[test]
fn a_guess_for_another_layer_is_not_matched() {
    let mut c = Chain::default();
    for _ in 0..2 {
        for s in [10, 20, 30] {
            let v = c.visit(shape(s)).unwrap();
            c.ahead(v.layer, 8);
        }
    }
    // After layer 0 comes layer 2 this time: the prefetch was layer 1's.
    let v = c.visit(shape(10)).unwrap();
    assert_eq!(c.ahead(v.layer, 8).unwrap().layer, 1);
    let v = c.visit(shape(30)).unwrap();
    assert_eq!(v.guess, None);
    // And the order follows: layer 0's next is layer 2 now.
    c.ahead(v.layer, 8);
    let v = c.visit(shape(10)).unwrap();
    assert_eq!(c.ahead(v.layer, 8).unwrap().layer, 2);
}

#[test]
fn a_guess_is_used_once_and_only_with_the_same_bounds() {
    let mut c = Chain::default();
    for _ in 0..2 {
        for s in [10, 20] {
            let v = c.visit(shape(s)).unwrap();
            c.ahead(v.layer, 8);
        }
    }
    // Prefetched for layer 0, then the buffer is new: no guess.
    c.reset();
    let v = c.visit(shape(10)).unwrap();
    assert_eq!(v.guess, None);
    // A layer that queued nothing leaves nothing for the next one.
    assert_eq!(c.visit(shape(20)).unwrap().guess, None);
    assert_eq!(c.ahead(1, 0), None);
    assert_eq!(c.visit(shape(10)).unwrap().guess, None);
    // The cache's slot count changed since the copy: its ids were another table's.
    assert!(c.ahead(0, 8).is_some());
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
    assert_eq!(c.ahead(0, 8).unwrap().shape, other);
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
