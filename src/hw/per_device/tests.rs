use std::cell::Cell;

use super::*;

#[test]
fn each_device_keeps_its_own_answer() {
    let slots = PerDevice::new();
    let asked = Cell::new(0);
    // A box with a GB10-like device 0 and a V100-like device 1.
    let probe = |dev: usize| {
        asked.set(asked.get() + 1);
        if dev == 0 { 121 } else { 70 }
    };
    assert_eq!(slots.get(0, probe), 121);
    assert_eq!(slots.get(1, probe), 70);
    assert_eq!(slots.get(0, probe), 121);
    assert_eq!(slots.get(1, probe), 70);
    assert_eq!(asked.get(), 2, "one probe per device");
}

#[test]
fn an_unset_answer_is_asked_again() {
    let slots = PerDevice::new();
    let asked = Cell::new(0);
    let probe = |_| {
        asked.set(asked.get() + 1);
        i32::MIN
    };
    slots.get(3, probe);
    slots.get(3, probe);
    assert_eq!(asked.get(), 2);
    assert_eq!(slots.get(3, |_| 5), 5);
    assert_eq!(slots.get(3, |_| 6), 5);
}

#[test]
fn a_device_past_the_slots_is_asked_every_time() {
    let slots = PerDevice::new();
    let asked = Cell::new(0);
    let probe = |dev: usize| {
        asked.set(asked.get() + 1);
        dev as i32
    };
    assert_eq!(slots.get(DEVICES, probe), DEVICES as i32);
    assert_eq!(slots.get(DEVICES, probe), DEVICES as i32);
    assert_eq!(asked.get(), 2);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn without_cuda_no_device_answers() {
    assert_eq!(compute_capability(0), None);
    assert_eq!(cc(0), None);
    assert_eq!(current(), None);
    assert_eq!(sm_count(0), 0);
    assert_eq!(sms_if_cc12(0), 0);
}

#[test]
fn ampere_is_8_0_to_8_7() {
    for cc in [(8, 0), (8, 6), (8, 7)] {
        assert!(ampere(cc), "{cc:?}");
    }
    for cc in [(7, 0), (7, 5), (8, 9), (9, 0), (12, 0), (12, 1)] {
        assert!(!ampere(cc), "{cc:?}");
    }
}

#[test]
fn a_pick_follows_the_device_it_is_asked_for() {
    let rule = |cc: Option<(i32, i32)>| cc.map_or(-1, |(major, minor)| 10 * major + minor);
    let picks: Vec<i32> = (0..6).map(|d| pick_by(Some(d), 0, rule, test_cc)).collect();
    assert_eq!(picks, [121, 89, 70, 80, 120, -1]);
    // Tensors that name no device: the engine's answer, whatever the devices are.
    assert_eq!(pick_by(None, 42, rule, test_cc), 42);
    assert_eq!(pick_by(None, 42, rule, |_| unreachable!()), 42);
}

#[test]
fn sms_follow_the_device_they_are_asked_for() {
    // The fake box's SM counts: a GB10's 48, a 4090's 128, a V100's 80, an A100's 108, a
    // 5090's 170.
    let sms_of = |dev: usize| [48, 128, 80, 108, 170].get(dev).copied().unwrap_or(0);
    for engine in [0, 48] {
        let sms: Vec<i32> = (0..6)
            .map(|d| sms_if_cc12_by(Some(d), engine, test_cc, sms_of))
            .collect();
        assert_eq!(sms, [48, 0, 0, 0, 170, 0], "engine {engine}");
        assert_eq!(sms_if_cc12_by(None, engine, test_cc, sms_of), engine);
    }
}
