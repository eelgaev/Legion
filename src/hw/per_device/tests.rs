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
fn tu11x_names_have_no_tensor_cores() {
    assert!(no_tensor_cores_name("NVIDIA GeForce GTX 1660 Ti"));
    assert!(no_tensor_cores_name("NVIDIA GeForce GTX 1650"));
    assert!(no_tensor_cores_name("NVIDIA GeForce MX450"));
    assert!(no_tensor_cores_name("NVIDIA GeForce MX550"));
    assert!(!no_tensor_cores_name("NVIDIA GeForce RTX 2080 Ti"));
    assert!(!no_tensor_cores_name("Tesla T4"));
    assert!(!no_tensor_cores_name("NVIDIA GeForce RTX 4090"));
}
