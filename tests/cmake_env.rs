//! Unit tests for `build/cmake_env.rs`, the env-switch parser used by `build.rs`.

#[path = "../build/cmake_env.rs"]
mod cmake_env;

use cmake_env::{arch_specific, archs_from_compute_caps, cmake_bool};

#[test]
fn truthy_values_map_to_on() {
    for v in ["1", "on", "ON", "True", " yes "] {
        assert_eq!(cmake_bool(v), Ok("ON"), "{v}");
    }
}

#[test]
fn falsy_values_map_to_off() {
    for v in ["0", "off", "OFF", "False", "no"] {
        assert_eq!(cmake_bool(v), Ok("OFF"), "{v}");
    }
}

#[test]
fn unknown_values_are_rejected() {
    assert!(cmake_bool("maybe").is_err());
    assert!(cmake_bool("").is_err());
}

#[test]
fn blackwell_consumer_archs_become_arch_specific() {
    assert_eq!(arch_specific("121"), "121a");
    assert_eq!(arch_specific("120-real"), "120a-real");
    assert_eq!(arch_specific("121a"), "121a");
    assert_eq!(arch_specific("70"), "70");
    assert_eq!(arch_specific("90-virtual"), "90-virtual");
    assert_eq!(arch_specific("native"), "native");
}

#[test]
fn compute_caps_parse() {
    assert_eq!(archs_from_compute_caps("12.1\n"), vec!["121"]);
    assert_eq!(archs_from_compute_caps("7.0\n7.0\n8.9\n"), vec!["70", "89"]);
    assert!(archs_from_compute_caps("").is_empty());
    assert!(archs_from_compute_caps("No devices were found\n").is_empty());
}
