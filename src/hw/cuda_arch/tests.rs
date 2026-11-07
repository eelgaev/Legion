use super::*;

#[test]
fn levels_follow_the_kernels_guards() {
    assert_eq!(level(61), None);
    assert_eq!(level(70), Some(70));
    assert_eq!(level(75), Some(70));
    assert_eq!(level(80), Some(80));
    assert_eq!(level(86), Some(80));
    assert_eq!(level(89), Some(89));
    assert_eq!(level(90), Some(90));
    assert_eq!(level(100), Some(90));
    assert_eq!(level(121), Some(90));
}

#[test]
fn code_of_the_gpus_level_runs() {
    // V100 on an sm_70 build, GB10 on 121a, an L40 on 89, an A100 on an 80 build.
    assert_eq!(missing_code(0, "V100", 70, Some(70)), None);
    assert_eq!(missing_code(0, "GB10", 121, Some(121)), None);
    assert_eq!(missing_code(0, "L40", 89, Some(89)), None);
    assert_eq!(missing_code(0, "A100", 80, Some(80)), None);
    // An 86 card on 80 code: the same level.
    assert_eq!(missing_code(0, "A6000", 86, Some(80)), None);
    // Below every level Legion builds for: nothing to check.
    assert_eq!(missing_code(0, "P100", 60, Some(60)), None);
}

#[test]
fn older_code_on_a_newer_gpu_is_refused() {
    let e = missing_code(1, "GB10", 121, Some(80)).expect("refused");
    assert!(e.contains("GPU 1 (GB10, compute capability 12.1)"), "{e}");
    assert!(
        e.contains("built for compute capability 8.0 and older"),
        "{e}"
    );
    assert!(e.contains("including 121a"), "{e}");
    let e = missing_code(0, "A100", 80, Some(70)).expect("refused");
    assert!(e.contains("including 80 "), "{e}");
    let e = missing_code(0, "H100", 90, Some(89)).expect("refused");
    assert!(e.contains("including 90a"), "{e}");
}

#[test]
fn no_code_at_all_is_refused() {
    let e = missing_code(2, "V100", 70, None).expect("refused");
    assert!(e.contains("without code it runs"), "{e}");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn without_cuda_nothing_is_refused() {
    assert_eq!(image(0), Image::Unknown);
    assert_eq!(check(0, || unreachable!()), None);
}
