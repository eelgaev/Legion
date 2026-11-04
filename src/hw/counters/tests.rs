use super::*;

fn uuids() -> Vec<String> {
    ["GPU-aaaa-1", "GPU-bbbb-2", "GPU-cccc-3", "GPU-dddd-4"]
        .map(String::from)
        .to_vec()
}

#[test]
fn visible_gpus_follow_cuda_visible_devices() {
    assert_eq!(visible_gpus(None, &uuids()), vec![0, 1, 2, 3]);
    assert_eq!(visible_gpus(Some("2,3"), &uuids()), vec![2, 3]);
    assert_eq!(visible_gpus(Some(" 3 , 1 "), &uuids()), vec![3, 1]);
    assert_eq!(visible_gpus(Some("GPU-cccc"), &uuids()), vec![2]);
    // CUDA ignores everything from the first invalid entry on.
    assert_eq!(visible_gpus(Some("1,9,2"), &uuids()), vec![1]);
    assert!(visible_gpus(Some(""), &uuids()).is_empty());
}

#[test]
fn counter_increases_skip_resets() {
    let mut last = None;
    assert_eq!(increase(&mut last, Some(100)), None);
    assert_eq!(increase(&mut last, Some(160)), Some(60));
    assert_eq!(increase(&mut last, None), None);
    assert_eq!(increase(&mut last, Some(10)), None);
    assert_eq!(increase(&mut last, Some(15)), Some(5));
}
