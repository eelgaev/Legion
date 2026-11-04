use super::*;

#[test]
fn lookup_by_gpu_name() {
    let v = lookup("Tesla V100-SXM2-32GB").unwrap();
    assert_eq!(v.memory_gb_s, 894.0);
    assert_eq!(v.link_gb_s, 68.5);
    assert_eq!(lookup("NVIDIA GB10").unwrap().memory_gb_s, 233.0);
    assert!(lookup("NVIDIA GeForce RTX 4090").is_none());
}

#[test]
fn stage_rooflines_add_up_over_tensor_parallel_gpus() {
    let one = StageRoofline::of("Tesla V100-SXM2-32GB", 1);
    let two = StageRoofline::of("Tesla V100-SXM2-32GB", 2);
    assert_eq!(two.memory_gb_s, 2.0 * one.memory_gb_s);
    assert_eq!(two.link_gb_s, 137.0);
    assert_eq!(StageRoofline::of("CPU", 1), StageRoofline::default());
    assert_eq!(share(447.0, 894.0), Some(0.5));
    assert_eq!(share(1.0, 0.0), None);
}
