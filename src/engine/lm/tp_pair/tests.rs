//! `--tp-per-gpu`'s checks and the ranks' configs: one GPU each, their link's ends taken once.

use super::*;

fn cfg(tp: usize) -> LmConfig {
    LmConfig {
        tp,
        tp_per_gpu: true,
        ..LmConfig::new(std::path::Path::new("m.gguf"), DeviceChoice::Cpu)
    }
}

fn cpu_sum() -> Arc<PairSum> {
    let cpu = crate::ggml::Backend::cpu().unwrap().buffer_type();
    Arc::new(PairSum::new([cpu, cpu], cpu, 64).unwrap())
}

#[test]
fn needs_tp_2_and_one_gpu_per_rank() {
    assert!(check(&cfg(2)).is_ok());
    for bad in [
        cfg(1),
        cfg(4),
        LmConfig { gpus: 2, ..cfg(2) },
        LmConfig {
            layer_split: vec![30],
            ..cfg(2)
        },
        LmConfig {
            pair_tp: true,
            ..cfg(2)
        },
        LmConfig {
            device: DeviceChoice::Cuda,
            attention: Attention::Gather,
            ..cfg(2)
        },
    ] {
        let e = check(&bad).unwrap_err().to_string();
        assert!(e.starts_with("--tp-per-gpu: "), "{e}");
    }
}

#[test]
fn each_rank_loads_one_gpu_of_the_pair() {
    let [r0, r1] = rank_configs(&cfg(2), cpu_sum(), 2);
    for (want, c) in [r0, r1].iter().enumerate() {
        assert_eq!(c.tp, 1);
        assert!(!c.tp_per_gpu, "a rank's load must not split again");
        let p = c.tp_pair.as_ref().unwrap();
        assert_eq!((p.rank, p.cuda), (want, 2 + want));
        // Rings for the sliding layers with gather attention too, as under the meta device.
        assert!(c.ring_gather);
    }
}

#[test]
fn each_rank_takes_its_own_end_once() {
    let [r0, r1] = rank_configs(&cfg(2), cpu_sum(), 0);
    let (r0, r1) = (r0.tp_pair.unwrap(), r1.tp_pair.unwrap());
    let rank1 = std::thread::spawn(|| ());
    r1.set_rank1(rank1);
    assert!(matches!(r0.take_end(), Some(End::Lead(_, Some(_)))));
    assert!(matches!(r1.take_end(), Some(End::Serve(_))));
    // A second load of the same rank finds nothing.
    assert!(r0.clone().take_end().is_none());
    assert!(r1.take_end().is_none());
    // Both ranks hold the same all-reduce.
    assert!(Arc::ptr_eq(&r0.sum, &r1.sum));
}
