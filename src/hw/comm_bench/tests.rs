//! The comm benchmark's plumbing on CPU "GPUs" (threads exchanging through host memory).

use super::*;

#[test]
fn quantiles_pick_by_nearest_rank() {
    let v = [5.0, 1.0, 3.0, 2.0, 4.0];
    assert_eq!(quantile(&v, 0.0), 1.0);
    assert_eq!(quantile(&v, 0.5), 3.0);
    assert_eq!(quantile(&v, 1.0), 5.0);
    assert_eq!(quantile(&v, 0.9), 5.0);
    assert!(quantile(&[], 0.5).is_nan());
}

#[test]
fn every_shape_and_chain_gets_its_timed_runs() {
    let shapes = [Shape { e: 64, n: 1 }, Shape { e: 40, n: 3 }];
    let chains = [Chain::Reduce, Chain::Add];
    let r = exchange_bench(&Devices::Cpu(3), &shapes, &chains, 5, 1, 4).unwrap();
    assert_eq!(r.len(), 4);
    for (res, (s, c)) in r.iter().zip(
        shapes
            .iter()
            .flat_map(|s| chains.iter().map(move |c| (s, c))),
    ) {
        assert_eq!((res.shape, res.chain, res.len), (*s, *c, 5));
        assert_eq!(res.runs_us.len(), 4);
        assert!(res.runs_us.iter().all(|&t| t > 0.0));
        assert!(res.per_node_us(0.5) > 0.0);
    }
    assert_eq!(r[2].bytes(), 40 * 3 * 4);
}

#[test]
fn one_device_or_no_nodes_is_refused() {
    let s = [Shape { e: 8, n: 1 }];
    assert!(exchange_bench(&Devices::Cpu(1), &s, &[Chain::Reduce], 3, 0, 1).is_err());
    assert!(exchange_bench(&Devices::Cpu(2), &s, &[Chain::Reduce], 0, 0, 1).is_err());
}
