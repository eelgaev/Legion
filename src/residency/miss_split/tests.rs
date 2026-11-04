use super::*;

fn seen(layer: usize, slots: usize, misses: u64) -> Option<LayerMisses> {
    Some(LayerMisses {
        layer,
        slots,
        hits: 0,
        misses,
    })
}

#[test]
fn equal_misses_keep_an_even_split() {
    let now = [10, 10, 10];
    let bytes = [4, 4, 4];
    let s = [seen(0, 10, 50), seen(1, 10, 50), seen(2, 10, 50)];
    let out = split_by_misses(&now, &bytes, &s, &[64; 3]);
    assert_eq!(out, vec![10, 10, 10]);
}

#[test]
fn layers_that_missed_more_get_more_slots_in_the_same_bytes() {
    let now = [10, 10, 10];
    let bytes = [4, 4, 4];
    let s = [seen(0, 10, 400), seen(1, 10, 100), seen(2, 10, 100)];
    let out = split_by_misses(&now, &bytes, &s, &[64; 3]);
    assert_eq!(out.iter().sum::<usize>(), 30);
    assert!(out[0] > 10 && out[1] < 10 && out[1] == out[2], "{out:?}");
    // a / n^2 equal at the optimum: 4x the misses, about 2x the slots.
    assert_eq!(out, vec![15, 8, 7]);
}

#[test]
fn the_byte_budget_holds_with_mixed_slot_sizes() {
    let now = [10, 10, 10];
    let bytes = [5, 4, 4];
    let s = [seen(0, 10, 100), seen(1, 10, 300), seen(2, 10, 100)];
    let out = split_by_misses(&now, &bytes, &s, &[64; 3]);
    let used: u64 = out.iter().zip(bytes).map(|(&n, b)| n as u64 * b).sum();
    assert!(used <= 130, "{out:?} uses {used}");
    assert!(used > 130 - 5, "{out:?} leaves bytes a slot could use");
    assert!(out[1] > out[2] && out[2] >= out[0], "{out:?}");
}

#[test]
fn unmeasured_and_other_device_layers_keep_their_counts() {
    let now = [10, 7, 10, 3];
    let bytes = [4, 4, 4, 0];
    let s = [seen(0, 10, 900), None, seen(2, 10, 100), seen(3, 3, 900)];
    let out = split_by_misses(&now, &bytes, &s, &[64; 4]);
    assert_eq!(out[1], 7);
    assert_eq!(out[3], 3);
    assert_eq!(out[0] + out[2], 20);
    assert!(out[0] > out[2]);
}

#[test]
fn a_layer_stops_at_its_cap_and_the_rest_goes_elsewhere() {
    let now = [10, 10];
    let bytes = [4, 4];
    let s = [seen(0, 10, 10_000), seen(1, 10, 10)];
    assert_eq!(split_by_misses(&now, &bytes, &s, &[12, 64]), vec![12, 8]);
}

#[test]
fn a_layer_that_never_missed_keeps_a_slot() {
    let now = [4, 4];
    let bytes = [1, 1];
    let s = [seen(0, 4, 0), seen(1, 4, 1_000_000)];
    let out = split_by_misses(&now, &bytes, &s, &[64; 2]);
    assert!(out[0] >= 1 && out.iter().sum::<usize>() == 8, "{out:?}");
}

#[test]
fn measured_split_resizes_with_the_lowest_owned_experts() {
    // Layers 0 and 1 on this device, layer 2 elsewhere; layer 1 owns experts 4..8.
    let alloc = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![]];
    let measured = MissStats {
        layers: vec![
            LayerMisses {
                layer: 0,
                slots: 4,
                hits: 0,
                misses: 900,
            },
            LayerMisses {
                layer: 1,
                slots: 4,
                hits: 0,
                misses: 100,
            },
        ],
    };
    let owned = [None, Some(4..8), None];
    let out = split_measured(alloc, &[2, 2, 0], &measured, &owned, 8);
    assert_eq!(out, vec![vec![0, 1, 2, 3, 4, 5], vec![4, 5], vec![]]);
}

#[test]
fn stats_round_trip_through_the_file() {
    let dir = std::env::temp_dir().join(format!("legion-miss-split-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("misses.json");
    let stats = MissStats {
        layers: vec![
            LayerMisses {
                layer: 0,
                slots: 318,
                hits: 242_253,
                misses: 34_517,
            },
            LayerMisses {
                layer: 48,
                slots: 0,
                hits: 0,
                misses: 0,
            },
        ],
    };
    stats.write(&path).unwrap();
    let back = MissStats::read(&path).unwrap();
    assert_eq!(back, stats);
    assert_eq!(back.get(0).map(|l| l.misses), Some(34_517));
    // No slots: nothing to scale from.
    assert_eq!(back.get(48), None);
    assert_eq!(back.get(7), None);
    std::fs::write(&path, "not json").unwrap();
    assert!(MissStats::read(&path).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
