use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Barrier, Mutex};

use super::{
    by_device, move_runs, per_device_thread, pick_cover, pick_parts, piece_groups, plan_staging,
    refill_per_device, refill_until, run_rows, runs_of, stage_width, stages_all, stages_entry,
    take_device,
};

#[test]
fn refill_runs_every_layer_in_order_unless_stopped() {
    let stop = AtomicBool::new(false);
    let mut seen = Vec::new();
    let done = refill_until(vec![(3, 'a'), (5, 'b'), (7, 'c')], &stop, |w| {
        seen.push(w);
        if w == 'b' {
            Err("no room".into())
        } else {
            Ok(())
        }
    });
    assert_eq!(seen, ['a', 'b', 'c']);
    assert_eq!(
        done,
        vec![(3, Ok(())), (5, Err("no room".to_string())), (7, Ok(()))]
    );
}

/// A lend during the refill (#447) stops it at the next layer: the layer under way finishes,
/// the rest are not touched and not reported, so they keep their held experts.
#[test]
fn a_stopped_refill_finishes_its_layer_and_skips_the_rest() {
    let stop = AtomicBool::new(false);
    let mut seen = Vec::new();
    let done = refill_until(vec![(0, 0), (1, 1), (2, 2), (3, 3)], &stop, |w| {
        seen.push(w);
        if w == 1 {
            stop.store(true, Ordering::Relaxed);
        }
        Ok(())
    });
    assert_eq!(seen, [0, 1]);
    assert_eq!(done, vec![(0, Ok(())), (1, Ok(()))]);
}

#[test]
fn a_refill_stopped_before_it_starts_does_nothing() {
    let stop = AtomicBool::new(true);
    let done = refill_until(vec![(0, ()), (1, ())], &stop, |_| panic!("ran a layer"));
    assert!(done.is_empty());
}

#[test]
fn staging_footprint_grows_with_the_ring() {
    use super::stage_footprint;
    let mib = 1usize << 20;
    let bytes = [10 * mib, 10 * mib, 12 * mib];
    // Each buffer is rounded up by a 2 MiB granule, one buffer per kind and ring position.
    let one = (32 + 3 * 2) * mib as u64;
    assert_eq!(stage_footprint(bytes, 2), 2 * one);
    assert_eq!(stage_footprint(bytes, 3), 3 * one);
    assert_eq!(stage_footprint(bytes, 8), 8 * one);
}

#[test]
fn layers_group_by_device_in_order() {
    let work = vec![(0, (2, 'a')), (1, (3, 'b')), (2, (2, 'c')), (5, (3, 'd'))];
    assert_eq!(
        by_device(work),
        vec![(2, vec![(0, 'a'), (2, 'c')]), (3, vec![(1, 'b'), (5, 'd')])]
    );
}

/// #450: every device refills at once, each on a thread of its own (the barrier holds each
/// device's first layer until the other's has started), and the results come in device order.
#[test]
fn devices_refill_at_once_in_order() {
    let stop = AtomicBool::new(false);
    let met = Barrier::new(2);
    let groups = vec![(0, vec![(0, 'a'), (1, 'b')]), (1, vec![(2, 'c'), (3, 'd')])];
    let done = refill_per_device(groups, |device, work| {
        let mut first = true;
        refill_until(work, &stop, |w| {
            if std::mem::take(&mut first) {
                met.wait();
            }
            if w == 'd' {
                Err(format!("no room on {device}"))
            } else {
                Ok(())
            }
        })
    });
    assert_eq!(
        done,
        vec![
            (0, Ok(())),
            (1, Ok(())),
            (2, Ok(())),
            (3, Err("no room on 1".to_string()))
        ]
    );
}

/// A stop seen by one device's thread stops the others at their next layer too (#447).
#[test]
fn a_stop_reaches_every_device() {
    let stop = AtomicBool::new(false);
    let met = Barrier::new(2);
    let groups = vec![(0, vec![(0, 0), (1, 1)]), (1, vec![(2, 2), (3, 3)])];
    let done = refill_per_device(groups, |_, work| {
        refill_until(work, &stop, |w| {
            if w % 2 == 0 {
                // Both first layers are under way when the stop comes.
                met.wait();
                stop.store(true, Ordering::Relaxed);
                met.wait();
            }
            Ok(())
        })
    });
    assert_eq!(done, vec![(0, Ok(())), (2, Ok(()))]);
}

#[test]
fn a_panicked_device_reports_its_layers_failed() {
    let groups = vec![(0, vec![(0, ())]), (1, vec![(4, ()), (6, ())])];
    let done = refill_per_device(groups, |device, work| {
        assert_eq!(device, 0, "device 1 fails");
        work.into_iter().map(|(il, ())| (il, Ok(()))).collect()
    });
    let failed = Err("the slot thread panicked".to_string());
    assert_eq!(done, vec![(0, Ok(())), (4, failed.clone()), (6, failed)]);
}

#[test]
fn one_device_refills_on_the_callers_thread() {
    let caller = std::thread::current().id();
    let done = refill_per_device(vec![(7, vec![(1, ())])], |_, work| {
        assert_eq!(std::thread::current().id(), caller);
        work.into_iter().map(|(il, ())| (il, Ok(()))).collect()
    });
    assert_eq!(done, vec![(1, Ok(()))]);
}

#[test]
fn each_device_takes_its_own_staging_once() {
    let lent = Mutex::new(vec![(2, "two"), (0, "zero"), (3, "three")]);
    let done = refill_per_device(
        vec![(0, vec![(0, ())]), (3, vec![(9, ())]), (5, vec![(11, ())])],
        |device, work| {
            let mine = take_device(&lent, device);
            assert_eq!(
                mine,
                [(0, "zero"), (3, "three")]
                    .iter()
                    .find(|(d, _)| *d == device)
                    .map(|p| p.1)
            );
            assert_eq!(take_device(&lent, device), None, "taken once");
            work.into_iter().map(|(il, ())| (il, Ok(()))).collect()
        },
    );
    assert_eq!(done, vec![(0, Ok(())), (9, Ok(())), (11, Ok(()))]);
    // Left for the restore thread to free: the device with nothing to refill.
    assert_eq!(lent.into_inner().unwrap(), vec![(2, "two")]);
}

/// The lend (#439) runs every device at once: all four threads meet at a barrier, so this only
/// finishes when they run together; the results keep the order of the work.
#[test]
fn per_device_threads_run_together_in_order() {
    let met = Barrier::new(4);
    let caller = std::thread::current().id();
    let done = per_device_thread(
        "legion-test",
        vec![(3, 30), (1, 10), (0, 0), (2, 20)],
        |d, w| {
            assert_ne!(std::thread::current().id(), caller);
            assert_eq!(
                std::thread::current().name(),
                Some(format!("legion-test{d}").as_str())
            );
            met.wait();
            w + d
        },
    );
    assert_eq!(
        done,
        vec![(3, Ok(33)), (1, Ok(11)), (0, Ok(0)), (2, Ok(22))]
    );
}

#[test]
fn one_device_runs_on_the_callers_thread() {
    let caller = std::thread::current().id();
    let done = per_device_thread("legion-test", vec![(5, ())], |_, ()| {
        std::thread::current().id() == caller
    });
    assert_eq!(done, vec![(5, Ok(true))]);
    let none: Vec<(usize, Result<(), String>)> =
        per_device_thread("legion-test", Vec::<(usize, ())>::new(), |_, ()| ());
    assert!(none.is_empty());
}

#[test]
fn a_panicked_device_thread_reports_an_error_and_the_rest_finish() {
    let done = per_device_thread("legion-test", vec![(0, ()), (1, ()), (2, ())], |d, ()| {
        assert_ne!(d, 1, "device 1 fails");
        d
    });
    assert_eq!(
        done,
        vec![
            (0, Ok(0)),
            (1, Err("the legion-test thread panicked".to_string())),
            (2, Ok(2))
        ]
    );
}

const MIB: usize = 1 << 20;

/// A lend (#439) maps staging buffers over the lent slots' VRAM: the parts that fill a buffer
/// best without going over it (here 226 + 246 = 472 of 472), not the biggest first (300 would
/// leave 172 MiB to allocate).
#[test]
fn staging_parts_fill_the_buffer_best() {
    let sizes = [226 * MIB, 300 * MIB, 246 * MIB, 2 * MIB];
    assert_eq!(pick_parts(&sizes, 472 * MIB), vec![0, 2]);
    // Nothing over the need: 300 + 2 is the best under 303.
    assert_eq!(pick_parts(&sizes, 303 * MIB), vec![1, 3]);
    let total: usize = sizes.iter().sum();
    assert_eq!(pick_parts(&sizes, total + MIB), vec![0, 1, 2, 3]);
}

#[test]
fn staging_parts_never_go_over_the_need() {
    let sizes = [64 * MIB, 64 * MIB, 128 * MIB];
    assert!(pick_parts(&sizes, 32 * MIB).is_empty());
    assert!(pick_parts(&[], 32 * MIB).is_empty());
    assert!(pick_parts(&[0, 0], 32 * MIB).is_empty());
    let picked = pick_parts(&sizes, 200 * MIB);
    let sum: usize = picked.iter().map(|&i| sizes[i]).sum();
    assert_eq!(sum, 192 * MIB);
    // Every part at most once.
    assert_eq!(pick_parts(&[64 * MIB], 200 * MIB), vec![0]);
}

/// Odd sizes make the common unit tiny: past the exact search's reach, the largest parts that
/// still fit, still never over the need.
#[test]
fn staging_parts_fall_back_to_largest_first() {
    let sizes = [300 * MIB + 1, 200 * MIB + 7, 100 * MIB + 3];
    let need = 320 * MIB;
    assert_eq!(pick_parts(&sizes, need), vec![0]);
    let need = 520 * MIB;
    assert_eq!(pick_parts(&sizes, need), vec![0, 1]);
}

/// Kept staging buffers are made of lent slots only: the parts with the smallest total that
/// still covers the need (here 246 + 226 = 472 for 400, not 300 + 246).
#[test]
fn staging_cover_goes_over_the_need_least() {
    let sizes = [226 * MIB, 300 * MIB, 246 * MIB, 2 * MIB];
    assert_eq!(pick_cover(&sizes, 400 * MIB), vec![0, 2]);
    assert_eq!(pick_cover(&sizes, 472 * MIB), vec![0, 2]);
    assert_eq!(pick_cover(&sizes, 301 * MIB), vec![1, 3]);
    assert_eq!(pick_cover(&sizes, 2 * MIB), vec![3]);
    // Short of the need: all of them.
    let total: usize = sizes.iter().sum();
    assert_eq!(pick_cover(&sizes, total + MIB), vec![0, 1, 2, 3]);
    assert!(pick_cover(&[], 32 * MIB).is_empty());
}

#[test]
fn staging_plan_takes_the_biggest_buffers_first() {
    // Two buffers per kind: 40 MiB (kind 0), 100 MiB (kind 1), 20 MiB (kind 2).
    let bytes = [40 * MIB, 100 * MIB, 20 * MIB];
    let sizes = [60 * MIB, 50 * MIB, 50 * MIB, 40 * MIB, 20 * MIB, 800 * MIB];
    let plan = plan_staging(&sizes, bytes, 2, true);
    // Kind 1's buffers first (2, 3), then kind 0's (0, 1), then kind 2's (4, 5).
    let order: Vec<usize> = plan.iter().map(|(i, _)| *i).collect();
    assert_eq!(order, vec![2, 3, 0, 1, 4, 5]);
    // Each takes the smallest cover of what is left: 50 + 50, then 60 + 40, then the whole
    // 800 MiB part for a 40 MiB buffer (20 MiB does not cover it), then the 20 MiB part.
    let parts = |i: usize| -> Vec<usize> {
        let mut p = plan.iter().find(|(b, _)| *b == i).unwrap().1.clone();
        p.sort_unstable();
        p
    };
    assert_eq!(parts(2), vec![1, 2]);
    assert_eq!(parts(3), vec![0, 3]);
    assert_eq!(parts(0), vec![5]);
    assert_eq!(parts(1), vec![4]);
    // Nothing left for the last two: new VRAM.
    assert!(parts(4).is_empty() && parts(5).is_empty());
    // Every part once.
    let mut all: Vec<usize> = plan.iter().flat_map(|(_, p)| p.clone()).collect();
    all.sort_unstable();
    assert_eq!(all, vec![0, 1, 2, 3, 4, 5]);
    // Without `cover`: the parts that fit, the rest new.
    let plan = plan_staging(&sizes, bytes, 2, false);
    assert_eq!(plan[0], (2, vec![2, 1]));
}

#[test]
fn staging_cover_fallback_still_covers() {
    let sizes = [300 * MIB + 1, 200 * MIB + 7, 100 * MIB + 3];
    let picked = pick_cover(&sizes, 320 * MIB);
    let sum: usize = picked.iter().map(|&i| sizes[i]).sum();
    assert!(sum >= 320 * MIB);
    assert_eq!(picked, vec![0, 2]);
}

#[test]
fn stage_pieces_cover_the_misses_in_order() {
    assert_eq!(piece_groups(10, 4), vec![0..2, 2..5, 5..7, 7..10]);
    assert_eq!(piece_groups(3, 8), vec![0..1, 1..2, 2..3]);
    assert_eq!(piece_groups(5, 0), vec![0..5]);
    assert!(piece_groups(0, 4).is_empty());
    assert_eq!(
        runs_of(&[3, 4, 5, 9, 11, 12]),
        vec![(0, 3, 3), (3, 9, 1), (4, 11, 2)]
    );
}

#[test]
fn expert_parallel_copies_stage_as_their_layer() {
    // Layers 0..3 (3: an MTP head, never staged), then copies of 1, 2 and 3 on other GPUs.
    let real = [0, 1, 2, 3, 1, 2, 3];
    let staged: Vec<bool> = (0..real.len())
        .map(|il| stages_entry(Some(&real), il, 3, false))
        .collect();
    assert_eq!(staged, [true, true, true, false, true, true, false]);
    // Without expert parallelism, the entry is the layer.
    assert!(stages_entry(None, 2, 3, false));
    assert!(!stages_entry(None, 4, 3, false));
    assert!(stages_entry(None, 2, 3, true));
}

#[test]
fn pair_copies_lend_every_slot_to_prompts() {
    // `--pair-tp` (#681): a layer's prompts run on its own entry over every expert, so its
    // copies on the NVLink partner compute nothing then and never stage.
    let real = [0, 1, 2, 3, 1, 2, 3];
    let staged: Vec<bool> = (0..real.len())
        .map(|il| stages_entry(Some(&real), il, 3, true))
        .collect();
    assert_eq!(staged, [true, true, true, false, false, false, false]);
}

#[test]
fn staging_width_counts_every_expert_an_entry_computes() {
    // An owner of half of 512 experts that computes them all in prompts sizes its staging
    // buffers for all 512 (#681: sized for 256, they took the prompt's compute room).
    let half = 0..256u32;
    assert_eq!(stage_width(Some(&half), true, 512), 512);
    // GLM's parallel decode: owned experts only.
    assert_eq!(stage_width(Some(&half), false, 512), 256);
    assert_eq!(stage_width(None, false, 512), 512);
    assert_eq!(stage_width(None, true, 512), 512);
}

#[test]
fn rows_follow_the_layers_a_device_runs() {
    const NONE: usize = usize::MAX;
    // Without expert parallelism: each device's entries in their own order (entry 1 has no
    // experts, a dense layer).
    assert_eq!(run_rows(&[0, NONE, 0, 1, 1], None), [0, NONE, 1, 0, 1]);
    // Layers 0..3 owned by GPUs 0, 1, 0 (3: the MTP head, on GPU 1), then each decoder layer's
    // copy on the other GPU: entries 4, 5, 6 copy layers 0, 1, 2 onto GPUs 1, 0, 1. GPU 0 runs
    // layer 0 (entry 0), 1 (its copy, entry 5), 2 (entry 2); GPU 1 runs 0 (entry 4), 1
    // (entry 1), 2 (entry 6), then the MTP head (entry 3).
    let dev = [0, 1, 0, 1, 1, 0, 1];
    let real = [0, 1, 2, 3, 0, 1, 2];
    assert_eq!(run_rows(&dev, Some(&real)), [0, 1, 2, 3, 0, 1, 2]);
}

#[test]
fn only_a_layers_own_entry_stages_every_expert() {
    // Layers 0..3 (3: the MTP head), then copies of layers 0, 1, 2.
    let real = [0, 1, 2, 3, 0, 1, 2];
    let all: Vec<bool> = (0..real.len())
        .map(|il| stages_all(Some(&real), il, true))
        .collect();
    assert_eq!(all, [true, true, true, true, false, false, false]);
    // Off (GLM's parallel decode), or without expert parallelism: owned experts only.
    assert!((0..real.len()).all(|il| !stages_all(Some(&real), il, false)));
    assert!(!stages_all(None, 0, true));
}

#[test]
fn repack_moves_merge_into_runs() {
    // Slots 0-2 from host experts 4-6 (n_slots 8: ids 12-14), slot 3 from slot 9 (a tail
    // slot), slot 5 from expert 8, slots 6-7 from slots 10-11.
    let moves = [(0, 12), (1, 13), (2, 14), (3, 9), (5, 16), (6, 10), (7, 11)];
    assert_eq!(
        move_runs(&moves, 8),
        vec![(0, 12, 3), (3, 9, 1), (5, 16, 1), (6, 10, 2)]
    );
    // A run never crosses from the slots into the host experts.
    assert_eq!(move_runs(&[(0, 7), (1, 8)], 8), vec![(0, 7, 1), (1, 8, 1)]);
    assert!(move_runs(&[], 8).is_empty());
}

#[test]
fn settles_only_deterministic_pair_balance() {
    use crate::residency::{EpLayers, ResidencyConfig};
    // #1165: the cache waits for its own work only under --deterministic, and only where
    // the sums read which experts are cached (GLM's pair balance or pair split).
    let pair = ResidencyConfig {
        ep: Some(EpLayers::default()),
        ep_cached: true,
        ..Default::default()
    };
    assert!(!pair.settles());
    assert!(
        ResidencyConfig {
            deterministic: true,
            ..pair.clone()
        }
        .settles()
    );
    // --deterministic alone, or without the cached view: copies taken as they finish.
    let det = ResidencyConfig {
        deterministic: true,
        ..Default::default()
    };
    assert!(!det.settles());
    assert!(
        !ResidencyConfig {
            ep: Some(EpLayers::default()),
            ..det.clone()
        }
        .settles()
    );
    assert!(!ResidencyConfig::default().settles());
}
