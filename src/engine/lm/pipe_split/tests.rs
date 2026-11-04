use std::time::Duration;

use super::*;

#[test]
fn no_extra_work_keeps_the_even_split() {
    assert_eq!(mtp_split(48, 4, 0.0), vec![12, 24, 36]);
    // Less than half a layer's worth does not move a layer.
    assert_eq!(mtp_split(48, 4, 0.5), vec![12, 24, 36]);
    assert_eq!(mtp_split(48, 2, 0.0), vec![24]);
}

#[test]
fn the_last_stage_gives_up_layers_for_its_extra_work() {
    // An AC922's four V100s with MTP k=2 (#762): ~4 ms of drafts and commit at ~0.5 ms a layer.
    assert_eq!(mtp_split(48, 4, 8.0), vec![14, 28, 42]);
    // A tie keeps the larger last stage: 14/14/13/7 and 14/14/14/6 both take 14.
    assert_eq!(mtp_split(48, 4, 7.0), vec![14, 28, 41]);
    assert_eq!(mtp_split(48, 4, 4.0), vec![13, 26, 39]);
    assert_eq!(mtp_split(48, 2, 4.0), vec![26]);
}

#[test]
fn the_split_stays_valid() {
    for n_layer in 2..64 {
        for n in 2..=n_layer.min(8) {
            for e in [-3.0, 0.0, 1.5, 7.0, 40.0, 1e9, f64::NAN] {
                let cuts = mtp_split(n_layer, n, e);
                assert_eq!(cuts.len(), n - 1, "{n_layer} layers, {n} stages, {e}");
                let mut prev = 0;
                for &c in &cuts {
                    assert!(c > prev, "{cuts:?}");
                    prev = c;
                }
                assert!(prev < n_layer, "{cuts:?}: the last stage keeps a layer");
            }
        }
    }
    assert!(mtp_split(3, 4, 1.0).is_empty());
    assert!(mtp_split(48, 1, 1.0).is_empty());
}

fn ms(x: f64) -> Option<Duration> {
    Some(Duration::from_secs_f64(x / 1e3))
}

#[test]
fn the_probe_measures_a_layer_and_the_last_stages_extra() {
    let mut p = Probe::new(vec![12, 12, 12, 12]);
    // Stages 0-2: 6 ms for 12 layers; the last 7 ms (the head) plus 3 ms of drafts.
    for _ in 0..SKIP_ROUNDS + MIN_ROUNDS - 1 {
        p.record(
            &[ms(6.0), ms(6.0), ms(6.0), ms(7.0)],
            true,
            Duration::from_millis(3),
        );
    }
    assert!(p.hint().is_none(), "too few rounds");
    // Rounds without a piece on every stage, or not handed out early, do not count.
    p.record(&[ms(6.0), None, ms(6.0), ms(7.0)], true, Duration::ZERO);
    p.record(&[ms(6.0), ms(6.0), ms(6.0), ms(7.0)], false, Duration::ZERO);
    p.record(&[ms(6.0), ms(6.0)], true, Duration::ZERO);
    assert_eq!(p.rounds(), MIN_ROUNDS - 1);
    p.record(
        &[ms(6.0), ms(6.0), ms(6.0), ms(7.0)],
        true,
        Duration::from_millis(3),
    );
    let h = p.hint().expect("enough rounds");
    assert_eq!(h.rounds, MIN_ROUNDS);
    assert!((h.layer_ms - 0.5).abs() < 1e-9, "{h:?}");
    assert!((h.extra_ms - 4.0).abs() < 1e-9, "{h:?}");
    assert!((h.extra_layers - 8.0).abs() < 1e-6, "{h:?}");
    assert_eq!(h.current, vec![12, 24, 36]);
    assert_eq!(h.cuts, vec![14, 28, 42]);
}

#[test]
fn a_balanced_round_suggests_the_split_it_measured() {
    let mut p = Probe::new(vec![12, 12, 12, 12]);
    for _ in 0..SKIP_ROUNDS + MIN_ROUNDS {
        p.record(
            &[ms(6.0), ms(6.0), ms(6.0), ms(6.0)],
            true,
            Duration::from_micros(100),
        );
    }
    let h = p.hint().expect("enough rounds");
    assert_eq!(h.cuts, h.current);
}
