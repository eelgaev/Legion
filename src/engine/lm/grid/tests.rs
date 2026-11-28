use super::grid_take;

/// The chunk ends of a `prompt`-token prompt in steps of `step` tokens on `grid`.
fn ends(prompt: usize, step: usize, grid: usize) -> Vec<usize> {
    let (mut at, mut out) = (0, Vec::new());
    while at < prompt {
        at += grid_take(step.min(prompt - at), grid);
        out.push(at);
    }
    out
}

#[test]
fn off_or_small_steps_keep_the_chunk() {
    assert_eq!(grid_take(8240, 0), 8240);
    assert_eq!(grid_take(4096, 4096), 4096);
    assert_eq!(grid_take(48, 4096), 48);
    assert_eq!(grid_take(4000, 4096), 4000);
}

#[test]
fn bigger_chunks_hold_whole_cells() {
    assert_eq!(grid_take(12288, 4096), 12288);
    assert_eq!(grid_take(8240, 4096), 8192);
    assert_eq!(grid_take(4100, 4096), 4096);
    // A room that is no multiple of the grid (decode rows took some of the step).
    assert_eq!(grid_take(12280, 4096), 8192);
}

#[test]
fn the_tail_runs_as_in_grid_sized_steps() {
    assert_eq!(ends(8240, 12288, 4096), [8192, 8240]);
    assert_eq!(ends(8240, 4096, 4096), [4096, 8192, 8240]);
    assert_eq!(ends(32738, 12288, 4096), [12288, 24576, 28672, 32738]);
    assert_eq!(ends(8192, 12288, 4096), [8192]);
    assert_eq!(ends(3000, 12288, 4096), [3000]);
}

#[test]
fn every_chunk_end_is_one_of_the_grid_sized_steps() {
    for prompt in (1..40_000).step_by(97) {
        let small = ends(prompt, 4096, 4096);
        for step in [8192, 12288, 16384, 10000] {
            let big = ends(prompt, step, 4096);
            assert!(
                big.iter().all(|e| small.contains(e)),
                "prompt {prompt} step {step}: {big:?} vs {small:?}"
            );
            // A part past the last whole cell runs alone, as in the grid-sized steps.
            let tail = prompt % 4096;
            if prompt > 4096 && tail > 0 {
                let n = big.len();
                assert_eq!(big[n - 1] - big[n - 2], tail, "prompt {prompt} step {step}");
            }
        }
    }
}
