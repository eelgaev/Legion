use super::{FOLD_MIN_TOKENS, grid_take, grid_take_folded};

/// The chunk ends of a `prompt`-token prompt in steps of `step` tokens on `grid`, folding last
/// parts of `fold` tokens or more.
fn ends_folded(prompt: usize, step: usize, grid: usize, fold: usize) -> Vec<usize> {
    let (mut at, mut out) = (0, Vec::new());
    while at < prompt {
        let len = step.min(prompt - at);
        at += grid_take_folded(len, grid, fold, at + len == prompt);
        out.push(at);
    }
    out
}

/// [`ends_folded`] without the fold.
fn ends(prompt: usize, step: usize, grid: usize) -> Vec<usize> {
    ends_folded(prompt, step, grid, 0)
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

#[test]
fn a_long_last_part_folds_into_the_cells_before_it() {
    let fold = FOLD_MIN_TOKENS;
    assert_eq!(grid_take_folded(8162, 4096, fold, true), 8162);
    assert_eq!(grid_take_folded(6176, 4096, fold, true), 6176);
    assert_eq!(grid_take_folded(11300, 4096, fold, true), 11300);
    // Too short, not the prompt's end, or off: the cells alone.
    assert_eq!(grid_take_folded(8240, 4096, fold, true), 8192);
    assert_eq!(grid_take_folded(4096 + 2047, 4096, fold, true), 4096);
    assert_eq!(grid_take_folded(8162, 4096, fold, false), 4096);
    assert_eq!(grid_take_folded(8162, 4096, 0, true), 4096);
    assert_eq!(grid_take_folded(8162, 0, fold, true), 8162);
    assert_eq!(grid_take_folded(3000, 4096, fold, true), 3000);
    assert_eq!(ends_folded(32738, 12288, 4096, fold), [12288, 24576, 32738]);
    assert_eq!(ends_folded(8240, 12288, 4096, fold), [8192, 8240]);
    assert_eq!(ends_folded(11300, 12288, 4096, fold), [11300]);
    assert_eq!(ends_folded(16384 + 3000, 12288, 4096, fold), [12288, 19384]);
}

#[test]
fn folded_chunk_ends_stay_on_the_grid_sized_steps() {
    for prompt in (1..40_000).step_by(97) {
        let small = ends(prompt, 4096, 4096);
        for step in [8192, 12288, 16384, 10000] {
            let big = ends_folded(prompt, step, 4096, FOLD_MIN_TOKENS);
            assert!(
                big.iter().all(|e| small.contains(e)),
                "prompt {prompt} step {step}: {big:?} vs {small:?}"
            );
            // A short last part still runs alone; a long one shares the step before it.
            let tail = prompt % 4096;
            if prompt > 4096 && tail > 0 && tail < FOLD_MIN_TOKENS {
                let n = big.len();
                assert_eq!(big[n - 1] - big[n - 2], tail, "prompt {prompt} step {step}");
            }
            assert!(big.len() <= ends(prompt, step, 4096).len());
        }
    }
}
