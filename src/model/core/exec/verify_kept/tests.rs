use super::*;

fn group(n_tok: usize, verify: bool, n_pool: usize) -> GroupShape {
    GroupShape {
        slot: 0,
        n_seqs: 4,
        n_tok,
        n_kv: 0,
        table: 4,
        ring: 0,
        logits: vec![n_tok],
        last_only: false,
        verify,
        n_pool,
        n_new: 0,
        pick_all: false,
        kv_slots: Vec::new(),
        par: false,
        pair: false,
        img: 0,
        captures: Vec::new(),
    }
}

#[test]
fn verify_shapes_are_those_without_a_prompt_chunk() {
    assert!(is_verify(&[group(3, true, 64)]));
    assert!(is_verify(&[group(3, true, 64), group(1, false, 64)]));
    assert!(!is_verify(&[group(512, false, 64)]));
    assert!(!is_verify(&[group(512, false, 64), group(3, true, 64)]));
    // A decode step's shape is not one (it is kept with the decode graphs).
    assert!(!is_verify(&[group(1, false, 64)]));
}

#[test]
fn without_a_verify_budget_the_oldest_mixed_graph_goes() {
    assert_eq!(evict(&[], true, 3, 0), None);
    assert_eq!(evict(&[true, false], true, 3, 0), None);
    assert_eq!(evict(&[false, true, true], true, 3, 0), Some(0));
    assert_eq!(evict(&[true, true, false], false, 3, 0), Some(0));
    // While the expert cache is lent: one kept.
    assert_eq!(evict(&[true], false, 1, 0), Some(0));
}

#[test]
fn with_a_verify_budget_each_kind_evicts_its_own() {
    // Prompt graphs no longer evict verify ones, nor verify graphs prompt ones.
    let kept = [false, true, false, true, false];
    assert_eq!(evict(&kept, false, 3, 4), Some(0));
    assert_eq!(evict(&kept, true, 3, 4), None);
    assert_eq!(evict(&kept, true, 3, 2), Some(1));
    assert_eq!(evict(&[true, true], false, 3, 2), None);
    assert_eq!(evict(&[], true, 3, 2), None);
}

#[test]
fn a_cycle_of_verify_shapes_within_the_budget_is_built_once() {
    // A round of requests walking through 8 verify shapes, then prompt chunks, three times:
    // with 8 verify graphs kept, only the first round builds them.
    let rounds = |verify_kept: usize| {
        let mut kept: Vec<(bool, usize)> = Vec::new();
        let mut builds = 0;
        for _ in 0..3 {
            let steps = (0..8)
                .map(|p| (true, p))
                .chain((0..3).map(|c| (false, 100 + c)));
            for (verify, id) in steps {
                if let Some(i) = kept.iter().position(|&(_, k)| k == id) {
                    let g = kept.remove(i);
                    kept.push(g);
                    continue;
                }
                builds += 1;
                let kinds: Vec<bool> = kept.iter().map(|&(v, _)| v).collect();
                if let Some(i) = evict(&kinds, verify, 3, verify_kept) {
                    kept.remove(i);
                }
                kept.push((verify, id));
            }
        }
        builds
    };
    assert_eq!(rounds(0), 33);
    assert_eq!(rounds(8), 11);
}

#[test]
fn a_full_verify_budget_evicts_before_the_build() {
    // The kept ones and one more of the newest's size.
    assert!(at_budget(true, 1900, 149, 2048));
    assert!(!at_budget(true, 1900, 148, 2048));
    assert!(!at_budget(true, 0, 0, 2048));
    // No budget, or a prompt graph: the count decides alone.
    assert!(!at_budget(true, 1 << 40, 1 << 30, 0));
    assert!(!at_budget(false, 4096, 100, 2048));
}

#[test]
fn the_oldest_verify_graphs_go_past_the_budget() {
    // (verify, bytes), oldest first.
    let kept = [
        (true, 100),
        (false, 900),
        (true, 100),
        (true, 100),
        (true, 300),
    ];
    // 600 verify bytes: within 600, nothing; prompt graphs do not count.
    assert!(over_budget(&kept, 600).is_empty());
    assert!(over_budget(&kept, 0).is_empty());
    assert_eq!(over_budget(&kept, 500), vec![0]);
    assert_eq!(over_budget(&kept, 300), vec![0, 2, 3]);
    // The newest verify graph stays even alone over the budget.
    assert_eq!(over_budget(&kept, 100), vec![0, 2, 3]);
    assert_eq!(over_budget(&[(true, 500)], 100), Vec::<usize>::new());
}

#[test]
fn a_budget_bounds_a_growing_cycle_of_verify_shapes() {
    // Verify shapes cycling at C=16 (each graph 150 units) under a count of 32: the budget of
    // 1000 keeps at most 6 (the newest may pass it alone), the count alone all 12.
    let run = |budget: usize| {
        let mut kept: Vec<(bool, usize, usize)> = Vec::new();
        let mut most = 0;
        for _ in 0..3 {
            for id in 0..12 {
                if let Some(i) = kept.iter().position(|k| k.2 == id) {
                    let g = kept.remove(i);
                    kept.push(g);
                    continue;
                }
                let kinds: Vec<bool> = kept.iter().map(|k| k.0).collect();
                let bytes: usize = kept.iter().filter(|k| k.0).map(|k| k.1).sum();
                let next = kept.iter().rev().find(|k| k.0).map_or(0, |k| k.1);
                let full = at_budget(true, bytes, next, budget);
                if let Some(i) = evict(&kinds, true, 3, 32)
                    .or_else(|| full.then(|| kinds.iter().position(|&v| v)).flatten())
                {
                    kept.remove(i);
                }
                kept.push((true, 150, id));
                let sizes: Vec<(bool, usize)> = kept.iter().map(|k| (k.0, k.1)).collect();
                for i in over_budget(&sizes, budget).into_iter().rev() {
                    kept.remove(i);
                }
                most = most.max(kept.iter().map(|k| k.1).sum::<usize>());
            }
        }
        most
    };
    assert_eq!(run(0), 12 * 150);
    assert!(run(1000) <= 1000, "{}", run(1000));
}
