use super::*;
use crate::model::core::StepPart;

const MIB: usize = 1 << 20;

/// Compute bytes growing with the context, as the gather path's do.
fn linear(base: usize, per_token: usize) -> impl Fn(usize) -> usize {
    move |ctx| base + per_token * ctx
}

/// Runs `fit_context` with a recorded measurement; checks what every fit must satisfy.
fn fit(
    asked: usize,
    spec_k: usize,
    max_seqs: usize,
    block: usize,
    free: usize,
    compute: impl Fn(usize) -> usize,
) -> Result<(Fitted, Vec<usize>), FitError<()>> {
    let mut calls = Vec::new();
    let room = |c: usize| free.saturating_sub(c);
    let f = fit_context(asked, spec_k, max_seqs, block, room, |ctx| {
        calls.push(ctx);
        Ok(compute(ctx))
    })?;
    // Measured at the very context returned, never above the asked one.
    assert!(calls.contains(&f.ctx), "{} not in {calls:?}", f.ctx);
    assert!(f.ctx <= asked);
    assert_eq!(f.compute, compute(f.ctx));
    // The blocks hold the context and the drafts, and fit next to what was measured.
    assert!(f.n_blocks * KV_BLOCK >= f.ctx + spec_k);
    assert!(f.n_blocks * block <= room(f.compute));
    assert!(f.n_blocks <= max_seqs * (f.ctx + spec_k).div_ceil(KV_BLOCK));
    Ok((f, calls))
}

#[test]
fn a_context_that_fits_is_measured_once() {
    let (f, calls) = fit(8192, 3, 4, MIB, 4096 * MIB, linear(512 * MIB, 1024)).unwrap();
    assert_eq!(calls, vec![8192]);
    assert_eq!(f.ctx, 8192);
    // Room for every sequence at the full context.
    assert_eq!(f.n_blocks, 4 * (8192 + 3usize).div_ceil(KV_BLOCK));
}

/// At 32K the buffers leave room for less than 32K: the lowered context is measured itself
/// (the result must never rest on a measurement at another context) before it is returned.
#[test]
fn a_lowered_context_is_measured_again() {
    let compute = linear(1024 * MIB, 64 * 1024);
    let (f, calls) = fit(
        32768,
        3,
        1,
        MIB,
        1024 * MIB + 2048 * MIB + 400 * MIB,
        compute,
    )
    .unwrap();
    assert!(calls.len() >= 2, "{calls:?}");
    assert_eq!(calls[0], 32768);
    assert!(f.ctx < 32768);
}

/// Buffers that do not depend on the context: the first lowering is the answer.
#[test]
fn constant_buffers_lower_once() {
    let block = 2 * MIB;
    let (f, calls) = fit(32768, 0, 1, block, 1000 * MIB, |_| 200 * MIB).unwrap();
    // 800 MiB / 2 MiB = 400 blocks = 25600 tokens.
    assert_eq!(f.ctx, 400 * KV_BLOCK);
    assert_eq!(f.n_blocks, 400);
    assert_eq!(calls[..2], [32768, 25600]);
    // Bisection between 25600 and 32768 finds nothing larger.
    assert!(
        calls[2..].iter().all(|&c| c > 25600 && c < 32768),
        "{calls:?}"
    );
}

/// Buffers that do not shrink with the context (here they grow as it drops): the search
/// still ends within a few measurements.
#[test]
fn steep_buffers_converge() {
    let block = MIB;
    // Room for (free - compute) / 1 MiB blocks; compute steps down slowly.
    let compute = |ctx: usize| 1000 * MIB - ctx * 1024;
    let (f, calls) = fit(65536, 1, 2, block, 1100 * MIB, compute).unwrap();
    assert!(calls.len() <= 12, "{calls:?}");
    assert!(f.ctx >= 2 * KV_BLOCK);
}

/// Buffers larger than the free memory at the asked context (a 4-sequence gather at 32K on
/// two V100s): halved until they fit, then narrowed back up by bisection.
#[test]
fn oversized_buffers_halve_then_bisect() {
    // 4 GiB of free memory, buffers of 1 GiB + 128 KiB per token, 64 KiB per token of cache.
    let compute = linear(1024 * MIB, 128 << 10);
    let block = 64 * (64 << 10);
    let (f, calls) = fit(32768, 3, 1, block, 4096 * MIB, compute).unwrap();
    assert_eq!(calls[..2], [32768, 16384]);
    // The exact limit: 1 GiB + 192 KiB x ctx <= 4 GiB.
    let limit = 3 * 1024 * MIB / (192 << 10);
    assert!(
        f.ctx <= limit && f.ctx + 1024 > limit,
        "{f:?}, limit {limit}"
    );
}

#[test]
fn no_room_is_an_error() {
    let err = fit(8192, 3, 1, MIB, 600 * MIB, |_| 600 * MIB).unwrap_err();
    assert!(matches!(err, FitError::NoRoom { compute, .. } if compute == 600 * MIB));
}

#[test]
fn a_measurement_error_is_passed_on() {
    let r = fit_context(8192, 0, 1, MIB, |_| usize::MAX, |_| Err::<usize, _>("oom"));
    assert_eq!(r, Err(FitError::Measure("oom")));
}

/// Many shapes: the invariants of [`fit`] hold, and the context is the asked one whenever the
/// cache for it fits beside its own buffers.
#[test]
fn invariants_over_shapes() {
    for asked in [100, 4096, 8192 + 7, 32768, 131072] {
        for spec_k in [0, 3] {
            for max_seqs in [1, 4] {
                for free in [700, 2000, 9000] {
                    for per_token in [0, 4096, 65536] {
                        let compute = linear(500 * MIB, per_token);
                        let free = free * MIB;
                        let r = fit(asked, spec_k, max_seqs, MIB, free, &compute);
                        let need = (asked + spec_k).div_ceil(KV_BLOCK) * MIB;
                        let fits_asked = free.saturating_sub(compute(asked)) >= need;
                        match r {
                            Ok((f, _)) => assert_eq!(f.ctx == asked, fits_asked),
                            Err(FitError::NoRoom { .. }) => assert!(!fits_asked),
                            Err(FitError::Measure(())) => unreachable!(),
                        }
                    }
                }
            }
        }
    }
}

/// Keys a measured step gathers, as the executor counts them.
fn gathered(step: &[StepPart]) -> usize {
    step.iter()
        .map(|p| p.n_seqs * Executor::gather_span(p.n_past + p.n_tok))
        .sum()
}

/// The steps measured cover what the scheduler can run: several prompt chunks at once (one
/// per prefilling sequence, as many as the batch and the prefill slots take), beside the
/// other sequences' verify group, which comes first; every corner within the gather budget.
#[test]
fn corners_cover_several_prompt_chunks() {
    let s = Sched {
        max_seqs: 4,
        spec_k: 3,
        chunk: 512,
        max_batch: 4096,
        prefill_slots: 4,
    };
    let ctx = 25600;
    let c = corners(&s, ctx);
    let b = budget(ctx, 3);
    for step in c.mixed.iter().chain(&c.decode) {
        assert!(
            gathered(step) <= b,
            "{step:?} gathers {} > {b}",
            gathered(step)
        );
        // The decode or verify group first, then the prompt chunks.
        let first_prompt = step.iter().position(|p| p.n_tok == 512);
        if let Some(i) = first_prompt {
            assert!(step[i..].iter().all(|p| p.n_tok == 512), "{step:?}");
        }
    }
    // Four prompt chunks at once (4096 / 512 = 8, capped by the slots and sequences).
    assert!(
        c.mixed
            .iter()
            .any(|st| st.iter().filter(|p| p.n_tok == 512).count() == 4),
        "{:?}",
        c.mixed
    );
    // A prompt chunk alone at the end of the context, a full verify batch, a long one.
    assert!(c.mixed.contains(&vec![StepPart {
        n_seqs: 1,
        n_tok: 512,
        n_past: b - 512,
        verify: false
    }]));
    assert!(c.mixed.iter().any(|st| st
        == &[StepPart {
            n_seqs: 4,
            n_tok: 4,
            n_past: b / 4 / 256 * 256 - 4,
            verify: true
        }]));
    assert!(c.decode.iter().any(|st| st
        == &[StepPart {
            n_seqs: 1,
            n_tok: 1,
            n_past: b - 1,
            verify: false
        }]));
    // Two prompt chunks with two sequences verifying beside them.
    let two = Sched {
        max_batch: 1024,
        ..s
    };
    let c = corners(&two, ctx);
    assert!(c.mixed.iter().any(|st| st.len() == 3
        && st[0].n_seqs == 2
        && st[0].verify
        && st[1..].iter().all(|p| p.n_tok == 512)));
    // Without speculation: decode groups beside the chunks, no verify step.
    let plain = corners(&Sched { spec_k: 0, ..s }, ctx);
    assert!(plain.mixed.iter().flatten().all(|p| !p.verify));
    assert!(
        plain
            .mixed
            .iter()
            .any(|st| st.len() == 2 && st[0].n_tok == 1)
    );
}

/// In sequence slots nothing is gathered: decode steps and drafts are measured with every
/// sequence at the full context (past the sliding windows' saturation, the shape they reach),
/// never sharing it; the mixed steps keep the budget's corners (the room prompt copies get).
#[test]
fn slot_corners_put_every_decoding_sequence_at_the_full_context() {
    let s = Sched {
        max_seqs: 16,
        spec_k: 3,
        chunk: 512,
        max_batch: 4096,
        prefill_slots: 4,
    };
    for ctx in [16384, 32768] {
        let b = budget(ctx, 3);
        let c = slot_corners(&s, ctx);
        assert_eq!(
            c.decode,
            vec![vec![StepPart {
                n_seqs: 16,
                n_tok: 1,
                n_past: b - 1,
                verify: false
            }]]
        );
        assert_eq!(c.drafts, vec![(16, b)]);
        assert_eq!(c.mixed, corners(&s, ctx).mixed);
        // The shared corners split the budget: 1024 positions each at 16K, under a
        // 1024-token window plus a step.
        assert!(corners(&s, ctx).decode[0][0].n_past < b / 8);
    }
}
