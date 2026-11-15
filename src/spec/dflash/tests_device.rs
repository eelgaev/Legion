//! The take-in from rows left where the target's step wrote them (#1181): the drafts of
//! [`ModelDrafter::step_device`] are those of [`ModelDrafter::step`] with the same rows on the
//! host, bit for bit.

use super::tests::{Fixture, hidden, sampling, tokens_to};
use super::tests_v2::v2_draft;
use super::tiny::TinyDraft;
use super::*;
use crate::ggml::{Backend, Context, RowsAt, types};

/// A request: `(seq, tokens, start, ran, max, salt)`, as `tests::step` takes them.
type Req<'a> = (u64, &'a [u32], usize, usize, usize, u64);

/// A step's captured rows as the target's graph holds them: request `i`'s from row `first[i]`
/// of every layer, a NaN row after each (a padded group's rows, never read).
struct StepRows {
    _ctx: Context,
    layers: Vec<RowsAt>,
    first: Vec<usize>,
    /// `[request][layer]`: the same rows on the host.
    host: Vec<Vec<Vec<f32>>>,
}

fn step_rows(backend: &Backend, n_layers: usize, reqs: &[Req<'_>]) -> StepRows {
    let width = 64;
    let mut first = Vec::new();
    let mut n_rows = 0;
    for r in reqs {
        first.push(n_rows);
        n_rows += r.3 + 1;
    }
    let host: Vec<Vec<Vec<f32>>> = reqs
        .iter()
        .map(|&(_, _, start, ran, _, salt)| {
            (0..n_layers)
                .map(|li| {
                    (start..start + ran)
                        .flat_map(|p| hidden(p, li, salt))
                        .collect()
                })
                .collect()
        })
        .collect();
    let ctx = Context::new(n_layers, 0);
    let layers = {
        let ts: Vec<_> = (0..n_layers)
            .map(|_| ctx.new_tensor(types::F32, &[width as i64, n_rows as i64]))
            .collect();
        ctx.alloc_tensors(backend.buffer_type()).unwrap();
        ts.iter()
            .enumerate()
            .map(|(li, t)| {
                let mut v = vec![f32::NAN; width * n_rows];
                for (h, &f) in host.iter().zip(&first) {
                    v[f * width..f * width + h[li].len()].copy_from_slice(&h[li]);
                }
                t.set_f32(&v);
                t.rows_at().expect("an allocated F32 matrix")
            })
            .collect()
    };
    StepRows {
        _ctx: ctx,
        layers,
        first,
        host,
    }
}

/// One step of `reqs`, its rows handed over on the host or where they are.
fn run(d: &mut DflashDrafter, reqs: &[Req<'_>], in_place: bool) -> Vec<Draft> {
    let backend = Backend::cpu().unwrap();
    let rows = step_rows(&backend, d.hidden_layers().len(), reqs);
    let mut outs = vec![Draft::default(); reqs.len()];
    let mut batch: Vec<DraftRequest<'_>> = reqs
        .iter()
        .zip(&rows.host)
        .zip(outs.iter_mut())
        .map(
            |((&(seq, tokens, start, ran, max, _), h), out)| DraftRequest {
                seq,
                tokens,
                start,
                ran,
                hidden: if in_place {
                    Vec::new()
                } else {
                    h.iter().map(Vec::as_slice).collect()
                },
                max,
                out,
            },
        )
        .collect();
    if in_place {
        let at = DeviceHidden {
            layers: &rows.layers,
            first: &rows.first,
        };
        d.step_device(&mut batch, &at).unwrap();
    } else {
        d.step(&mut batch).unwrap();
    }
    drop(batch);
    outs
}

#[test]
fn drafts_from_rows_in_place_are_those_from_the_host() {
    // DFlash 1 (one take-in per sequence) and 2 (one take-in graph for the batch, with and
    // without batched attention), greedy and sampled: prompts together with one that only
    // takes in, a decode step's few rows, then one sequence alone.
    let fx = Fixture::new();
    let (a, b, e) = (tokens_to(20), tokens_to(12), tokens_to(9));
    let steps: [&[Req<'_>]; 3] = [
        &[
            (1, &a[..13], 0, 13, 7, 0),
            (2, &b[..10], 0, 10, 7, 1),
            (3, &e, 0, 9, 0, 2),
        ],
        &[(1, &a[..16], 13, 3, 7, 0), (2, &b, 10, 2, 5, 1)],
        &[(1, &a, 16, 4, 6, 0)],
    ];
    let batched = DflashOptions {
        batched_attn: true,
        ..sampling(64)
    };
    for (name, c) in [("v1", TinyDraft::default()), ("v2", v2_draft())] {
        let path = fx.gguf(&format!("{name}.gguf"), &c);
        for opts in [DflashOptions::default(), sampling(64), batched.clone()] {
            let info = TargetInfo {
                max_seqs: 4,
                ..fx.info(&path)
            };
            let mut host = DflashDrafter::load(&info, opts.clone()).unwrap();
            let mut in_place = DflashDrafter::load(&info, opts).unwrap();
            assert_eq!(in_place.reads_device_rows(), None, "a CPU drafter");
            for (k, reqs) in steps.iter().enumerate() {
                let want = run(&mut host, reqs, false);
                let got = run(&mut in_place, reqs, true);
                assert_eq!(got, want, "{name}, step {k}");
                assert!(want.iter().any(|d| !d.tokens.is_empty()));
            }
        }
    }
}

#[test]
fn rows_in_place_must_cover_every_request() {
    let fx = Fixture::new();
    let path = fx.gguf("d.gguf", &v2_draft());
    let mut d = fx.load(&path, DflashOptions::default());
    let a = tokens_to(8);
    let backend = Backend::cpu().unwrap();
    let rows = step_rows(&backend, d.hidden_layers().len(), &[(1, &a, 0, 8, 4, 0)]);
    let mut out = Draft::default();
    let mut batch = [DraftRequest {
        seq: 1,
        tokens: &a,
        start: 0,
        ran: 8,
        hidden: Vec::new(),
        max: 4,
        out: &mut out,
    }];
    // A first row too far on for the request's 8 rows (9 in the step).
    let late = DeviceHidden {
        layers: &rows.layers,
        first: &[2],
    };
    assert!(d.step_device(&mut batch, &late).is_err());
    // A first row per request.
    let none = DeviceHidden {
        layers: &rows.layers,
        first: &[],
    };
    assert!(d.step_device(&mut batch, &none).is_err());
    let at = DeviceHidden {
        layers: &rows.layers,
        first: &rows.first,
    };
    d.step_device(&mut batch, &at).unwrap();
}
