//! A layer mixer split over the tensor-parallel ranks in decode and verify steps (#1176): every
//! rank used to run each hyper-connection mixer whole, the same work on the same values. Here
//! each rank computes its share of the up matmul's rows and of the gated mix, and an all-gather
//! of the mixed rows ([`crate::tp_net::seq_par::all_gather_rows`]) gives every rank the whole
//! block input:
//!
//! - up and mix: rank `r` of `R` takes the embedding rows `r · e / R..` of every stream (`hc`
//!   row ranges of the up matrix, one dense op, [`dense_mv`]), the gated mix of those rows
//!   ([`hc_mix16::hc_pre_part`]), then the gather of `[e / R, n]` a rank;
//! - down, with [`Mode::Both`]: rank `r` takes the rows `r · rank / R..` of the down matrix
//!   (its few row tiles spread over clusters, [`dense_mv::mul_mat_spread`]) and their
//!   activation, then the gather of `lo`.
//!
//! Every value is computed by the same kernel in the same order as the whole mixer's (dense_mv's
//! per-row sums do not depend on the rows beside them, the mix and the activation are per value)
//! and travels as it is, so every rank gets the bits of the mirrored mixer.
//!
//! On by default (up and mix only: the down matrix's tiles are latency bound, half of them take
//! as long); `LEGION_DEBUG_TP_HC_SPLIT=0` keeps the mixers mirrored, `=2` splits down as well.

use std::sync::OnceLock;

use super::{hc_chain, hc_mix16};
use crate::ggml::{Context, Tensor, types};
use crate::model::dense_mv;
use crate::tp_net::seq_par;

/// Which parts of a layer mixer the ranks split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mode {
    /// None: every rank runs the whole mixer.
    Off,
    /// The up matmul and the gated mix.
    Up,
    /// The down matmul too.
    Both,
}

/// The mode for a `LEGION_DEBUG_TP_HC_SPLIT` value: "0" [`Mode::Off`], "2" [`Mode::Both`],
/// anything else (or unset) [`Mode::Up`].
pub(super) fn mode_of(v: Option<&str>) -> Mode {
    match v.map(str::trim) {
        Some("0") => Mode::Off,
        Some("2") => Mode::Both,
        _ => Mode::Up,
    }
}

/// [`mode_of`] the environment, read once.
pub(super) fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| mode_of(std::env::var("LEGION_DEBUG_TP_HC_SPLIT").ok().as_deref()))
}

/// The communicator, this rank and the rank count of a split mixer.
pub(super) type Split = (i32, usize, usize);

/// Rows `first..first + m` of the matrix `w` (`[k, rows]`): `[k, m]`.
fn rows<'a>(ctx: &'a Context, w: Tensor<'a>, first: i64, m: i64) -> Tensor<'a> {
    let nb1 = w.nb()[1];
    ctx.view_2d(w, [w.ne()[0], m], nb1, first as usize * nb1)
}

/// `lo` of the mixer with the down matrix split ([`Mode::Both`]): `act(down @ xn)`, F32
/// `[rank, n]`, every rank's rows gathered; `None` where the split does not take it.
pub(super) fn down_lo<'a>(
    ctx: &'a Context,
    down: Tensor<'a>,
    up: Tensor<'a>,
    xn: Tensor<'a>,
    scale: f32,
    (comm, rank, ranks): Split,
) -> Option<Tensor<'a>> {
    let m = down.ne()[1];
    let piece = m / ranks as i64;
    if mode() != Mode::Both || ranks < 2 || m % ranks as i64 != 0 || down.ty() != types::BF16 {
        return None;
    }
    let part = rows(ctx, down, rank as i64 * piece, piece);
    if !dense_mv::takes_small(&[part], xn) || !dense_mv::fits(part, xn) {
        return None;
    }
    let lo = dense_mv::mul_mat_spread(ctx, part, xn);
    let lo = hc_chain::lo_act(ctx, lo, scale, hc_chain::on_gpu(up));
    Some(seq_par::all_gather_rows(ctx, comm, lo, m))
}

/// The mixed block input with the up matmul and the gated mix split: from `lo` (F32 `[rank,
/// n]`) and the normed streams `xn` (F32 `[e · hc, n]`), F32 `[e, n]`, every rank's rows
/// gathered; `None` where the split does not take it.
pub(super) fn up_mix<'a>(
    ctx: &'a Context,
    up: Tensor<'a>,
    lo: Tensor<'a>,
    xn: Tensor<'a>,
    hc: i64,
    scale: f32,
    (comm, rank, ranks): Split,
) -> Option<Tensor<'a>> {
    let rows_all = up.ne()[1];
    let n = lo.ne()[1];
    if mode() == Mode::Off
        || ranks < 2
        || !(1..=dense_mv::MAX_WEIGHTS as i64).contains(&hc)
        || rows_all % hc != 0
        || (rows_all / hc) % ranks as i64 != 0
        || up.ty() != types::BF16
        || !hc_chain::on_gpu(up)
        || !hc_mix16::pre_on()
        || xn.ty() != types::F32
        || !xn.is_contiguous()
        || xn.ne() != [rows_all, n, 1, 1]
        || !lo.is_contiguous()
    {
        return None;
    }
    let e = rows_all / hc;
    let eh = e / ranks as i64;
    let first = rank as i64 * eh;
    let parts: Vec<Tensor<'a>> = (0..hc).map(|c| rows(ctx, up, c * e + first, eh)).collect();
    if !dense_mv::takes_small(&parts, lo) || dense_mv::chain_count(&parts, lo) != Some(0) {
        return None;
    }
    let gates = dense_mv::mul_mat(ctx, &parts, lo);
    let mixed = hc_mix16::hc_pre_part(ctx, xn, &gates, first, scale);
    Some(seq_par::all_gather_rows(ctx, comm, mixed, e))
}

#[cfg(test)]
mod tests;
