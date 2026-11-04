//! Micro-chunks that ramp down at the end of a long step (`--long-micro-taper`).
//!
//! A pipelined step ends once its last micro-chunk has passed every stage, and the first stage
//! is done with it about `stages - 1` micro-chunk times before the last one (a 123K prompt
//! over 4 V100s in 3072-token micro-chunks: ~1.1 s of a ~16 s step). Smaller micro-chunks at
//! the end pass the later stages sooner. The start needs no ramp: while the later stages wait
//! for their first micro-chunk, the first stage is already busy with the ones after it, and it
//! sets the step's pace (a ramp there only adds small micro-chunks, which stage about as many
//! experts as full ones: a 4-GPU profile, +0.5 s).

#[cfg(test)]
mod tests;

/// The sizes of the micro-chunks a step of `n` tokens is cut into, in order. Without a taper
/// (`taper` 0 or not under `micro`): full `micro` ones, then the rest, as before. With one:
/// micro-chunks of `micro` at most, of about equal size (the bigger ones first), then a ramp
/// down that halves from under `micro` to `taper`. A step too short for the ramp and a full
/// micro-chunk is cut as without a taper.
pub(super) fn micro_sizes(n: usize, micro: usize, taper: usize) -> Vec<usize> {
    if micro == 0 || n == 0 {
        return vec![n];
    }
    let plain = || {
        let mut v = vec![micro; n / micro];
        if !n.is_multiple_of(micro) {
            v.push(n % micro);
        }
        v
    };
    let ramp: Vec<usize> = std::iter::successors((taper > 0).then_some(taper), |t| Some(t * 2))
        .take_while(|&t| t < micro)
        .collect();
    let end = ramp.iter().sum::<usize>();
    if ramp.is_empty() || n < end + micro {
        return plain();
    }
    let mid = n - end;
    let k = mid.div_ceil(micro);
    let (base, extra) = (mid / k, mid % k);
    let mut v: Vec<usize> = (0..k).map(|i| base + usize::from(i < extra)).collect();
    v.extend(ramp.iter().rev());
    v
}
