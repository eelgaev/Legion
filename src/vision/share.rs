//! How a GPU image encode shares the GPU with the language model's decode (#1195).
//!
//! At equal stream priority the GPU places a kernel's blocks only once every block of the
//! kernels queued before it is placed. An encode's GEMMs queue hundreds of 128x128 tiles, so
//! each decode kernel queued behind one waits for nearly all of it: with one block per tile,
//! other users kept about half their rate during the encode (C=4 on a DGX Spark). So the
//! encoder's kernels run a capped grid that loops over its tiles ([`Share::gemm_ctas`]), one
//! CTA per SM, all placed at once, and the language model's blocks find room beside them on
//! every SM: about three quarters of the rate kept, the worst gap 0.36 -> 0.25 s, for an encode
//! of 0.53 -> 0.72 s under that load (alone it takes as long as before). The tiles' math does
//! not depend on the grid: the rows are the same bits. (Raising the language model's stream
//! priority meanwhile measured no better.)
//!
//! The cap only pays when something else runs. An encode that starts while the engine has no
//! sequence decoding, prefilling or waiting runs uncapped ([`choose`]): alone, the cap made the
//! encode 6-10% slower for nothing. The decision is made once, at the encode's start (a launch
//! cannot be re-gridded once queued, and the whole encode is about half a second): a request
//! that arrives during an uncapped encode waits for the rest of it, as it did before the cap.
//!
//! `LEGION_DEBUG_VISION_SHARE` overrides the default, for A/B runs: a comma-separated list
//! used in turn, one entry per encode, whether or not anything else runs. An entry is `0` (one
//! block per tile, the grids as they were) or a CTA count per SM (`1`, or `1/2`).
//! `LEGION_DEBUG_VISION_IDLE_FULL=0` keeps the default share on an idle engine too.

#[cfg(test)]
mod tests;

/// One encode's share of the GPU: GEMM CTAs per SM, as a fraction `num / den`; `num == 0`: no
/// cap (one block per tile).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Share {
    pub num: u32,
    pub den: u32,
}

/// The default: one GEMM CTA per SM, the small kernels two blocks per SM.
pub const DEFAULT: Share = Share { num: 1, den: 1 };

impl Share {
    /// No cap: the encoder's grids of #1195 stage C.
    pub const FULL: Share = Share { num: 0, den: 1 };

    /// Most CTAs of a GEMM launch on a device of `sms` SMs (0: no cap), at least 1.
    pub fn gemm_ctas(&self, sms: usize) -> usize {
        if self.num == 0 {
            return 0;
        }
        (sms * self.num as usize / self.den.max(1) as usize).max(1)
    }

    /// Most blocks of a small kernel's launch (twice the GEMM's: they are light), 0: no cap.
    pub fn small_blocks(&self, sms: usize) -> usize {
        2 * self.gemm_ctas(sms)
    }
}

impl std::fmt::Display for Share {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.num, self.den) {
            (0, _) => write!(f, "0"),
            (n, 1) => write!(f, "{n}"),
            (n, d) => write!(f, "{n}/{d}"),
        }
    }
}

/// The shares `LEGION_DEBUG_VISION_SHARE=v` names, in turn; `None` if `v` does not parse.
pub fn parse(v: &str) -> Option<Vec<Share>> {
    let mut out = Vec::new();
    for item in v.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (num, den) = match item.split_once('/') {
            Some((n, d)) => (n.trim().parse().ok()?, d.trim().parse().ok()?),
            None => (item.parse().ok()?, 1),
        };
        if den == 0 {
            return None;
        }
        out.push(Share { num, den });
    }
    (!out.is_empty()).then_some(out)
}

/// One encode's caps: most blocks of a GEMM launch and of a small kernel's (0: one per work
/// item).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    pub gemm: usize,
    pub small: usize,
}

/// One encode's share: `explicit` (`LEGION_DEBUG_VISION_SHARE`'s turn) if set; else none
/// ([`Share::FULL`]) when the engine runs nothing else (`alone`) and the idle rule is on
/// (`idle_full`); else [`DEFAULT`].
pub fn choose(explicit: Option<Share>, idle_full: bool, alone: bool) -> Share {
    match explicit {
        Some(share) => share,
        None if idle_full && alone => Share::FULL,
        None => DEFAULT,
    }
}

/// The shares a GPU tower encodes with, in turn, on a device of `sms` SMs.
pub struct Turns {
    sms: usize,
    /// `LEGION_DEBUG_VISION_SHARE`'s list (empty: [`choose`] decides by what else runs).
    shares: Vec<Share>,
    /// [`choose`]'s idle rule (`LEGION_DEBUG_VISION_IDLE_FULL=0` turns it off).
    idle_full: bool,
    encodes: usize,
}

impl Turns {
    /// [`from_env`]'s shares on a device of `sms` SMs.
    pub fn new(sms: usize) -> Self {
        Self {
            sms: sms.max(1),
            shares: from_env(),
            idle_full: std::env::var_os("LEGION_DEBUG_VISION_IDLE_FULL").is_none_or(|v| v != "0"),
            encodes: 0,
        }
    }

    /// Encodes with `share` from now on, whatever else runs (tests).
    #[cfg(test)]
    pub fn set(&mut self, share: Share) {
        self.shares = vec![share];
    }

    /// The next encode's caps; `alone`: the engine runs nothing else meanwhile.
    pub fn next_caps(&mut self, alone: bool) -> Caps {
        let explicit =
            (!self.shares.is_empty()).then(|| self.shares[self.encodes % self.shares.len()]);
        self.encodes += 1;
        let share = choose(explicit, self.idle_full, alone);
        if explicit.is_some() {
            tracing::info!(share = %share, "vision: GPU encode share (LEGION_DEBUG_VISION_SHARE)");
        } else {
            tracing::debug!(share = %share, alone, "vision: GPU encode share");
        }
        Caps {
            gemm: share.gemm_ctas(self.sms),
            small: share.small_blocks(self.sms),
        }
    }
}

/// The shares `LEGION_DEBUG_VISION_SHARE` sets, in turn; empty when unset (or not a list).
pub fn from_env() -> Vec<Share> {
    match std::env::var("LEGION_DEBUG_VISION_SHARE") {
        Ok(v) => parse(&v).unwrap_or_else(|| {
            tracing::warn!("LEGION_DEBUG_VISION_SHARE={v}: not a share list; the default runs");
            Vec::new()
        }),
        Err(_) => Vec::new(),
    }
}
