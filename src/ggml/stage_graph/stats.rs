//! `legion serve --stage-graph-stats` (#858): how often the stage graphs run, how often they
//! are captured (each captured piece is one `cudaGraphInstantiate`) and the host time those
//! captures take, and how many kept graphs the executor builds (each new kept graph is a new
//! stage graph, captured again on its second run unless a graph with its words left captures,
//! counted as `captures_reused`, [`super::store`]). Logged every [`EVERY`] while anything was
//! captured or built, so a shape that keeps changing shows up as captures per run. With the
//! log on, each run's check (whether the graph's words match its last run, [`super::props_same`],
//! and the cache lookup) is timed too: the host time between the hand-off and the first launch.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// How often the counts are logged.
pub(crate) const EVERY: Duration = Duration::from_secs(10);

static ON: AtomicBool = AtomicBool::new(false);
static RUNS: AtomicU64 = AtomicU64::new(0);
static CAPTURES: AtomicU64 = AtomicU64::new(0);
static CAPTURE_NS: AtomicU64 = AtomicU64::new(0);
static CHECK_NS: AtomicU64 = AtomicU64::new(0);
static BUILDS: AtomicU64 = AtomicU64::new(0);
static REUSES: AtomicU64 = AtomicU64::new(0);
static PRECAPTURES: AtomicU64 = AtomicU64::new(0);
static UPDATES: AtomicU64 = AtomicU64::new(0);
static UPDATE_FAILS: AtomicU64 = AtomicU64::new(0);
static STEALS: AtomicU64 = AtomicU64::new(0);
static MTP_BUILDS: AtomicU64 = AtomicU64::new(0);
static REWORDS: AtomicU64 = AtomicU64::new(0);
static WHY: [AtomicU64; Why::COUNT] = [const { AtomicU64::new(0) }; Why::COUNT];
static SPLIT: [AtomicU64; Split::COUNT] = [const { AtomicU64::new(0) }; Split::COUNT];
static WINDOW: Mutex<Window> = Mutex::new(Window::new());

/// Turns the periodic log on or off (`--stage-graph-stats`). The counts run either way.
pub fn set_logging(on: bool) {
    ON.store(on, Relaxed);
}

/// Whether the log is on ([`set_logging`]): runs time their check only then.
pub(crate) fn logging() -> bool {
    ON.load(Relaxed)
}

/// The counts so far.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Totals {
    /// Stage-graph runs (one graph on one backend).
    pub runs: u64,
    /// Pieces captured and instantiated.
    pub captures: u64,
    /// Host time of those captures, recording and instantiation.
    pub capture_ns: u64,
    /// Host time of the runs' checks (words compare and cache lookup), while the log is on.
    pub check_ns: u64,
    /// Kept graphs built ([`crate::ggml::KeptGraph::new`]).
    pub builds: u64,
    /// Sets of captures a graph found kept for its new words ([`super::store`]) instead of
    /// capturing again.
    pub reuses: u64,
    /// Sets of captures made for a kept graph built ahead before its first run
    /// (`run::precapture`, #836), so that run found them (counted in `reuses` then).
    pub precaptures: u64,
    /// Of `captures`, those that updated a spare executable graph ([`super::spare`], #836)
    /// instead of instantiating.
    pub updates: u64,
    /// Updates of a spare that CUDA refused (the capture was instantiated then).
    pub update_fails: u64,
    /// Executable graphs taken over from a kept capture unlaunched for a while, to update
    /// ([`super::spare`]; counted in `updates` or `update_fails` too).
    pub steals: u64,
    /// Of `builds`, the MTP head's (its take-in and draft passes' kept graphs).
    pub mtp_builds: u64,
    /// Runs of a graph seen before whose words changed (not a new kept graph's first run).
    pub rewords: u64,
    /// Of `captures`, by why the run captured ([`Why`], indexed by `Why as usize`).
    pub why: [u64; Why::COUNT],
    /// Where the steps' graphs ran directly and were built ([`Split`], indexed by
    /// `Split as usize`).
    pub split: [u64; Split::COUNT],
}

/// What [`Totals::split`] counts: target (step) graphs apart from the MTP head's, so a capture
/// count can be put on the shapes that drive it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Runs of a target (step) kept graph whose units ran directly (no capture launched).
    DirectStep,
    /// The same for the MTP head's kept graphs.
    DirectMtp,
    /// Of [`Why::New`], the pieces of the MTP head's kept graphs.
    NewMtp,
    /// Host time building target kept graphs inside their step (before its first kernel), ns.
    BuildStepNs,
    /// Host time building target kept graphs ahead (under the running step, #836), ns.
    BuildAheadNs,
    /// Host time building the MTP head's kept graphs, ns.
    BuildMtpNs,
}

impl Split {
    pub const COUNT: usize = 6;
}

/// Why a run captured pieces of its graph ([`count_why`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// A kept graph's first captures: a graph built since its last run (or never run).
    New,
    /// A kept graph that ran before captured again, its words unchanged: its cache entry (unused
    /// for 10 s) or its set of captures (past the store's cap, or unused for 10 s) was dropped
    /// while the graph stayed kept.
    Lost,
    /// A graph whose words changed since its last run (for a kept graph: a new kept graph at a
    /// dropped one's address, which found the old one's entry).
    Changed,
    /// A graph that is not a kept graph.
    Unkept,
}

impl Why {
    pub const COUNT: usize = 4;
}

impl Totals {
    fn since(self, earlier: Totals) -> Totals {
        Totals {
            runs: self.runs - earlier.runs,
            captures: self.captures - earlier.captures,
            capture_ns: self.capture_ns - earlier.capture_ns,
            check_ns: self.check_ns - earlier.check_ns,
            builds: self.builds - earlier.builds,
            reuses: self.reuses - earlier.reuses,
            precaptures: self.precaptures - earlier.precaptures,
            updates: self.updates - earlier.updates,
            update_fails: self.update_fails - earlier.update_fails,
            steals: self.steals - earlier.steals,
            mtp_builds: self.mtp_builds - earlier.mtp_builds,
            rewords: self.rewords - earlier.rewords,
            why: std::array::from_fn(|i| self.why[i] - earlier.why[i]),
            split: std::array::from_fn(|i| self.split[i] - earlier.split[i]),
        }
    }
}

/// The counts so far.
pub fn totals() -> Totals {
    Totals {
        runs: RUNS.load(Relaxed),
        captures: CAPTURES.load(Relaxed),
        capture_ns: CAPTURE_NS.load(Relaxed),
        check_ns: CHECK_NS.load(Relaxed),
        builds: BUILDS.load(Relaxed),
        reuses: REUSES.load(Relaxed),
        precaptures: PRECAPTURES.load(Relaxed),
        updates: UPDATES.load(Relaxed),
        update_fails: UPDATE_FAILS.load(Relaxed),
        steals: STEALS.load(Relaxed),
        mtp_builds: MTP_BUILDS.load(Relaxed),
        rewords: REWORDS.load(Relaxed),
        why: std::array::from_fn(|i| WHY[i].load(Relaxed)),
        split: std::array::from_fn(|i| SPLIT[i].load(Relaxed)),
    }
}

/// Counts one stage-graph run, and logs the window when it is due.
pub(crate) fn count_run() {
    RUNS.fetch_add(1, Relaxed);
    if ON.load(Relaxed) {
        log_due(Instant::now());
    }
}

/// Counts one captured piece that took `took` on the host.
pub(crate) fn count_capture(took: Duration) {
    CAPTURES.fetch_add(1, Relaxed);
    CAPTURE_NS.fetch_add(took.as_nanos() as u64, Relaxed);
}

/// Counts `n` pieces one run captured, for `why`.
pub(crate) fn count_why(why: Why, n: u64) {
    if n > 0 {
        WHY[why as usize].fetch_add(n, Relaxed);
    }
}

/// Adds `n` to [`Split`] counter `k`.
pub fn count_split(k: Split, n: u64) {
    if n > 0 {
        SPLIT[k as usize].fetch_add(n, Relaxed);
    }
}

/// Adds one run's check time ([`logging`] only).
pub(crate) fn count_check(took: Duration) {
    CHECK_NS.fetch_add(took.as_nanos() as u64, Relaxed);
}

/// Counts one set of captures found by its words.
pub(crate) fn count_reuse() {
    REUSES.fetch_add(1, Relaxed);
}

/// Counts one set of captures made before its graph's first run (`run::precapture`, CUDA only).
pub(crate) fn count_precapture() {
    PRECAPTURES.fetch_add(1, Relaxed);
}

/// Counts one update of a spare executable graph ([`super::spare`], CUDA only): performed
/// (`ok`) or refused.
pub(crate) fn count_update(ok: bool) {
    if ok {
        UPDATES.fetch_add(1, Relaxed);
    } else {
        UPDATE_FAILS.fetch_add(1, Relaxed);
    }
}

/// Counts one executable graph taken over from a kept capture ([`super::spare`], CUDA only).
pub(crate) fn count_steal() {
    STEALS.fetch_add(1, Relaxed);
}

/// Counts one of the MTP head's kept graphs built (also counted by [`count_build`]).
pub fn count_mtp_build() {
    MTP_BUILDS.fetch_add(1, Relaxed);
}

/// Counts one run of a graph seen before whose words changed: a graph of `n` nodes, compared
/// in the shorter form when `nodes`, first different at `diff` ([`super::Diff`]); logged with
/// it while the log is on.
pub(crate) fn count_reword(n: i32, nodes: bool, diff: Option<super::Diff>) {
    REWORDS.fetch_add(1, Relaxed);
    if ON.load(Relaxed) {
        let (node, word, old, new) = diff.unwrap_or_default();
        tracing::info!(
            nodes = n,
            short_form = nodes,
            node,
            word,
            old = format!("{old:#x}"),
            new = format!("{new:#x}"),
            "stage graph words changed (--stage-graph-stats)"
        );
    }
}

/// What a build line says about a new kept graph ([`note_build`]).
#[derive(Debug, Default)]
pub struct BuildNote {
    /// Its shape, briefly.
    pub shape: String,
    /// How it differs from the nearest shape kept for its kind (the fields).
    pub nearest: String,
    /// Why no kept graph served it (new shape, evicted at the cap, dropped below it).
    pub why: &'static str,
    /// Graphs its kind kept before it was built, and its kind's cap (0: none).
    pub kept: usize,
    pub cap: usize,
}

/// Logs a kept graph built for `kind` (prompt, decode, verify, draft, take-in, piped) on stage
/// `stage` while the log is on (`note` is only made then).
pub fn note_build(kind: &str, stage: usize, note: impl FnOnce() -> BuildNote) {
    if ON.load(Relaxed) {
        let n = note();
        tracing::info!(
            kind,
            stage,
            shape = n.shape,
            nearest = n.nearest,
            why = n.why,
            kept = n.kept,
            cap = n.cap,
            "kept graph built (--stage-graph-stats)"
        );
    }
}

/// Shapes (by hash, per kind and stage) built while the log was on, for [`first_build`].
static BUILT: Mutex<Option<std::collections::HashSet<(String, usize, u64)>>> = Mutex::new(None);

/// At most this many shapes are remembered; past it the record starts again.
const BUILT_MAX: usize = 1 << 16;

/// Whether this is the first build of shape `hash` for `kind` on `stage` since the log was on
/// (and records it). For the build lines' why ([`BuildNote::why`]).
pub fn first_build(kind: &str, stage: usize, hash: u64) -> bool {
    let mut built = BUILT.lock().unwrap_or_else(|e| e.into_inner());
    let set = built.get_or_insert_with(Default::default);
    if set.len() >= BUILT_MAX {
        set.clear();
    }
    set.insert((kind.to_string(), stage, hash))
}

/// Counts one kept graph built.
pub(crate) fn count_build() {
    BUILDS.fetch_add(1, Relaxed);
}

/// What the log window keeps: when it started and the counts then.
pub(crate) struct Window {
    at: Option<Instant>,
    last: Totals,
}

impl Window {
    pub(crate) const fn new() -> Self {
        Self {
            at: None,
            last: Totals {
                runs: 0,
                captures: 0,
                capture_ns: 0,
                check_ns: 0,
                builds: 0,
                reuses: 0,
                precaptures: 0,
                updates: 0,
                update_fails: 0,
                steals: 0,
                mtp_builds: 0,
                rewords: 0,
                why: [0; Why::COUNT],
                split: [0; Split::COUNT],
            },
        }
    }

    /// The counts of the window ending `now` and its length, once [`EVERY`] has passed since it
    /// started, if anything was captured or built in it (a new window starts then either way).
    /// The first call only starts a window.
    pub(crate) fn due(&mut self, now: Instant, totals: Totals) -> Option<(Totals, Duration)> {
        let Some(at) = self.at else {
            self.at = Some(now);
            self.last = totals;
            return None;
        };
        let took = now.saturating_duration_since(at);
        if took < EVERY {
            return None;
        }
        let d = totals.since(self.last);
        self.at = Some(now);
        self.last = totals;
        (d.captures > 0 || d.builds > 0 || d.reuses > 0).then_some((d, took))
    }
}

fn log_due(now: Instant) {
    let Ok(mut w) = WINDOW.try_lock() else {
        return;
    };
    if let Some((d, took)) = w.due(now, totals()) {
        drop(w);
        let ms = d.capture_ns as f64 / 1e6;
        let sp = |k: Split| d.split[k as usize];
        let sp_ms = |k: Split| format!("{:.1}", sp(k) as f64 / 1e6);
        tracing::info!(
            seconds = format!("{:.1}", took.as_secs_f64()),
            runs = d.runs,
            captures = d.captures,
            capture_ms = format!("{ms:.1}"),
            capture_us_per_run = format!("{:.1}", ms * 1e3 / d.runs.max(1) as f64),
            check_us_per_run = format!("{:.1}", d.check_ns as f64 / 1e3 / d.runs.max(1) as f64),
            kept_graphs_built = d.builds,
            mtp_graphs_built = d.mtp_builds,
            words_changed = d.rewords,
            captures_reused = d.reuses,
            sets_captured_ahead = d.precaptures,
            spares_updated = d.updates,
            spare_updates_refused = d.update_fails,
            execs_taken_over = d.steals,
            captures_new = d.why[Why::New as usize],
            captures_lost = d.why[Why::Lost as usize],
            captures_changed = d.why[Why::Changed as usize],
            captures_unkept = d.why[Why::Unkept as usize],
            captures_new_mtp = sp(Split::NewMtp),
            direct_runs = sp(Split::DirectStep),
            direct_runs_mtp = sp(Split::DirectMtp),
            build_ms = sp_ms(Split::BuildStepNs),
            build_ahead_ms = sp_ms(Split::BuildAheadNs),
            mtp_build_ms = sp_ms(Split::BuildMtpNs),
            "stage graphs (--stage-graph-stats)"
        );
    }
}

#[cfg(test)]
mod tests;
