//! Stop-string matching over streamed text.
//!
//! Text that could still be the start of a stop string is held back until it either completes
//! the stop string (and is dropped) or can no longer match (and is released).

use super::TokenLogprob;

#[derive(Debug, Clone)]
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
    /// When false, `push` neither matches nor holds back: text passes straight through.
    /// `min_tokens` defers stop-string checks this way until enough tokens exist.
    checking: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopResult {
    /// Text that is safe to emit (may be empty).
    Continue(String),
    /// A stop string matched: emit the text before it, then finish.
    Stopped { text: String, stop: String },
}

impl StopMatcher {
    pub fn new(stops: impl IntoIterator<Item = String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
            checking: true,
        }
    }

    /// Whether there is no stop string to match: no text is ever held back or stops.
    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// Turns matching on and off. While off, [`Self::push`] neither matches nor holds back:
    /// every piece of text, and anything held back before, passes straight through.
    pub fn set_checking(&mut self, checking: bool) {
        self.checking = checking;
    }

    pub fn push(&mut self, piece: &str) -> StopResult {
        self.held.push_str(piece);
        if !self.checking {
            return StopResult::Continue(self.flush());
        }
        // Earliest complete match wins.
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|at| (at, s)))
            .min_by_key(|(at, _)| *at);
        if let Some((at, stop)) = hit {
            let stop = stop.clone();
            let text = self.held[..at].to_string();
            self.held.clear();
            return StopResult::Stopped { text, stop };
        }
        // Hold back the longest suffix that is a proper prefix of some stop string.
        let keep = self
            .stops
            .iter()
            .map(|s| longest_suffix_prefix(&self.held, s))
            .max()
            .unwrap_or(0);
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        StopResult::Continue(out)
    }

    /// End of generation: release whatever was held back.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// Keeps token log probabilities in step with the text a [`StopMatcher`] releases.
///
/// The matcher may hold text back, or drop it for good when it is part of a stop string. Log
/// probabilities attach to tokens, not to text, so they wait here, with the byte span their
/// token covers in the stream fed to the matcher. When the matcher has released `n` bytes in
/// total, every token with at least one byte below `n` is handed out (just before the text
/// that carries it). A token wholly inside a matched stop string is never released, so the
/// entries always describe exactly the text the client receives. A token that straddles the end
/// of the released text is released, as it contributed some of it.
#[derive(Debug, Clone, Default)]
pub struct LogprobHold {
    pending: std::collections::VecDeque<(usize, usize, TokenLogprob)>,
    /// Bytes fed to the matcher so far.
    fed: usize,
    /// Bytes the matcher has released so far.
    released: usize,
}

impl LogprobHold {
    /// Record a token whose text (`piece`) is about to be fed to the matcher.
    pub fn push(&mut self, piece: &str, lp: TokenLogprob) {
        let start = self.fed;
        self.fed += piece.len();
        self.pending.push_back((start, self.fed, lp));
    }

    /// The matcher released `text`: returns the entries to send before it.
    pub fn release(&mut self, text: &str) -> Vec<TokenLogprob> {
        self.released += text.len();
        let mut out = Vec::new();
        while let Some((start, end, _)) = self.pending.front() {
            // A token with no text is released once the text reaches it.
            if *start < self.released || (start == end && *end <= self.released) {
                out.push(self.pending.pop_front().expect("front checked").2);
            } else {
                break;
            }
        }
        out
    }

    /// End of generation without a stop match: the matcher flushed `text`; release the rest.
    pub fn flush(&mut self, text: &str) -> Vec<TokenLogprob> {
        self.released += text.len();
        self.pending.drain(..).map(|(_, _, lp)| lp).collect()
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `stop`, on char boundaries.
fn longest_suffix_prefix(text: &str, stop: &str) -> usize {
    let max = text.len().min(stop.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&n| {
            let start = text.len() - n;
            text.is_char_boundary(start) && stop.is_char_boundary(n) && text[start..] == stop[..n]
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, Option<String>) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()));
        let mut out = String::new();
        for p in pieces {
            match m.push(p) {
                StopResult::Continue(t) => out.push_str(&t),
                StopResult::Stopped { text, stop } => {
                    out.push_str(&text);
                    return (out, Some(stop));
                }
            }
        }
        out.push_str(&m.flush());
        (out, None)
    }

    #[test]
    fn no_stops_passes_through() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), None));
    }

    #[test]
    fn stop_split_across_pieces() {
        assert_eq!(
            run(&["END"], &["hello E", "N", "D more"]),
            ("hello ".into(), Some("END".into()))
        );
    }

    #[test]
    fn partial_prefix_is_released() {
        assert_eq!(
            run(&["END"], &["hello EN", "X"]),
            ("hello ENX".into(), None)
        );
        assert_eq!(run(&["END"], &["tail E"]), ("tail E".into(), None));
    }

    #[test]
    fn earliest_stop_wins() {
        assert_eq!(
            run(&["bb", "a"], &["xbbya"]),
            ("x".into(), Some("bb".into()))
        );
    }

    fn lp(token: &str) -> TokenLogprob {
        TokenLogprob {
            token: token.into(),
            logprob: 0.0,
            bytes: token.as_bytes().to_vec(),
            top: vec![],
        }
    }

    /// Runs tokens through matcher + hold; returns (text, tokens whose logprobs were released).
    fn run_lp(stops: &[&str], pieces: &[&str]) -> (String, Vec<String>) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()));
        let mut h = LogprobHold::default();
        let (mut text, mut toks) = (String::new(), Vec::new());
        let mut take = |lps: Vec<TokenLogprob>| toks.extend(lps.into_iter().map(|l| l.token));
        for p in pieces {
            h.push(p, lp(p));
            match m.push(p) {
                StopResult::Continue(t) => {
                    take(h.release(&t));
                    text.push_str(&t);
                }
                StopResult::Stopped { text: t, .. } => {
                    take(h.release(&t));
                    text.push_str(&t);
                    return (text, toks);
                }
            }
        }
        let rest = m.flush();
        take(h.flush(&rest));
        text.push_str(&rest);
        (text, toks)
    }

    #[test]
    fn logprobs_follow_released_text() {
        // Nothing matches: every token is released, none lost to hold-back.
        let (t, l) = run_lp(&["END"], &["a", " E", "N", "X"]);
        assert_eq!(
            (t.as_str(), l),
            (
                "a ENX",
                vec!["a", " E", "N", "X"]
                    .into_iter()
                    .map(String::from)
                    .collect()
            )
        );
        // Held-back tail released by the final flush.
        let (t, l) = run_lp(&["END"], &["a", " E"]);
        assert_eq!((t.as_str(), l.len()), ("a E", 2));
    }

    #[test]
    fn logprobs_of_stop_tokens_are_dropped() {
        // "EN" + "D" are wholly inside the stop string.
        let (t, l) = run_lp(&["END"], &["a", "b", "EN", "D", "c"]);
        assert_eq!(
            (t.as_str(), l),
            ("ab", vec!["a".to_string(), "b".to_string()])
        );
        // A token straddling the stop boundary stays: " xE" contributed " x" to the text.
        let (t, l) = run_lp(&["END"], &["a", " xE", "ND"]);
        assert_eq!(
            (t.as_str(), l),
            ("a x", vec!["a".to_string(), " xE".to_string()])
        );
    }

    #[test]
    fn multibyte_is_safe() {
        assert_eq!(
            run(&["é!"], &["café", "!"]),
            ("caf".into(), Some("é!".into()))
        );
    }

    #[test]
    fn paused_matcher_passes_text_through() {
        let mut m = StopMatcher::new(["END".to_string()]);
        m.set_checking(false);
        // Nothing matches and nothing is held back while paused.
        assert_eq!(m.push("the EN"), StopResult::Continue("the EN".into()));
        assert_eq!(m.push("D"), StopResult::Continue("D".into()));
        assert_eq!(m.flush(), String::new());
        // Once checking, later text matches normally.
        m.set_checking(true);
        assert_eq!(m.push("stop EN"), StopResult::Continue("stop ".into()));
        assert_eq!(
            m.push("D!"),
            StopResult::Stopped {
                text: String::new(),
                stop: "END".into()
            }
        );
    }
}
