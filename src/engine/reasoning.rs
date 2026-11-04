//! Streaming `<think>...</think>` splitter: turns model text into [`TokenEvent::Reasoning`] and
//! [`TokenEvent::Text`] events.
//!
//! Like [`super::stop::StopMatcher`], text that could still be the start of the tag we are
//! waiting for is held back until it either completes the tag (and is dropped) or can no longer
//! match (and is released).

use super::TokenEvent;

#[derive(Debug, Clone)]
pub struct ReasoningParser {
    start_tag: String,
    end_tag: String,
    inside: bool,
    strip_newline: bool,
    /// A tag was just consumed: drop one `\n` if it comes next.
    strip_pending: bool,
    held: String,
}

impl ReasoningParser {
    /// `starts_inside`: the prompt already opened the reasoning block (Qwen3 thinking
    /// templates), so generation starts in reasoning and only the end tag appears.
    pub fn new(
        start_tag: impl Into<String>,
        end_tag: impl Into<String>,
        starts_inside: bool,
    ) -> Self {
        Self {
            start_tag: start_tag.into(),
            end_tag: end_tag.into(),
            inside: starts_inside,
            strip_newline: true,
            strip_pending: false,
            held: String::new(),
        }
    }

    /// Whether a single newline right after a tag is dropped (default `true`).
    pub fn with_strip_newline(mut self, strip: bool) -> Self {
        self.strip_newline = strip;
        self
    }

    /// Whether the text pushed so far ends inside the reasoning block (its closing tag not seen).
    pub fn inside(&self) -> bool {
        self.inside
    }

    pub fn push(&mut self, piece: &str) -> Vec<TokenEvent> {
        self.held.push_str(piece);
        let mut events = Vec::new();
        loop {
            if self.strip_pending && !self.held.is_empty() {
                self.strip_pending = false;
                if self.held.starts_with('\n') {
                    self.held.drain(..1);
                }
            }
            let tag = if self.inside {
                &self.end_tag
            } else {
                &self.start_tag
            };
            // An empty tag can never match; treat the text as plain.
            if !tag.is_empty()
                && let Some(at) = self.held.find(tag.as_str())
            {
                let end = at + tag.len();
                let before: String = self.held[..at].to_string();
                self.held.drain(..end);
                self.emit(&mut events, before);
                self.inside = !self.inside;
                self.strip_pending = self.strip_newline;
                continue;
            }
            let keep = if tag.is_empty() || self.strip_pending {
                0
            } else {
                longest_suffix_prefix(&self.held, tag)
            };
            let cut = self.held.len() - keep;
            let out: String = self.held.drain(..cut).collect();
            self.emit(&mut events, out);
            return events;
        }
    }

    /// End of generation: release whatever was held back. Unterminated reasoning stays
    /// `Reasoning`.
    pub fn finish(&mut self) -> Vec<TokenEvent> {
        let mut events = Vec::new();
        let rest = std::mem::take(&mut self.held);
        self.emit(&mut events, rest);
        self.strip_pending = false;
        events
    }

    fn emit(&self, events: &mut Vec<TokenEvent>, text: String) {
        if text.is_empty() {
            return;
        }
        events.push(if self.inside {
            TokenEvent::Reasoning(text)
        } else {
            TokenEvent::Text(text)
        });
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `tag`, on char boundaries.
pub(super) fn longest_suffix_prefix(text: &str, tag: &str) -> usize {
    let max = text.len().min(tag.len().saturating_sub(1));
    (1..=max)
        .rev()
        .find(|&n| {
            let start = text.len() - n;
            text.is_char_boundary(start) && tag.is_char_boundary(n) && text[start..] == tag[..n]
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Runs the pieces through a parser and merges adjacent events of the same kind into
    /// `(reasoning, text)` strings.
    fn collect(p: &mut ReasoningParser, pieces: &[&str]) -> (String, String) {
        let mut events = Vec::new();
        for piece in pieces {
            events.extend(p.push(piece));
        }
        events.extend(p.finish());
        let (mut r, mut t) = (String::new(), String::new());
        for e in events {
            match e {
                TokenEvent::Reasoning(s) => r.push_str(&s),
                TokenEvent::Text(s) => t.push_str(&s),
                other => panic!("unexpected event {other:?}"),
            }
        }
        (r, t)
    }

    fn think(inside: bool) -> ReasoningParser {
        ReasoningParser::new("<think>", "</think>", inside)
    }

    fn s(a: &str, b: &str) -> (String, String) {
        (a.to_string(), b.to_string())
    }

    #[test]
    fn normal() {
        let mut p = think(false);
        assert_eq!(
            collect(&mut p, &["<think>\nhmm</think>\n\nanswer"]),
            s("hmm", "\nanswer")
        );
    }

    #[test]
    fn events_are_typed_and_ordered() {
        let mut p = think(false);
        let mut ev = p.push("hi <think>a</think>b");
        ev.extend(p.finish());
        assert_eq!(
            ev,
            vec![
                TokenEvent::Text("hi ".into()),
                TokenEvent::Reasoning("a".into()),
                TokenEvent::Text("b".into()),
            ]
        );
    }

    #[test]
    fn starts_inside() {
        let mut p = think(true);
        assert_eq!(
            collect(&mut p, &["step one", " two</think>\n\nanswer"]),
            s("step one two", "\nanswer")
        );
    }

    #[test]
    fn no_tags() {
        let mut p = think(false);
        assert_eq!(collect(&mut p, &["plain ", "text"]), s("", "plain text"));
    }

    #[test]
    fn tags_split_at_every_byte_position() {
        let input = "pre <think>\nréflexion</think>\n\nréponse";
        for inside_input in [false, true] {
            let input = if inside_input {
                &input["pre <think>\n".len()..]
            } else {
                input
            };
            let expected = if inside_input {
                s("réflexion", "\nréponse")
            } else {
                s("réflexion", "pre \nréponse")
            };
            for cut in 0..=input.len() {
                if !input.is_char_boundary(cut) {
                    continue;
                }
                let mut p = think(inside_input);
                let got = collect(&mut p, &[&input[..cut], &input[cut..]]);
                assert_eq!(got, expected, "cut at {cut}");
            }
        }
    }

    #[test]
    fn one_byte_at_a_time() {
        let input = "a<think>\nb</think>\nc<d";
        let mut p = think(false);
        let mut events = Vec::new();
        for ch in input.chars() {
            events.extend(p.push(&ch.to_string()));
        }
        events.extend(p.finish());
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                TokenEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "ac<d");
    }

    #[test]
    fn lookalike_is_released() {
        let mut p = think(false);
        assert_eq!(
            collect(&mut p, &["a <thi", "s is fine"]),
            s("", "a <this is fine")
        );
        let mut p = think(false);
        assert_eq!(collect(&mut p, &["tail <th"]), s("", "tail <th"));
    }

    #[test]
    fn partial_tag_is_held_back() {
        let mut p = think(false);
        assert_eq!(p.push("x <th"), vec![TokenEvent::Text("x ".into())]);
        assert_eq!(p.push("ink>r"), vec![TokenEvent::Reasoning("r".into())]);
    }

    #[test]
    fn unterminated_reasoning_is_flushed_as_reasoning() {
        let mut p = think(false);
        assert_eq!(
            collect(&mut p, &["<think>still going </thi"]),
            s("still going </thi", "")
        );
        let mut p = think(true);
        assert_eq!(collect(&mut p, &["never closed"]), s("never closed", ""));
    }

    #[test]
    fn multibyte_around_tags() {
        let mut p = think(false);
        assert_eq!(
            collect(&mut p, &["日本<think>", "é思", "考</think>", "語😀"]),
            s("é思考", "日本語😀")
        );
    }

    #[test]
    fn only_one_newline_is_stripped() {
        let mut p = think(false);
        assert_eq!(
            collect(&mut p, &["<think>\n\nx</think>\n\ny"]),
            s("\nx", "\ny")
        );
    }

    #[test]
    fn newline_strip_can_be_disabled() {
        let mut p = think(false).with_strip_newline(false);
        assert_eq!(collect(&mut p, &["<think>\nx</think>\ny"]), s("\nx", "\ny"));
    }

    #[test]
    fn newline_strip_works_across_pieces() {
        let mut p = think(false);
        assert_eq!(
            collect(&mut p, &["<think>", "\n", "x</think>", "\n", "y"]),
            s("x", "y")
        );
    }

    #[test]
    fn custom_tags() {
        let mut p = ReasoningParser::new("[[", "]]", false);
        assert_eq!(collect(&mut p, &["a[[b]]c"]), s("b", "ac"));
    }

    fn char_cuts(input: &str, mut cuts: Vec<usize>) -> Vec<&str> {
        cuts = cuts.into_iter().map(|c| c % (input.len() + 1)).collect();
        cuts.sort();
        let mut pieces = Vec::new();
        let mut last = 0;
        for c in cuts {
            let mut c = c;
            while !input.is_char_boundary(c) {
                c += 1;
            }
            if c > last {
                pieces.push(&input[last..c]);
                last = c;
            }
        }
        pieces.push(&input[last..]);
        pieces
    }

    proptest! {
        /// Output equals the input minus the tags and the one stripped newline after each,
        /// however the input is chunked.
        #[test]
        fn output_is_input_minus_tags(
            pre in "[abcé <>/\n]{0,12}",
            r in "[abcé <>/\n]{0,12}",
            a in "[abcé <>/\n]{0,12}",
            inside in any::<bool>(),
            cuts in proptest::collection::vec(0usize..100, 0..8),
        ) {
            let (input, exp_r, exp_t) = if inside {
                (format!("{r}</think>{a}"), r.clone(), a.strip_prefix('\n').unwrap_or(&a).to_string())
            } else {
                (
                    format!("{pre}<think>{r}</think>{a}"),
                    r.strip_prefix('\n').unwrap_or(&r).to_string(),
                    format!("{pre}{}", a.strip_prefix('\n').unwrap_or(&a)),
                )
            };
            let pieces = char_cuts(&input, cuts);
            let mut p = think(inside);
            let (got_r, got_t) = collect(&mut p, &pieces);
            prop_assert_eq!(got_r, exp_r);
            prop_assert_eq!(got_t, exp_t);
        }

        /// Without tags in the input, nothing is lost or changed.
        #[test]
        fn tagless_input_passes_through(
            input in "[abcé <>/\n]{0,30}",
            cuts in proptest::collection::vec(0usize..100, 0..8),
        ) {
            let pieces = char_cuts(&input, cuts);
            let mut p = think(false);
            prop_assert_eq!(collect(&mut p, &pieces), (String::new(), input.clone()));
        }
    }
}
