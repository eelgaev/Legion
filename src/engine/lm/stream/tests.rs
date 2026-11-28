use super::*;
use crate::engine::Tool;
use crate::engine::tool_parser::HermesParser;
use crate::gguf::{GgufFile, GgufWriter};
use tokio::sync::mpsc::UnboundedReceiver;

/// The tiny models' byte-level tokenizer: token `b` is byte `b`.
fn tokenizer() -> Arc<Tokenizer> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vocab.gguf");
    let mut w = GgufWriter::new();
    crate::model::qwen35::tiny::byte_tokenizer(&mut w).unwrap();
    w.write(&path).unwrap();
    Arc::new(Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap())
}

/// What a sequence's text is made of.
#[derive(Clone, Default)]
struct Kind {
    reasoning: bool,
    tools: bool,
    stop: Vec<String>,
    logprobs: Option<u32>,
    grammar: bool,
    trim_content: bool,
}

fn spec(kind: &Kind) -> (TextSpec, UnboundedReceiver<TokenEvent>) {
    let (out, rx) = tokio::sync::mpsc::unbounded_channel();
    let tools = [Tool {
        name: "get".into(),
        description: None,
        parameters: serde_json::json!({}),
        strict: false,
    }];
    let spec = TextSpec {
        grammar: kind.grammar,
        reasoning: kind
            .reasoning
            .then(|| ReasoningParser::new("<think>", "</think>", false)),
        trim_content: kind.trim_content,
        tools: kind
            .tools
            .then(|| Box::new(HermesParser::new(&tools, false)) as Box<dyn ToolParser>),
        stop: StopMatcher::new(kind.stop.clone()),
        logprobs: kind.logprobs,
        shown: Vec::new(),
        out,
    };
    (spec, rx)
}

/// A row of logits over the tokenizer's vocabulary, different per position.
fn row(at: usize) -> Vec<f32> {
    (0..259)
        .map(|i| ((i * 7 + at * 13) % 31) as f32 * 0.25)
        .collect()
}

/// How a scripted sequence ends after its tokens.
#[derive(Clone, Copy)]
enum Close {
    /// A stop token: flushed, `ToolCalls` when it called tools.
    StopToken,
    /// `max_tokens`: flushed.
    Length,
    /// Cancelled: not flushed.
    Cancel,
    /// The engine failed.
    Fail,
}

fn usage(n: usize) -> Usage {
    Usage {
        prompt_tokens: 3,
        completion_tokens: n as u32,
        ..Usage::default()
    }
}

/// Runs the sequences `(kind, text, close)` through an outbox with `min_seqs`, one token of
/// each per step (as the engine thread does), and returns each one's events and whether its
/// text went away.
fn drive(min_seqs: usize, seqs: &[(Kind, &str, Close)]) -> Vec<(Vec<TokenEvent>, bool)> {
    let tok = tokenizer();
    let mut outbox = Outbox::new(tok.clone(), min_seqs);
    let mut rxs = Vec::new();
    let mut texts = Vec::new();
    for (i, (kind, _, _)) in seqs.iter().enumerate() {
        let (spec, rx) = spec(kind);
        let (text, live) = outbox.text(&tok, i as SeqId, spec);
        rxs.push(rx);
        texts.push(Some((text, live)));
    }
    let away: Vec<bool> = texts
        .iter()
        .map(|t| matches!(t, Some((Text::Away(_), _))))
        .collect();
    let longest = seqs.iter().map(|(_, s, _)| s.len()).max().unwrap_or(0);
    for at in 0..=longest {
        for (i, (kind, s, close)) in seqs.iter().enumerate() {
            let Some((text, _)) = &mut texts[i] else {
                continue;
            };
            let bytes = s.as_bytes();
            if at < bytes.len() {
                let raw = kind.logprobs.map(|_| row(at));
                if let Some(stop) = text.token(&mut outbox, bytes[at] as u32, raw, true) {
                    text.end(
                        &mut outbox,
                        End {
                            flush: false,
                            reason: FinishReason::StopSequence,
                            stop_sequence: Some(stop),
                            usage: usage(at + 1),
                            tools: false,
                        },
                    );
                    texts[i] = None;
                }
                continue;
            }
            let end = |flush, reason, tools| End {
                flush,
                reason,
                stop_sequence: None,
                usage: usage(bytes.len()),
                tools,
            };
            match close {
                Close::StopToken => text.end(&mut outbox, end(true, FinishReason::Stop, true)),
                Close::Length => text.end(&mut outbox, end(true, FinishReason::Length, false)),
                Close::Cancel => text.end(&mut outbox, end(false, FinishReason::Cancelled, false)),
                Close::Fail => text.fail(&mut outbox, "boom".into()),
            }
            texts[i] = None;
        }
        outbox.flush();
    }
    // Joins the stream thread: everything it was sent is sent on.
    drop(outbox);
    rxs.into_iter()
        .zip(away)
        .map(|(mut rx, away)| {
            let mut events = Vec::new();
            while let Ok(ev) = rx.try_recv() {
                // Tool-call ids are random.
                events.push(match ev {
                    TokenEvent::ToolCall(c) => TokenEvent::ToolCall(crate::engine::ToolCall {
                        id: String::new(),
                        ..c
                    }),
                    ev => ev,
                });
            }
            (events, away)
        })
        .collect()
}

fn script() -> Vec<(Kind, &'static str, Close)> {
    let plain = Kind::default();
    let reasoning = Kind {
        reasoning: true,
        ..Kind::default()
    };
    let tools = Kind {
        reasoning: true,
        tools: true,
        ..Kind::default()
    };
    let logprobs = Kind {
        logprobs: Some(2),
        ..Kind::default()
    };
    vec![
        (plain.clone(), "héllo wörld, ünïcode", Close::Length),
        (
            reasoning,
            "<think>why not</think>the answer",
            Close::StopToken,
        ),
        (
            tools,
            "<think>call it</think>ok <tool_call>\n{\"name\": \"get\", \"arguments\": {\"a\": 1}}\n</tool_call>",
            Close::StopToken,
        ),
        (logprobs, "probé mé", Close::Length),
        (plain.clone(), "cut short", Close::Cancel),
        (plain, "and then", Close::Fail),
    ]
}

#[test]
fn text_away_streams_what_text_here_streams() {
    let here = drive(0, &script());
    let away = drive(1, &script());
    assert!(here.iter().all(|(_, away)| !away));
    assert!(away.iter().all(|(_, away)| *away));
    for ((h, _), (a, _)) in here.iter().zip(&away) {
        assert!(!h.is_empty());
        assert_eq!(h, a);
    }
    // The script covers what it means to: tool calls, reasoning, log probabilities, every end.
    let all: Vec<&TokenEvent> = here.iter().flat_map(|(e, _)| e).collect();
    assert!(all.iter().any(|e| matches!(e, TokenEvent::ToolCall(_))));
    assert!(all.iter().any(|e| matches!(e, TokenEvent::Reasoning(_))));
    assert!(all.iter().any(|e| matches!(e, TokenEvent::Logprobs(_))));
    assert!(all.iter().any(|e| matches!(
        e,
        TokenEvent::Finished {
            reason: FinishReason::ToolCalls,
            ..
        }
    )));
    assert!(matches!(here[5].0.last(), Some(TokenEvent::Error(_))));
}

#[test]
fn text_stays_here_below_min_seqs() {
    let seqs: Vec<_> = (0..5)
        .map(|_| (Kind::default(), "abc", Close::Length))
        .collect();
    let out = drive(3, &seqs);
    let away: Vec<bool> = out.iter().map(|(_, away)| *away).collect();
    assert_eq!(away, [false, false, true, true, true]);
    for (events, _) in &out {
        assert_eq!(events[0], TokenEvent::Text("a".into()));
    }
}

#[test]
fn stop_strings_keep_text_here() {
    let stop = Kind {
        stop: vec!["lo".into()],
        ..Kind::default()
    };
    let out = drive(1, &[(stop, "hello there", Close::Length)]);
    let (events, away) = &out[0];
    assert!(!away);
    assert_eq!(
        events,
        &[
            TokenEvent::Text("h".into()),
            TokenEvent::Text("e".into()),
            // The second "l" might have started "lo".
            TokenEvent::Text("l".into()),
            TokenEvent::Finished {
                reason: FinishReason::StopSequence,
                stop_sequence: Some("lo".into()),
                usage: usage(5),
            },
        ]
    );
}

#[test]
fn grammars_keep_text_here_and_see_the_reasoning() {
    let grammar = Kind {
        grammar: true,
        reasoning: true,
        ..Kind::default()
    };
    let out = drive(1, &[(grammar.clone(), "<think>a</think>b", Close::Length)]);
    let (events, away) = &out[0];
    assert!(!away);
    assert_eq!(events[0], TokenEvent::Reasoning("a".into()));
    // The grammar reads where the reasoning parser is after each token.
    let tok = tokenizer();
    let mut outbox = Outbox::new(tok.clone(), 1);
    let (s, _rx) = spec(&grammar);
    let (mut text, _live) = outbox.text(&tok, 0, s);
    let mut inside = Vec::new();
    for b in "<think>a</think>b".bytes() {
        text.token(&mut outbox, b as u32, None, true);
        inside.push(text.inside_reasoning());
    }
    assert!(inside[6] && inside[7]);
    assert!(!inside[15] && !inside[16]);
}

#[test]
fn live_sequences_count_until_dropped() {
    let tok = tokenizer();
    let mut outbox = Outbox::new(tok.clone(), 2);
    let (a, _rx_a) = spec(&Kind::default());
    let (first, live) = outbox.text(&tok, 0, a);
    assert!(matches!(first, Text::Here(_)));
    drop(live);
    // The first one ended: the second is alone again.
    let (b, _rx_b) = spec(&Kind::default());
    let (second, _live_b) = outbox.text(&tok, 1, b);
    assert!(matches!(second, Text::Here(_)));
    let (c, _rx_c) = spec(&Kind::default());
    let (third, _live_c) = outbox.text(&tok, 2, c);
    assert!(matches!(third, Text::Away(2)));
}

#[test]
fn away_text_leaves_on_flush() {
    let tok = tokenizer();
    let mut outbox = Outbox::new(tok.clone(), 1);
    let (s, mut rx) = spec(&Kind::default());
    let (mut text, _live) = outbox.text(&tok, 7, s);
    assert!(text.token(&mut outbox, b'x' as u32, None, true).is_none());
    // Queued on the engine thread, not sent yet.
    assert!(rx.try_recv().is_err());
    outbox.flush();
    assert_eq!(
        rx.blocking_recv(),
        Some(TokenEvent::Text("x".into())),
        "the stream thread sends the step's text"
    );
}

#[test]
fn zero_min_seqs_never_starts_the_thread() {
    let tok = tokenizer();
    let mut outbox = Outbox::new(tok.clone(), 0);
    let mut lives = Vec::new();
    for i in 0..64 {
        let (s, _rx) = spec(&Kind::default());
        let (text, live) = outbox.text(&tok, i, s);
        assert!(matches!(text, Text::Here(_)));
        lives.push(live);
    }
    assert!(outbox.thread.is_none());
}

/// A sequence's events merged into `(reasoning, text)`, and whether its first `Text` event
/// starts with whitespace (a trimmed start sends none before the content).
fn merged(events: &[TokenEvent]) -> (String, String, bool) {
    let (mut r, mut t, mut led) = (String::new(), String::new(), false);
    for e in events {
        match e {
            TokenEvent::Reasoning(s) => r.push_str(s),
            TokenEvent::Text(s) => {
                led |= t.is_empty() && s.starts_with(char::is_whitespace);
                t.push_str(s);
            }
            _ => {}
        }
    }
    (r, t, led)
}

#[test]
fn trimmed_content_drops_only_its_leading_whitespace() {
    let trim = Kind {
        trim_content: true,
        ..Kind::default()
    };
    let thinking = Kind {
        reasoning: true,
        ..trim.clone()
    };
    let stop = Kind {
        stop: vec!["x".into()],
        ..trim.clone()
    };
    let logprobs = Kind {
        logprobs: Some(1),
        ..trim.clone()
    };
    let tools = Kind {
        tools: true,
        ..trim.clone()
    };
    let seqs = [
        (trim.clone(), " red apple \n", Close::StopToken),
        (trim.clone(), "\n\n \t two  words", Close::Length),
        // Reasoning keeps its whitespace; the content after it starts trimmed.
        (
            thinking,
            "<think> plan </think>\n\n answer",
            Close::StopToken,
        ),
        (trim.clone(), " \n ", Close::StopToken),
        (stop, "  ax", Close::Length),
        (logprobs, "  hi", Close::Length),
        (
            tools,
            "\n<tool_call>\n{\"name\": \"get\", \"arguments\": {}}\n</tool_call>",
            Close::StopToken,
        ),
        // Without the trim, the text is as the model wrote it.
        (Kind::default(), " red apple", Close::StopToken),
    ];
    let want = [
        ("", "red apple \n"),
        ("", "two  words"),
        (" plan ", "answer"),
        ("", ""),
        ("", "a"),
        ("", "hi"),
        ("", ""),
        ("", " red apple"),
    ];
    let here = drive(0, &seqs);
    let away = drive(1, &seqs);
    for ((h, _), (a, _)) in here.iter().zip(&away) {
        assert_eq!(h, a);
    }
    for (i, ((events, _), (r, t))) in here.iter().zip(want).enumerate() {
        let (got_r, got_t, led) = merged(events);
        assert_eq!((got_r.as_str(), got_t.as_str()), (r, t), "{i}: {events:?}");
        assert_eq!(led, i == 7, "{i}: {events:?}");
    }
    // The leading whitespace's tokens keep their log probabilities.
    let lps: usize = here[5]
        .0
        .iter()
        .map(|e| match e {
            TokenEvent::Logprobs(l) => l.len(),
            _ => 0,
        })
        .sum();
    assert_eq!(lps, 4, "{:?}", here[5].0);
    assert!(
        here[6]
            .0
            .iter()
            .any(|e| matches!(e, TokenEvent::ToolCall(_)))
    );
    assert!(matches!(
        here[4].0.last(),
        Some(TokenEvent::Finished {
            reason: FinishReason::StopSequence,
            ..
        })
    ));
}
