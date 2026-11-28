use super::super::chat_format::ChatFormat;
use super::*;
use crate::engine::TokenEvent;
use crate::engine::tool_parser::{ToolParser, inkling, muse};
use crate::gguf::{GgufFile, GgufWriter, Value};

/// A byte-level vocabulary: `special` control tokens first, then plain ones.
fn tokenizer(special: &[&str], plain: &[&str], pre: &str) -> Tokenizer {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vocab.gguf");
    let tokens: Vec<&str> = special.iter().chain(plain).copied().collect();
    let types = (0..tokens.len())
        .map(|i| Value::I32(if i < special.len() { 3 } else { 1 }))
        .collect();
    let mut writer = GgufWriter::new();
    for (key, value) in [
        ("tokenizer.ggml.model", Value::String("gpt2".into())),
        ("tokenizer.ggml.pre", Value::String(pre.into())),
        (
            "tokenizer.ggml.tokens",
            Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
        ),
        ("tokenizer.ggml.token_type", Value::Array(types)),
        ("tokenizer.ggml.merges", Value::Array(vec![])),
    ] {
        writer.set(key, &value).unwrap();
    }
    writer.write(&path).unwrap();
    Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
}

fn shipped(file: &str) -> Tokenizer {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file);
    Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
}

fn id(tok: &Tokenizer, t: &str) -> u32 {
    tok.token_id(t).unwrap_or_else(|| panic!("no token {t:?}"))
}

fn text(tok: &Tokenizer, ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids
        .iter()
        .flat_map(|&t| tok.token_to_piece(t).to_vec())
        .collect();
    String::from_utf8(bytes).unwrap()
}

/// A decode loop: the model writes `think` until its reasoning is closed (by itself after
/// `natural` tokens, or by the budget), then `answer`. Returns the tokens kept, each step
/// taking the budget's token when it has one.
fn decode(b: &mut ThinkBudget, think: u32, natural: &[u32], answer: &[u32]) -> Vec<u32> {
    let close = b.marks.close;
    let (mut kept, mut answer, mut natural) = (Vec::new(), answer.iter(), natural.iter());
    loop {
        // The model's token is only drawn when the budget has none to put in its place.
        let t = match b.forced() {
            Some(t) => t,
            None if kept.contains(&close) => match answer.next() {
                Some(&t) => t,
                None => return kept,
            },
            None => natural.next().copied().unwrap_or(think),
        };
        b.accept(t);
        kept.push(t);
        assert!(kept.len() < 10_000, "the reasoning never closed");
    }
}

/// Reasoning events kept, text events through the tool parser when there is one.
fn route(
    events: &mut Vec<TokenEvent>,
    evs: Vec<TokenEvent>,
    tools: &mut Option<Box<dyn ToolParser>>,
) {
    for ev in evs {
        match (ev, tools.as_mut()) {
            (TokenEvent::Text(t), Some(p)) => events.extend(p.push(&t)),
            (ev, _) => events.push(ev),
        }
    }
}

/// The reply's text through the format's parsers a character at a time, as streamed:
/// `(reasoning, answer)`.
fn stream(format: &ChatFormat, inside: Option<bool>, reply: &str) -> (String, String) {
    let mut reasoning = inside.map(|i| format.reasoning_parser(i));
    let mut tools = format.framed().then(|| format.tool_parser(&[], false));
    let mut events = Vec::new();
    for c in reply.chars() {
        let evs = match &mut reasoning {
            Some(r) => r.push(&c.to_string()),
            None => vec![TokenEvent::Text(c.to_string())],
        };
        route(&mut events, evs, &mut tools);
    }
    if let Some(r) = &mut reasoning {
        let evs = r.finish();
        route(&mut events, evs, &mut tools);
    }
    if let Some(p) = &mut tools {
        events.extend(p.finish());
    }
    let (mut thought, mut answer) = (String::new(), String::new());
    for ev in events {
        match ev {
            TokenEvent::Reasoning(t) => thought.push_str(&t),
            TokenEvent::Text(t) => answer.push_str(&t),
            other => panic!("unexpected event: {other:?}"),
        }
    }
    (thought, answer)
}

/// Qwen: `<think>`/`</think>` with `\n</think>\n\n` written, as the template closes it.
fn qwen() -> Tokenizer {
    tokenizer(
        &[
            "<|im_end|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
        ],
        &["Ċ", "a", "answer", "Ã", "©"],
        "qwen2",
    )
}

#[test]
fn each_format_closes_its_reasoning_its_own_way() {
    let tok = qwen();
    let format = ChatFormat::detect(&tok);
    let marks = format.think_marks().unwrap();
    assert_eq!(text(&tok, marks.inject()), "\n</think>\n\n");
    assert!(marks.inject().contains(&id(&tok, "</think>")));

    let glm = tokenizer(
        &[
            "<think>",
            "</think>",
            "<tool_call>",
            "<arg_key>",
            "<arg_value>",
        ],
        &[],
        "glm4",
    );
    let marks = ChatFormat::detect(&glm).think_marks().cloned().unwrap();
    assert_eq!(marks.inject(), [id(&glm, "</think>")]);

    let gemma = shipped("ggml-vocab-gemma-4.gguf");
    let marks = ChatFormat::detect(&gemma).think_marks().cloned().unwrap();
    assert_eq!(marks.inject(), [id(&gemma, "<channel|>")]);

    let ink = tokenizer(&inkling::MARKUP, &["a"], "inkling2");
    let marks = ChatFormat::detect(&ink).think_marks().cloned().unwrap();
    assert_eq!(marks.inject(), [id(&ink, inkling::END)]);

    let m = tokenizer(&muse::MARKUP, &["a"], "llama4");
    let marks = ChatFormat::detect(&m).think_marks().cloned().unwrap();
    assert_eq!(marks.inject(), [id(&m, muse::EOM)]);

    // No reasoning markers, no budget.
    let plain = tokenizer(&["<|im_end|>"], &["a"], "qwen2");
    assert!(ChatFormat::detect(&plain).think_marks().is_none());
}

#[test]
fn qwen_reasoning_stops_at_the_budget_and_the_answer_follows() {
    let tok = qwen();
    let format = ChatFormat::detect(&tok);
    let marks = format.think_marks().unwrap();
    let (a, answer) = (id(&tok, "a"), id(&tok, "answer"));
    for budget in [0, 1, 5, 64] {
        // The prompt opened `<think>\n`.
        let mut b = ThinkBudget::new(marks, &tok, budget, true);
        let kept = decode(&mut b, a, &[], &[answer]);
        assert_eq!(b.used(), budget);
        assert!(!b.inside());
        let reply = text(&tok, &kept);
        assert_eq!(
            reply,
            format!("{}\n</think>\n\nanswer", "a".repeat(budget as usize))
        );
        let (thought, text) = stream(&format, Some(true), &reply);
        assert_eq!(thought.trim(), "a".repeat(budget as usize), "{budget}");
        assert_eq!(text.trim(), "answer", "{budget}");
    }
}

#[test]
fn a_budget_larger_than_the_reasoning_changes_nothing() {
    let tok = qwen();
    let format = ChatFormat::detect(&tok);
    let marks = format.think_marks().unwrap();
    let (a, answer, close) = (id(&tok, "a"), id(&tok, "answer"), id(&tok, "</think>"));
    let natural = [a, a, a, close];
    let mut b = ThinkBudget::new(marks, &tok, 4, true);
    let kept = decode(&mut b, a, &natural, &[answer, answer]);
    assert_eq!(kept, [a, a, a, close, answer, answer]);
    assert_eq!(b.used(), 3);
    // Spent, but closed: nothing more is replaced, not even a later `</think>`.
    assert!(b.forced().is_none());
}

#[test]
fn reasoning_the_model_opens_counts_from_its_opener() {
    let tok = qwen();
    let marks = ChatFormat::detect(&tok).think_marks().cloned().unwrap();
    let (a, open, close) = (id(&tok, "a"), id(&tok, "<think>"), id(&tok, "</think>"));
    let mut b = ThinkBudget::new(&marks, &tok, 2, false);
    // Outside the reasoning nothing counts.
    for _ in 0..5 {
        assert!(b.forced().is_none());
        b.accept(a);
    }
    b.accept(open);
    assert!(b.inside());
    b.accept(a);
    b.accept(a);
    let mut closed = Vec::new();
    while let Some(t) = b.forced() {
        b.accept(t);
        closed.push(t);
    }
    assert_eq!(closed, marks.inject());
    assert!(closed.contains(&close));
    assert_eq!(b.used(), 2);
}

#[test]
fn the_close_waits_for_a_whole_character() {
    let tok = qwen();
    let marks = ChatFormat::detect(&tok).think_marks().cloned().unwrap();
    // `Ã` `©` are the two bytes of `é`.
    let (lead, cont) = (id(&tok, "Ã"), id(&tok, "©"));
    let mut b = ThinkBudget::new(&marks, &tok, 1, true);
    b.accept(lead);
    assert_eq!(b.used(), 1);
    assert!(b.forced().is_none(), "not inside `é`");
    b.accept(cont);
    assert_eq!(b.forced(), Some(marks.inject()[0]));
}

#[test]
fn drafts_end_where_the_budget_replaces_a_token() {
    let tok = qwen();
    let marks = ChatFormat::detect(&tok).think_marks().cloned().unwrap();
    let (a, open, close) = (id(&tok, "a"), id(&tok, "<think>"), id(&tok, "</think>"));
    // 3 left: 3 drafts kept, the step's 4th token is the close.
    let b = ThinkBudget::new(&marks, &tok, 3, true);
    let mut drafts = vec![a; 8];
    b.cut(&mut drafts);
    assert_eq!(drafts.len(), 3);
    // The model closing first keeps every draft.
    let mut drafts = vec![a, close, a, a, a, a];
    b.cut(&mut drafts);
    assert_eq!(drafts.len(), 6);
    // Spent: none.
    let mut b = ThinkBudget::new(&marks, &tok, 0, true);
    let mut drafts = vec![a; 4];
    b.cut(&mut drafts);
    assert!(drafts.is_empty());
    // Closing: none either, a token per step.
    b.accept(b.forced().unwrap());
    let mut drafts = vec![a; 4];
    b.cut(&mut drafts);
    assert!(drafts.is_empty());
    // Outside: drafts that open the reasoning end with its budget there.
    let b = ThinkBudget::new(&marks, &tok, 1, false);
    let mut drafts = vec![a, a];
    b.cut(&mut drafts);
    assert_eq!(drafts.len(), 2);
    let mut drafts = vec![a, open, a, a, a];
    b.cut(&mut drafts);
    assert_eq!(drafts, [a, open, a]);
}

#[test]
fn gemma_counts_the_thought_channel_only() {
    let tok = shipped("ggml-vocab-gemma-4.gguf");
    let format = ChatFormat::detect(&tok);
    let marks = format.think_marks().cloned().unwrap();
    let (open, close) = (id(&tok, "<|channel>"), id(&tok, "<channel|>"));
    let words = |s: &str| tok.encode_with(s, false, false);
    let mut b = ThinkBudget::new(&marks, &tok, 4, false);
    let mut kept = vec![open];
    b.accept(open);
    for t in words("thought") {
        assert!(b.forced().is_none());
        b.accept(t);
        kept.push(t);
    }
    assert!(b.inside());
    assert_eq!(b.used(), 0, "the channel's name is not reasoning");
    let think = words("\nplan the answer carefully and at length");
    let mut think = think.iter();
    while let Some(t) = b.forced().or_else(|| think.next().copied()) {
        let forced = b.forced().is_some();
        b.accept(t);
        kept.push(t);
        if forced {
            break;
        }
    }
    assert_eq!(kept.last(), Some(&close));
    assert_eq!(b.used(), 4);
    kept.extend(words("Paris."));
    let (thought, answer) = stream(&format, Some(false), &text(&tok, &kept));
    // The parser drops the newline after the channel's name.
    let counted = text(&tok, &kept[1 + words("thought").len()..][..4]);
    assert_eq!(thought, counted.trim_start_matches('\n'));
    assert_eq!(answer, "Paris.");
    // Another channel's name: not reasoning.
    let mut b = ThinkBudget::new(&marks, &tok, 0, false);
    b.accept(open);
    for t in words("final") {
        b.accept(t);
    }
    assert!(!b.inside() && b.forced().is_none());
    // The prompt opened the channel (after a tool response): inside from the start.
    let b = ThinkBudget::new(&marks, &tok, 0, true);
    assert_eq!(b.forced(), Some(close));
}

#[test]
fn inkling_closes_the_thinking_message_and_the_answer_message_follows() {
    let tok = tokenizer(&inkling::MARKUP, &["a", "answer"], "inkling2");
    let format = ChatFormat::detect(&tok);
    let marks = format.think_marks().unwrap();
    let (msg, thinking, end, text_part) = (
        id(&tok, inkling::MESSAGE),
        id(&tok, inkling::THINKING),
        id(&tok, inkling::END),
        id(&tok, inkling::MARKUP[3]),
    );
    let (a, answer) = (id(&tok, "a"), id(&tok, "answer"));
    for budget in [0, 3] {
        let mut b = ThinkBudget::new(marks, &tok, budget, false);
        let kept = decode(&mut b, a, &[msg, thinking], &[msg, text_part, answer, end]);
        assert_eq!(b.used(), budget);
        let (thought, text) = stream(&format, Some(false), &text(&tok, &kept));
        assert_eq!(
            (thought.as_str(), text.as_str()),
            ("a".repeat(budget as usize).as_str(), "answer")
        );
    }
}

#[test]
fn muse_counts_messages_to_self_only() {
    let tok = tokenizer(
        &muse::MARKUP,
        &["a", "answer", "Ġto=self", "Ġto=user", "assistant"],
        "llama4",
    );
    let format = ChatFormat::detect(&tok);
    let marks = format.think_marks().unwrap();
    let (start, message, eom) = (
        id(&tok, muse::START),
        id(&tok, muse::MESSAGE),
        id(&tok, muse::EOM),
    );
    let (a, answer) = (id(&tok, "a"), id(&tok, "answer"));
    let (to_self, to_user, assistant) = (
        id(&tok, "Ġto=self"),
        id(&tok, "Ġto=user"),
        id(&tok, "assistant"),
    );
    for budget in [0, 2, 64] {
        // The reply starts in the first message's header.
        let mut b = ThinkBudget::new(marks, &tok, budget, false);
        let kept = decode(
            &mut b,
            a,
            &[to_self, message],
            // (`<|eot|>`, a stop token, ends the reply before its text.)
            &[start, assistant, to_user, message, answer],
        );
        assert_eq!(b.used(), budget);
        let reply = text(&tok, &kept);
        let (thought, text) = stream(&format, None, &reply);
        assert_eq!(thought, "a".repeat(budget as usize), "{reply}");
        assert_eq!(text, "answer", "{reply}");
    }
    // The answer's message is not counted.
    let mut b = ThinkBudget::new(marks, &tok, 0, false);
    for t in [to_user, message, a, a, a, eom] {
        assert!(b.forced().is_none());
        b.accept(t);
    }
    assert_eq!(b.used(), 0);
    // Nor is a reply without framing, however long.
    let mut b = ThinkBudget::new(marks, &tok, 0, false);
    for _ in 0..100 {
        assert!(b.forced().is_none());
        b.accept(a);
    }
}

#[test]
fn the_effort_goes_to_the_templates_switch_or_becomes_a_budget() {
    assert_eq!(
        Switch::of("{% if enable_thinking %}<think>{% endif %}"),
        Switch::Native
    );
    assert_eq!(Switch::of("{{ reasoning_effort }}"), Switch::Effort);
    assert_eq!(
        Switch::of("{{ enable_thinking }}{{ reasoning_effort }}"),
        Switch::Native
    );
    assert_eq!(
        Switch::of("{{ reasoning_effort if reasoning_effort in ['low', 'high'] }}"),
        Switch::LowHigh
    );
    assert_eq!(
        Switch::of("{% set rs = reasoning_strength %}"),
        Switch::Strength
    );
    assert_eq!(Switch::of("{{ '<think>\\n' }}"), Switch::None);
    assert_eq!(strength("low"), Some("low"));
    assert_eq!(strength("none"), Some("low"));
    assert_eq!(strength("medium"), Some("high"));
    assert_eq!(strength("high"), Some("high"));
    assert_eq!(strength("bogus"), None);
    assert_eq!(effort_budget("none"), Some(0));
    assert_eq!(effort_budget("low"), Some(1024));
    assert_eq!(effort_budget("medium"), Some(4096));
    assert_eq!(effort_budget("high"), Some(16384));
    assert_eq!(effort_budget("max"), None);
    assert_eq!(low_high("none"), Some("low"));
    assert_eq!(low_high("minimal"), Some("low"));
    assert_eq!(low_high("low"), Some("low"));
    assert_eq!(low_high("medium"), Some("high"));
    assert_eq!(low_high("high"), Some("high"));
    assert_eq!(low_high("xhigh"), None);
    assert_eq!(low_high("max"), None);
}

/// A shipped chat template fixture and its source.
fn fixture(name: &str, bos: &str, eos: &str) -> (crate::chat_template::ChatTemplate, String) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/chat_template/templates")
        .join(format!("{name}.jinja"));
    let src = std::fs::read_to_string(path).unwrap();
    let tpl = crate::chat_template::ChatTemplate::new(&src, bos, eos).unwrap();
    (tpl, src)
}

type Kwargs = serde_json::Map<String, serde_json::Value>;

fn render_kwargs(tpl: &crate::chat_template::ChatTemplate, kwargs: &Kwargs) -> String {
    let user = crate::engine::Message {
        role: Some(crate::engine::Role::User),
        content: "Write a haiku about rain.".into(),
        ..Default::default()
    };
    tpl.render(None, &[user], &[], true, kwargs).unwrap()
}

/// The prompt a chat renders to with top-level `reasoning_effort` `effort` and the client's
/// `chat_template_kwargs` `client`, the way the server and `render` build the variables:
/// `enable_thinking` from the effort unless the client set it, then [`effort_kwargs`].
fn render_effort(
    tpl: &crate::chat_template::ChatTemplate,
    src: &str,
    framed: bool,
    effort: Option<&str>,
    client: &[(&str, serde_json::Value)],
) -> String {
    let mut kwargs: Kwargs = client
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect();
    if let Some(e) = effort
        && !kwargs.contains_key("enable_thinking")
    {
        let on = !matches!(e, "none" | "minimal");
        kwargs.insert("enable_thinking".into(), on.into());
    }
    effort_kwargs(&mut kwargs, effort.map(Into::into), Switch::of(src), framed);
    render_kwargs(tpl, &kwargs)
}

/// The pieces of a rendered prompt (split at each `<|`) that hold `needle`, trimmed.
fn lines_with<'a>(text: &'a str, needle: &str) -> Vec<&'a str> {
    text.split("<|")
        .filter(|l| l.contains(needle))
        .map(str::trim)
        .collect()
}

#[test]
fn glm_takes_the_effort_as_low_or_high() {
    let (tpl, src) = fixture("glm53", "", "<|endoftext|>");
    assert_eq!(Switch::of(&src), Switch::LowHigh);
    // No effort: the prompt is what the template renders without variables.
    let max = render_kwargs(&tpl, &Kwargs::new());
    assert_eq!(render_effort(&tpl, &src, false, None, &[]), max);
    assert_eq!(
        lines_with(&max, "Reasoning Effort"),
        ["system|>Reasoning Effort: Max"]
    );
    for (effort, line) in [
        ("none", "Low"),
        ("minimal", "Low"),
        ("low", "Low"),
        ("medium", "High"),
        ("high", "High"),
    ] {
        let got = render_effort(&tpl, &src, false, Some(effort), &[]);
        assert_eq!(
            lines_with(&got, "Reasoning Effort"),
            [format!("system|>Reasoning Effort: {line}")],
            "{effort}"
        );
        // Only that line differs from Max.
        assert_eq!(
            got.replace(&format!("Effort: {line}"), "Effort: Max"),
            max,
            "{effort}"
        );
    }
    // xhigh and max are the template's Max; an unknown effort leaves it too.
    for effort in ["xhigh", "max", "bogus"] {
        assert_eq!(
            render_effort(&tpl, &src, false, Some(effort), &[]),
            max,
            "{effort}"
        );
    }
    // The client's own kwarg wins over the top-level effort.
    let got = render_effort(
        &tpl,
        &src,
        false,
        Some("low"),
        &[("reasoning_effort", "high".into())],
    );
    assert_eq!(
        lines_with(&got, "Reasoning Effort"),
        ["system|>Reasoning Effort: High"]
    );
}

#[test]
fn inkling_takes_the_effort_as_sent() {
    let (tpl, src) = fixture("inkling2", "", "<|content_model_end_sampling|>");
    assert_eq!(Switch::of(&src), Switch::Effort);
    let default = render_kwargs(&tpl, &Kwargs::new());
    assert_eq!(render_effort(&tpl, &src, true, None, &[]), default);
    assert!(default.contains("Thinking effort level: 0.9<|end_message|>"));
    for (effort, level) in [
        ("none", "0"),
        ("minimal", "0.1"),
        ("low", "0.2"),
        ("medium", "0.7"),
        ("high", "0.9"),
        ("xhigh", "0.99"),
        ("max", "0.99"),
    ] {
        // As before: the effort as sent is the template's `reasoning_effort`.
        let mut sent = Kwargs::new();
        sent.insert("reasoning_effort".into(), effort.into());
        let got = render_effort(&tpl, &src, true, Some(effort), &[]);
        assert_eq!(got, render_kwargs(&tpl, &sent), "{effort}");
        assert!(
            got.contains(&format!("Thinking effort level: {level}<|end_message|>")),
            "{effort}: {got}"
        );
    }
}

#[test]
fn templates_with_enable_thinking_get_no_effort() {
    for (name, bos, eos) in [
        ("qwen3", "", "<|im_end|>"),
        (
            "deepseek41",
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
            "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
        ),
    ] {
        let (tpl, src) = fixture(name, bos, eos);
        assert_eq!(Switch::of(&src), Switch::Native, "{name}");
        assert_eq!(
            render_effort(&tpl, &src, false, None, &[]),
            render_kwargs(&tpl, &Kwargs::new()),
            "{name}"
        );
        // Every effort is only `enable_thinking`, as before (DeepSeek-V4.1's own
        // `reasoning_effort` would reject `medium`).
        for effort in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
            let mut thinking = Kwargs::new();
            thinking.insert(
                "enable_thinking".into(),
                (!matches!(effort, "none" | "minimal")).into(),
            );
            assert_eq!(
                render_effort(&tpl, &src, false, Some(effort), &[]),
                render_kwargs(&tpl, &thinking),
                "{name} {effort}"
            );
        }
    }
}

#[test]
fn muse_takes_the_effort_as_strength() {
    let src = "{{ reasoning_strength }}|{{ reasoning_effort }}";
    assert_eq!(Switch::of(src), Switch::Strength);
    let mut kwargs = Kwargs::new();
    effort_kwargs(&mut kwargs, Some("medium".into()), Switch::of(src), true);
    assert_eq!(kwargs["reasoning_strength"], "high");
    assert_eq!(kwargs["reasoning_effort"], "medium");
    let mut kwargs = Kwargs::new();
    kwargs.insert("reasoning_strength".into(), "low".into());
    effort_kwargs(&mut kwargs, Some("high".into()), Switch::of(src), true);
    assert_eq!(kwargs["reasoning_strength"], "low");
    let mut kwargs = Kwargs::new();
    effort_kwargs(&mut kwargs, None, Switch::of(src), true);
    assert!(kwargs.is_empty());
}
