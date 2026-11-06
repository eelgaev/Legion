use super::*;
use crate::engine::TokenEvent;
use crate::gguf::{GgufFile, GgufWriter, Value};
use serde_json::json;

fn tokenizer(tokens: &[&str], pre: &str) -> Tokenizer {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vocab.gguf");
    let mut writer = GgufWriter::new();
    for (key, value) in [
        ("tokenizer.ggml.model", Value::String("gpt2".into())),
        ("tokenizer.ggml.pre", Value::String(pre.into())),
        (
            "tokenizer.ggml.tokens",
            Value::Array(tokens.iter().map(|t| Value::String((*t).into())).collect()),
        ),
        ("tokenizer.ggml.merges", Value::Array(vec![])),
    ] {
        writer.set(key, &value).unwrap();
    }
    writer.write(&path).unwrap();
    Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
}

fn check_parsers(format: &ChatFormat, reasoning: &str, call: &str) {
    let mut parser = format.reasoning_parser(false);
    let mut events = Vec::new();
    for c in reasoning.chars() {
        events.extend(parser.push(&c.to_string()));
    }
    events.extend(parser.finish());
    let mut thought = String::new();
    let mut answer = String::new();
    for event in events {
        match event {
            TokenEvent::Reasoning(t) => thought.push_str(&t),
            TokenEvent::Text(t) => answer.push_str(&t),
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert_eq!(thought, "plan");
    assert_eq!(answer, "answer");

    let mut parser = format.tool_parser(&[Tool {
        name: "lookup".into(),
        description: None,
        parameters: json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
        strict: false,
    }]);
    let mut events = Vec::new();
    for c in call.chars() {
        events.extend(parser.push(&c.to_string()));
    }
    events.extend(parser.finish());
    let [TokenEvent::ToolCall(call)] = events.as_slice() else {
        panic!("expected one tool call: {events:?}");
    };
    assert_eq!(call.name, "lookup");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap(),
        json!({"n": 3})
    );
    assert!(parser.had_calls());
}

#[test]
fn qwen_and_nemotron_keep_think_and_coder_parsers() {
    // Qwen3.5/3.6 and Qwen4 use the Qwen pre-tokenizers, Nemotron-H uses pixtral.
    for pre in ["qwen35", "qwen2", "pixtral"] {
        let tok = tokenizer(&["<think>", "</think>", "<tool_call>", "</tool_call>"], pre);
        let format = ChatFormat::detect(&tok);
        for (prompt, inside) in [
            ("assistant", false),
            ("assistant<think>\n", true),
            ("assistant<think></think>\n", false),
            ("<think>old</think>assistant<think>", true),
            ("</think>", false),
        ] {
            assert_eq!(
                format.prompt_reasoning(prompt),
                Some(inside),
                "{pre}: {prompt}"
            );
        }
        check_parsers(
            &format,
            "<think>plan</think>answer",
            "<tool_call><function=lookup><parameter=n>3</parameter></function></tool_call>",
        );
    }
}

#[test]
fn glm_vocabularies_take_the_arg_key_parser_and_no_strict_tools() {
    let tok = tokenizer(
        &[
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
            "<|observation|>",
        ],
        "glm4",
    );
    let format = ChatFormat::detect(&tok);
    assert_eq!(format.prompt_reasoning("<|assistant|><think>"), Some(true));
    check_parsers(
        &format,
        "<think>plan</think>answer",
        "<tool_call>lookup<arg_key>n</arg_key><arg_value>3</arg_value></tool_call>",
    );
    assert_eq!(format.strict_tools(), None);
    assert!(format.stop_tokens(&tok).contains(&8));
}

#[test]
fn gemma_channels_take_precedence_and_start_inside_only_when_the_prompt_opens_them() {
    let tok = tokenizer(
        &["<|channel>", "<channel|>", "<|tool_call>", "<think>"],
        "qwen2",
    );
    let format = ChatFormat::detect(&tok);
    for (prompt, inside) in [
        ("assistant", false),
        ("<|channel>thought<channel|>", false),
        ("<|turn>model\n<|channel>thought\n<channel|>", false),
        ("<|channel>thought\nold\n<channel|>user", false),
        ("<think>", false),
        // The template opens the channel after a tool response when thinking is on.
        ("<|channel>thought", true),
        (
            "<|tool_response>...<tool_response|><|channel>thought\n",
            true,
        ),
    ] {
        assert_eq!(format.prompt_reasoning(prompt), Some(inside), "{prompt}");
    }
    check_parsers(
        &format,
        "<|channel>thoughtplan<channel|>answer",
        "<|tool_call>call:lookup{n:3}<tool_call|>",
    );
}

#[test]
fn partial_channel_vocabularies_keep_the_fallback_format() {
    let markers = ["<|channel>", "<channel|>", "<|tool_call>"];
    for missing in 0..markers.len() {
        let tokens: Vec<_> = markers
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (i != missing).then_some(t))
            .collect();
        let format = ChatFormat::detect(&tokenizer(&tokens, "qwen2"));
        assert_eq!(format.prompt_reasoning("<think>"), None);
        check_parsers(
            &format,
            "<think>plan</think>answer",
            "<tool_call><function=lookup><parameter=n>3</parameter></function></tool_call>",
        );
    }
}

#[test]
fn stop_tokens_include_each_available_marker_even_without_channels() {
    let tokens = [
        "<|im_end|>",
        "<|endoftext|>",
        "<turn|>",
        "<|tool_response>",
        "<|observation|>",
    ];
    for mask in 0..1 << tokens.len() {
        let vocab: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (mask & (1 << i) != 0).then_some(t))
            .collect();
        let tok = tokenizer(&vocab, "qwen2");
        let format = ChatFormat::detect(&tok);
        let expected: Vec<_> = [tok.eos_id(), tok.eot_id()]
            .into_iter()
            .flatten()
            .chain(0..vocab.len() as u32)
            .collect();
        assert_eq!(format.stop_tokens(&tok), expected);
    }
}

#[test]
fn shipped_vocabularies_select_the_expected_reasoning_markers() {
    for (file, markers) in [
        ("ggml-vocab-qwen35.gguf", ("<think>", "</think>")),
        (
            "ggml-vocab-gemma-4.gguf",
            ("<|channel>thought", "<channel|>"),
        ),
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(file);
        let tok = Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap();
        assert_eq!(ChatFormat::detect(&tok).reasoning, markers, "{file}");
    }
}
