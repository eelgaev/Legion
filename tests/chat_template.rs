//! Byte-for-byte comparison against HF transformers' rendering. The fixtures in
//! `tests/data/chat_template/` are produced by `generate.py` (see the README there).

use std::collections::BTreeMap;
use std::path::PathBuf;

use legion::chat_template::ChatTemplate;
use legion::engine::{Message, Tool};
use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Deserialize)]
struct Case {
    name: String,
    system: Option<String>,
    messages: Vec<Message>,
    tools: Vec<Tool>,
    add_generation_prompt: bool,
    kwargs: Map<String, Value>,
    templates: Vec<String>,
}

#[derive(Deserialize)]
struct Expected {
    output: Option<String>,
    error: Option<String>,
}

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/chat_template")
}

fn cases() -> Vec<Case> {
    serde_json::from_slice(&std::fs::read(dir().join("cases.json")).unwrap()).unwrap()
}

fn template(name: &str) -> ChatTemplate {
    let source = std::fs::read_to_string(dir().join(format!("templates/{name}.jinja"))).unwrap();
    let (bos, eos) = match name {
        "qwen3" => ("", "<|im_end|>"),
        "qwen36" => ("<|endoftext|>", "<|im_end|>"),
        "gemma3" => ("<bos>", "<end_of_turn>"),
        "gemma4" => ("<bos>", "<eos>"),
        "nemotron35" => ("<s>", "<|im_end|>"),
        "deepseek41" => ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"),
        other => panic!("unknown template {other}"),
    };
    ChatTemplate::new(&source, bos, eos).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn check(name: &str) -> usize {
    let tpl = template(name);
    let expected: BTreeMap<String, Expected> = serde_json::from_slice(
        &std::fs::read(dir().join(format!("expected/{name}.json"))).unwrap(),
    )
    .unwrap();
    let mut checked = 0;
    for c in cases()
        .iter()
        .filter(|c| c.templates.iter().any(|t| t == name))
    {
        let want = &expected[&c.name];
        let got = tpl.render(
            c.system.as_deref(),
            &c.messages,
            &c.tools,
            c.add_generation_prompt,
            &c.kwargs,
        );
        match (&want.output, &want.error, got) {
            (Some(out), _, Ok(got)) => {
                assert_eq!(
                    &got, out,
                    "{name}/{}: output differs from transformers",
                    c.name
                )
            }
            (_, Some(err), Err(got)) => assert!(
                got.to_string().contains(err.as_str()),
                "{name}/{}: error {got:?} does not contain {err:?}",
                c.name
            ),
            (_, _, got) => panic!(
                "{name}/{}: expected {:?} / {:?}, got {got:?}",
                c.name, want.output, want.error
            ),
        }
        checked += 1;
    }
    assert!(checked > 0);
    checked
}

#[test]
fn qwen3_matches_transformers() {
    check("qwen3");
}

#[test]
fn qwen36_matches_transformers() {
    check("qwen36");
}

#[test]
fn gemma3_matches_transformers() {
    check("gemma3");
}

#[test]
fn gemma4_matches_transformers() {
    check("gemma4");
}

#[test]
fn nemotron35_matches_transformers() {
    check("nemotron35");
}

#[test]
fn deepseek41_matches_transformers() {
    check("deepseek41");
}

fn render(source: &str) -> Result<String, legion::chat_template::ChatTemplateError> {
    ChatTemplate::new(source, "<s>", "</s>")?.render(None, &[], &[], false, &Map::new())
}

#[test]
fn raise_exception_returns_err_with_message() {
    let err = render("{{ raise_exception('boom: bad roles') }}").unwrap_err();
    assert!(err.to_string().contains("boom: bad roles"), "{err}");
}

#[test]
fn syntax_error_returns_err() {
    assert!(ChatTemplate::new("{% if %}", "", "").is_err());
    assert!(ChatTemplate::new("{{ unclosed", "", "").is_err());
}

#[test]
fn runtime_errors_return_err() {
    assert!(render("{{ 1 // 0 }}").is_err());
    assert!(render("{{ nope.attr.deeper }}").is_err());
    assert!(render("{% for x in 5 %}{% endfor %}").is_err());
}

#[test]
fn message_without_role_is_err() {
    let tpl = ChatTemplate::new("x", "", "").unwrap();
    let m = Message::default();
    assert!(tpl.render(None, &[m], &[], false, &Map::new()).is_err());
}

#[test]
fn tojson_matches_python_json_dumps() {
    let src = r#"{{ {"a": [1, 2.5, none, true], "b": "é\n\"x\" <&>"} | tojson }}|{{ {"k": "é"} | tojson(ensure_ascii=true) }}|{{ [1, {"z": 1, "a": 2}] | tojson(indent=2, sort_keys=true) }}"#;
    let got = render(src).unwrap();
    assert_eq!(
        got,
        "{\"a\": [1, 2.5, null, true], \"b\": \"é\\n\\\"x\\\" <&>\"}|{\"k\": \"\\u00e9\"}|[\n  1,\n  {\n    \"a\": 2,\n    \"z\": 1\n  }\n]"
    );
}

#[test]
fn python_string_methods_and_loop_items() {
    let src = "{% for x in ['  ab ', 'cd'] %}{{ x.strip() }}{{ loop.previtem }}{{ loop.nextitem }};{% endfor %}\
               {{ 'abc'.startswith('a') }}{{ 'abc'.endswith('c') }}{{ 'a-b'.split('-')[1] }}{{ 'xxaxx'.lstrip('x') }}{{ 'xxaxx'.rstrip('x') }}{{ 'aXa'.replace('X', '-') }}";
    assert_eq!(render(src).unwrap(), "abcd;cd  ab ;TrueTruebaxxxxaa-a");
}

#[test]
fn strftime_now_formats() {
    let got = render("{{ strftime_now('%Y') }}").unwrap();
    assert_eq!(got.len(), 4);
    assert!(got.chars().all(|c| c.is_ascii_digit()));
}

#[test]
fn python_style_scalars() {
    assert_eq!(
        render("{{ true }}{{ false }}{{ none }}{{ true|string }}{{ none|string }}{{ 2.5 }}{{ 3 }}")
            .unwrap(),
        "TrueFalseNoneTrueNone2.53"
    );
}

/// `int` as Jinja2 has it (the expected line is what jinja2 renders): what does not convert
/// gives the default, 0 without one.
#[test]
fn int_filter_matches_jinja2() {
    assert_eq!(
        render(
            r#"{{ "huge"|int }},{{ "42"|int }},{{ " 7 "|int }},{{ "3.7"|int }},{{ "x"|int(5) }},{{ "x"|int(default=-1) }},{{ "0x1f"|int(base=16) }},{{ true|int }},{{ none|int }},{{ 2.9|int }},{{ "-12"|int }},{{ ("30"|int|string) == "30" }},{{ ("huge"|int|string) == "huge" }}"#
        )
        .unwrap(),
        "0,42,7,3,5,-1,31,1,0,2,-12,True,False"
    );
}

#[test]
fn kwargs_are_visible_but_builtins_win() {
    let tpl = ChatTemplate::new(
        "{{ enable_thinking }}{{ bos_token }}{{ add_generation_prompt }}",
        "<s>",
        "",
    )
    .unwrap();
    let kwargs = serde_json::json!({"enable_thinking": false, "bos_token": "evil", "add_generation_prompt": true});
    let kwargs = kwargs.as_object().unwrap();
    assert_eq!(
        tpl.render(None, &[], &[], false, kwargs).unwrap(),
        "False<s>False"
    );
}

#[test]
fn none_is_not_iterable() {
    // HF passes `tools=None` without tools; Jinja2 says None is not iterable, so a template
    // may guard `| length` with it (Nemotron-H's does).
    assert_eq!(
        render("{% if tools is iterable and tools | length > 0 %}x{% else %}y{% endif %}").unwrap(),
        "y"
    );
    assert_eq!(
        render(
            "{{ none is iterable }}{{ 'ab' is iterable }}{{ [1] is iterable }}{{ 3 is iterable }}"
        )
        .unwrap(),
        "FalseTrueTrueFalse"
    );
}

/// #146: a message's images go to the template as HF content parts before its text, so Qwen's
/// template renders one `<|vision_start|><|image_pad|><|vision_end|>` per image there. A message
/// without images still passes its text as a plain string.
#[test]
fn images_render_as_qwen_vision_placeholders() {
    use legion::engine::{ImageInput, Role};
    let image = ImageInput {
        bytes: vec![1, 2, 3],
        mime: "image/png".into(),
    };
    let messages = [
        Message {
            role: Some(Role::User),
            content: "Compare them.".into(),
            images: vec![image.clone(), image],
            ..Default::default()
        },
        Message {
            role: Some(Role::Assistant),
            content: "Same.".into(),
            ..Default::default()
        },
    ];
    let out = template("qwen36")
        .render(None, &messages, &[], true, &Map::new())
        .unwrap();
    let pad = "<|vision_start|><|image_pad|><|vision_end|>";
    assert!(
        out.contains(&format!(
            "<|im_start|>user\n{pad}{pad}Compare them.<|im_end|>"
        )),
        "{out}"
    );
    assert_eq!(out.matches("<|image_pad|>").count(), 2, "{out}");
    assert!(out.contains("Same."), "{out}");
}
