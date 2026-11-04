//! Strict tool-call grammars (Qwen3-Coder format) against the Qwen 3.5 vocabulary fixture:
//! what they accept and reject, and that every call they let through parses into arguments
//! that follow the tool's schema.

use std::path::PathBuf;
use std::sync::OnceLock;

use legion::engine::tool_parser::{Qwen3CoderParser, ToolParser};
use legion::engine::{TokenEvent, Tool};
use legion::gguf::GgufFile;
use legion::grammar::{Grammar, GrammarCompiler, GrammarError, Matcher, ToolSpec};
use legion::tokenizer::Tokenizer;
use serde_json::{Map, Value, json};

fn tokenizer() -> &'static Tokenizer {
    static T: OnceLock<Tokenizer> = OnceLock::new();
    T.get_or_init(|| {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ggml-vocab-qwen35.gguf");
        Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
    })
}

fn compiler() -> &'static GrammarCompiler {
    static C: OnceLock<GrammarCompiler> = OnceLock::new();
    C.get_or_init(|| GrammarCompiler::new(tokenizer()))
}

/// The call's end token. The fixture's vocabulary has only three special tokens (no
/// `</tool_call>`): one the grammar never produces otherwise stands in for it.
fn end_token() -> u32 {
    tokenizer().token_id("<|im_start|>").unwrap()
}

fn weather() -> Tool {
    Tool {
        name: "get_weather".into(),
        description: None,
        parameters: json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer"},
                "alerts": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2}
            },
            "required": ["city", "unit"],
            "additionalProperties": false
        }),
        strict: true,
    }
}

fn free_tool() -> Tool {
    Tool {
        name: "search".into(),
        description: None,
        parameters: json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        strict: false,
    }
}

fn compile(tools: &[Tool]) -> Result<Grammar, GrammarError> {
    let specs: Vec<ToolSpec<'_>> = tools
        .iter()
        .map(|t| ToolSpec {
            name: &t.name,
            strict: t.strict.then_some(&t.parameters),
        })
        .collect();
    compiler().compile_tool_calls(&specs, end_token())
}

/// Feeds `text`'s tokens; the first rejected token's index on failure.
fn feed(m: &mut Matcher, text: &str) -> Result<(), usize> {
    for (i, id) in tokenizer().encode(text, false).into_iter().enumerate() {
        m.accept_token(id).map_err(|_| i)?;
    }
    Ok(())
}

/// Whether the grammar takes `body` as a whole call (then the end token).
fn accepts(g: &Grammar, body: &str) -> bool {
    let mut m = g.matcher();
    feed(&mut m, body).is_ok() && m.accept_token(end_token()).is_ok() && m.is_finished()
}

/// The calls the engine's parser reads from `<tool_call>{body}</tool_call>`.
fn parse(tools: &[Tool], body: &str) -> Vec<(String, Value)> {
    let mut p = Qwen3CoderParser::new(tools);
    let mut events = p.push(&format!("<tool_call>{body}</tool_call>"));
    events.extend(p.finish());
    events
        .into_iter()
        .filter_map(|e| match e {
            TokenEvent::ToolCall(c) => Some((c.name, serde_json::from_str(&c.arguments).unwrap())),
            TokenEvent::Text(t) if t.trim().is_empty() => None,
            other => panic!("not a call: {other:?}"),
        })
        .collect()
}

#[test]
fn a_valid_call_is_accepted_and_parses() {
    let tools = [weather(), free_tool()];
    let g = compile(&tools).unwrap();
    let body = "\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n\
                <parameter=unit>\ncelsius\n</parameter>\n<parameter=days>\n3\n</parameter>\n\
                <parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n";
    assert!(accepts(&g, body));
    assert_eq!(
        parse(&tools, body),
        vec![(
            "get_weather".into(),
            json!({"city": "New York", "unit": "celsius", "days": 3, "tags": ["a", "b"]})
        )]
    );
    // Raw text may span lines and hold markup that is not a closing tag of the call.
    let body = "\n<function=get_weather>\n<parameter=city>\n<b>Oslo</b>\nNorway\n</parameter>\n\
                <parameter=unit>\nfahrenheit\n</parameter>\n</function>";
    assert!(accepts(&g, body));
    assert_eq!(
        parse(&tools, body),
        vec![(
            "get_weather".into(),
            json!({"city": "<b>Oslo</b>\nNorway", "unit": "fahrenheit"})
        )]
    );
    // A tool that is not strict takes any parameters.
    let body = "\n<function=search>\n<parameter=q>\nrust\n</parameter>\n\
                <parameter=extra>\n1\n</parameter>\n</function>\n";
    assert!(accepts(&g, body));
    // The parser reads a parameter the schema does not type as a string.
    assert_eq!(parse(&tools, body)[0].1, json!({"q": "rust", "extra": "1"}));
    assert!(accepts(&g, "\n<function=search>\n</function>\n"));
}

#[test]
fn calls_off_the_schema_are_rejected() {
    let g = compile(&[weather(), free_tool()]).unwrap();
    let call = |params: &str| format!("\n<function=get_weather>\n{params}</function>\n");
    let city = "<parameter=city>\nParis\n</parameter>\n";
    let unit = "<parameter=unit>\ncelsius\n</parameter>\n";
    assert!(accepts(&g, &call(&format!("{city}{unit}"))));
    for bad in [
        "\n<function=get_time>\n</function>\n".to_string(),
        // Missing a required parameter (the call cannot end).
        call(city),
        call(unit),
        // Out of declaration order.
        call(&format!("{unit}{city}")),
        // Not in the enum.
        call(&format!("{city}<parameter=unit>\nkelvin\n</parameter>\n")),
        // Unknown parameter (additionalProperties: false).
        call(&format!("{city}{unit}<parameter=x>\n1\n</parameter>\n")),
        // Not an integer, not a boolean.
        call(&format!(
            "{city}{unit}<parameter=days>\nthree\n</parameter>\n"
        )),
        call(&format!(
            "{city}{unit}<parameter=alerts>\nyes\n</parameter>\n"
        )),
        // Too many items.
        call(&format!(
            "{city}{unit}<parameter=tags>\n[\"a\",\"b\",\"c\"]\n</parameter>\n"
        )),
        // A closing tag inside raw text would cut the value or the call short.
        call(&format!(
            "<parameter=city>\nParis</parameter>\n</parameter>\n{unit}"
        )),
        call(&format!(
            "<parameter=city>\nParis</tool_call>\n</parameter>\n{unit}"
        )),
        // The same parameter twice.
        call(&format!("{city}{city}{unit}")),
    ] {
        assert!(!accepts(&g, &bad), "accepted {bad:?}");
    }
    // Only the call's end token ends it: not EOS, not end of turn.
    let mut m = g.matcher();
    feed(&mut m, &call(&format!("{city}{unit}"))).unwrap();
    let tok = tokenizer();
    for t in [tok.eos_id().unwrap(), tok.token_id("<|im_end|>").unwrap()] {
        assert!(m.clone().accept_token(t).is_err());
    }
    assert!(m.accept_token(end_token()).is_ok());
}

#[test]
fn unsupported_strict_schemas_are_errors() {
    for params in [
        json!({"type": "object", "properties": {"s": {"type": "string", "pattern": "a+"}}}),
        json!({"anyOf": [{"type": "object"}]}),
        json!({"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}]}}}),
        json!({"type": "array"}),
    ] {
        let tool = Tool {
            parameters: params.clone(),
            ..weather()
        };
        assert!(
            matches!(
                compile(&[tool]),
                Err(GrammarError::Unsupported(_) | GrammarError::Invalid(_))
            ),
            "{params}"
        );
    }
    assert!(compile(&[]).is_err());
}

// ---- random calls ----

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn allowed(m: &Matcher) -> Vec<u32> {
    let mut mask = vec![0u32; m.mask_words()];
    m.fill_mask(&mut mask);
    let mut out = Vec::new();
    for (w, bits) in mask.iter().enumerate() {
        let mut b = *bits;
        while b != 0 {
            out.push((w * 32) as u32 + b.trailing_zeros());
            b &= b - 1;
        }
    }
    out
}

/// A random call the grammar allows: random allowed tokens, steered to markup once past
/// `budget` tokens so the call ends. `None` if it did not end.
fn random_call(g: &Grammar, seed: u64, budget: usize) -> Option<String> {
    let tok = tokenizer();
    let end = end_token();
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut m = g.matcher();
    let mut ids = Vec::new();
    // Past the budget: the rest of the closing markup being written.
    let mut rest: Vec<u8> = Vec::new();
    for step in 0..budget * 4 {
        let all = allowed(&m);
        assert!(!all.is_empty(), "dead end after {:?}", tok.decode(&ids));
        assert_eq!(all.contains(&end), m.is_complete());
        if m.is_complete() && (all.len() == 1 || rng.next().is_multiple_of(3) || step > budget) {
            m.accept_token(end).unwrap();
            assert!(m.is_finished());
            return Some(tok.decode(&ids));
        }
        let mut cands: Vec<u32> = all.into_iter().filter(|&t| t != end).collect();
        if step > budget {
            // Markup that closes a key, a string, an array, a raw value, the call.
            const CLOSERS: [&[u8]; 3] = [
                b">\n\n</parameter>\n</function>\n",
                b"\"]\n</parameter>\n</function>\n",
                b"\n</parameter>\n</function>\n",
            ];
            let targets: Vec<&[u8]> = if rest.is_empty() {
                CLOSERS.to_vec()
            } else {
                vec![&rest]
            };
            // The allowed token that writes the most of one of them.
            let best = targets
                .iter()
                .flat_map(|s| cands.iter().map(move |&t| (s, t)))
                .filter_map(|(s, t)| {
                    let p = tok.token_to_piece(t);
                    (!p.is_empty() && s.starts_with(p)).then(|| (p.len(), s[p.len()..].to_vec(), t))
                })
                .max_by_key(|b| b.0);
            match best {
                Some((_, left, t)) => {
                    rest = left;
                    cands = vec![t];
                }
                None => rest.clear(),
            }
        }
        let t = cands[(rng.next() % cands.len() as u64) as usize];
        m.accept_token(t).unwrap();
        ids.push(t);
    }
    None
}

fn follows(schema: &Value, args: &Map<String, Value>) -> bool {
    let props = schema["properties"].as_object().unwrap();
    let required = schema["required"].as_array().unwrap();
    required
        .iter()
        .all(|k| args.contains_key(k.as_str().unwrap()))
        && args.iter().all(|(k, v)| {
            let Some(p) = props.get(k) else {
                return false;
            };
            if let Some(e) = p.get("enum") {
                return e.as_array().unwrap().contains(v);
            }
            match p["type"].as_str().unwrap() {
                "string" => v.is_string(),
                // Integers past 64 bits parse as floats.
                "integer" => v.as_f64().is_some_and(|f| f.fract() == 0.0),
                "boolean" => v.is_boolean(),
                "array" => v
                    .as_array()
                    .is_some_and(|a| a.len() <= 2 && a.iter().all(Value::is_string)),
                other => panic!("{other}"),
            }
        })
}

#[test]
fn random_calls_parse_and_follow_the_schema() {
    let tools = [weather(), free_tool()];
    let g = compile(&tools).unwrap();
    let schema = &tools[0].parameters;
    let (mut done, mut strict) = (0, 0);
    let runs = 24;
    for seed in 1..=runs {
        let Some(body) = random_call(&g, seed, 40) else {
            continue;
        };
        done += 1;
        let calls = parse(&tools, &body);
        assert_eq!(calls.len(), 1, "{body:?}");
        let (name, args) = &calls[0];
        if name == "get_weather" {
            strict += 1;
            assert!(
                follows(schema, args.as_object().unwrap()),
                "{body:?} -> {args}"
            );
        } else {
            assert_eq!(name, "search");
        }
    }
    assert!(done * 2 >= runs, "only {done}/{runs} calls ended");
    assert!(strict > 0, "no call of the strict tool");
}
