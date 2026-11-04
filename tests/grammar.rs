//! Grammar engine tests against the Qwen 3.5 vocabulary fixture.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use legion::gguf::GgufFile;
use legion::grammar::{Grammar, GrammarCompiler, GrammarError, Matcher};
use legion::tokenizer::Tokenizer;
use serde_json::{Value, json};

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

// ---- a small validator for the supported subset ----

fn resolve<'a>(root: &'a Value, r: &str) -> &'a Value {
    if r == "#" {
        root
    } else {
        root.pointer(&r[1..]).unwrap()
    }
}

fn valid(root: &Value, schema: &Value, v: &Value) -> bool {
    let m = match schema {
        Value::Bool(b) => return *b,
        Value::Object(m) => m,
        _ => unreachable!(),
    };
    if let Some(r) = m.get("$ref") {
        return valid(root, resolve(root, r.as_str().unwrap()), v);
    }
    if let Some(c) = m.get("const") {
        return c == v;
    }
    if let Some(e) = m.get("enum") {
        return e.as_array().unwrap().contains(v);
    }
    for k in ["anyOf", "oneOf"] {
        if let Some(a) = m.get(k) {
            return a.as_array().unwrap().iter().any(|s| valid(root, s, v));
        }
    }
    let Some(t) = m.get("type") else {
        return true;
    };
    let types: Vec<&str> = match t {
        Value::String(s) => vec![s],
        Value::Array(a) => a.iter().map(|x| x.as_str().unwrap()).collect(),
        _ => unreachable!(),
    };
    types.iter().any(|t| match (*t, v) {
        ("string", Value::String(s)) => {
            let n = s.chars().count() as u64;
            n >= m.get("minLength").and_then(Value::as_u64).unwrap_or(0)
                && n <= m
                    .get("maxLength")
                    .and_then(Value::as_u64)
                    .unwrap_or(u64::MAX)
        }
        ("integer", Value::Number(n)) => {
            n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f == 0.0)
        }
        ("number", Value::Number(_)) => true,
        ("boolean", Value::Bool(_)) | ("null", Value::Null) => true,
        ("array", Value::Array(a)) => {
            let n = a.len() as u64;
            n >= m.get("minItems").and_then(Value::as_u64).unwrap_or(0)
                && n <= m
                    .get("maxItems")
                    .and_then(Value::as_u64)
                    .unwrap_or(u64::MAX)
                && a.iter()
                    .all(|x| m.get("items").is_none_or(|s| valid(root, s, x)))
        }
        ("object", Value::Object(o)) => {
            let props = m.get("properties").and_then(Value::as_object);
            let req = m.get("required").and_then(Value::as_array);
            let extra = m.get("additionalProperties");
            req.is_none_or(|r| r.iter().all(|k| o.contains_key(k.as_str().unwrap())))
                && o.iter().all(|(k, x)| match props.and_then(|p| p.get(k)) {
                    Some(s) => valid(root, s, x),
                    None => match extra {
                        Some(Value::Bool(false)) => false,
                        Some(s @ Value::Object(_)) => valid(root, s, x),
                        None if props.is_some() => false,
                        _ => true,
                    },
                })
        }
        _ => false,
    })
}

// ---- random generation ----

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

/// Generate by picking random allowed tokens. Returns the text if the run ended with EOS.
fn generate(g: &Grammar, seed: u64, budget: usize) -> Option<String> {
    let tok = tokenizer();
    let eos = tok.eos_id().unwrap();
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut m = g.matcher();
    let mut ids = Vec::new();
    for step in 0..budget * 4 {
        let all = allowed(&m);
        assert!(
            !all.is_empty(),
            "dead end after {:?} (complete {})",
            tok.decode(&ids),
            m.is_complete()
        );
        assert_eq!(all.contains(&eos), m.is_complete());
        let mut cands: Vec<u32> = all
            .iter()
            .copied()
            .filter(|&t| t != eos && Some(t) != tok.eot_id())
            .collect();
        if m.is_complete() && (cands.is_empty() || rng.next().is_multiple_of(4) || step > budget) {
            m.accept_token(eos).unwrap();
            assert!(m.is_finished());
            let text = tok.decode(&ids);
            return Some(text);
        }
        if step > budget {
            // Steer to closers so the document can finish.
            let closers: Vec<u32> = cands
                .iter()
                .copied()
                .filter(|&t| matches!(tok.token_to_piece(t).first(), Some(b'"' | b'}' | b']')))
                .collect();
            if !closers.is_empty() {
                cands = closers;
            }
        }
        let t = cands[(rng.next() % cands.len() as u64) as usize];
        m.accept_token(t).unwrap();
        ids.push(t);
    }
    None
}

fn check_schema(schema: Value, runs: u64) {
    let g = compiler().compile_json_schema(&schema).unwrap();
    let mut done = 0;
    for seed in 1..=runs {
        if let Some(text) = generate(&g, seed, 150) {
            let v: Value =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("not JSON ({e}): {text:?}"));
            assert!(
                valid(&schema, &schema, &v),
                "invalid against {schema}: {text}"
            );
            done += 1;
        }
    }
    assert!(
        done * 2 >= runs,
        "only {done}/{runs} runs finished for {schema}"
    );
}

#[test]
fn random_generation_validates() {
    let schemas = vec![
        json!({"type": "string", "minLength": 2, "maxLength": 6}),
        json!({"type": "integer"}),
        json!({"type": "number"}),
        json!({"type": "boolean"}),
        json!({"type": ["string", "null"], "maxLength": 5}),
        json!({"enum": ["red", "green", 3, null, {"a": [1, 2]}]}),
        json!({"const": {"k": "v"}}),
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 12},
                "age": {"type": "integer"},
                "email": {"type": "string", "maxLength": 8},
                "tags": {"type": "array", "items": {"type": "string", "maxLength": 4}, "maxItems": 3},
                "kind": {"enum": ["a", "b"]}
            },
            "required": ["name", "age"],
            "additionalProperties": false
        }),
        json!({"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 4}),
        json!({"type": "array", "items": {"type": "boolean"}, "minItems": 1}),
        json!({
            "anyOf": [
                {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
                {"type": "array", "items": {"enum": [1, 2, 3]}, "maxItems": 5},
                {"type": "null"}
            ]
        }),
        json!({
            "$defs": {"pt": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}},
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/pt"}, "b": {"type": "array", "items": {"$ref": "#/$defs/pt"}, "maxItems": 2}},
            "required": ["a"]
        }),
        json!({
            "type": "object",
            "properties": {"children": {"type": "array", "items": {"$ref": "#"}, "maxItems": 2}, "v": {"type": "boolean"}},
            "required": ["v"]
        }),
        json!({"type": "object", "additionalProperties": {"type": "integer"}}),
        json!({"type": "object", "properties": {"o": {"type": "string"}}}),
        json!({"type": "array", "items": {"type": "object", "properties": {"s": {"type": "string", "maxLength": 3}}, "required": ["s"]}, "maxItems": 3}),
    ];
    for s in schemas {
        check_schema(s, 14);
    }
}

#[test]
fn json_object_generates_objects() {
    let g = compiler().compile_json_object().unwrap();
    let mut done = 0;
    for seed in 1..=30 {
        if let Some(text) = generate(&g, seed, 200) {
            let v: Value = serde_json::from_str(&text).unwrap();
            assert!(v.is_object(), "{text}");
            done += 1;
        }
    }
    assert!(done >= 15, "only {done} finished");
}

#[test]
fn hand_written_documents_are_accepted_and_bad_ones_rejected() {
    let schema = json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
        "required": ["a"]
    });
    let g = compiler().compile_json_schema(&schema).unwrap();
    let tok = tokenizer();
    let run = |text: &str| -> Option<bool> {
        let mut m = g.matcher();
        for id in tok.encode(text, false) {
            m.accept_token(id).ok()?;
        }
        Some(m.is_complete())
    };
    assert_eq!(run(r#"{"a": 1, "b": "xé\n"}"#), Some(true));
    assert_eq!(run("{\n  \"a\": -12\n}"), Some(true));
    assert_eq!(run(r#"{"a":1}"#), Some(true));
    assert_eq!(run(r#"{"a":1"#), Some(false));
    assert_eq!(run(r#"{"b":"x"}"#), None);
    assert_eq!(run(r#"{"a":1.5}"#), None);
    assert_eq!(run(r#"{"a":1,"c":2}"#), None);
    assert_eq!(run(r#"{"a":01}"#), None);
    assert_eq!(run(r#"{"a":1,"b":"\ud800"}"#), None);
    assert_eq!(run(r#"[1]"#), None);
}

#[test]
fn eos_only_when_complete_and_disallowed_tokens_rejected() {
    let tok = tokenizer();
    let eos = tok.eos_id().unwrap();
    let g = compiler()
        .compile_json_schema(&json!({"type": "array", "items": {"type": "boolean"}, "maxItems": 1}))
        .unwrap();
    let mut m = g.matcher();
    assert!(!m.is_complete());
    assert_eq!(m.accept_token(eos), Err(legion::grammar::Rejected(eos)));
    let mut rng = Rng(7);
    for text in ["[", "true", "]"] {
        // Every token outside the mask is rejected and leaves the state alone.
        let mask = allowed(&m);
        for _ in 0..200 {
            let t = (rng.next() % tok.vocab_size() as u64) as u32;
            let mut probe = m.clone();
            assert_eq!(
                probe.accept_token(t).is_ok(),
                mask.contains(&t),
                "token {t}"
            );
        }
        for id in tok.encode(text, false) {
            m.accept_token(id).unwrap();
        }
    }
    assert!(m.is_complete());
    // Nothing but EOS can follow a closed array.
    let mut stop = vec![eos, tok.eot_id().unwrap()];
    stop.sort_unstable();
    assert_eq!(allowed(&m), stop);
    m.accept_token(eos).unwrap();
    assert!(m.is_finished());
    assert!(allowed(&m).is_empty());
    assert!(m.accept_token(eos).is_err());
    // Special tokens are never produced.
    let mut m2 = g.matcher();
    let im_start = tok.token_id("<|im_start|>").unwrap();
    assert!(m2.accept_token(im_start).is_err());
}

#[test]
fn unsupported_and_invalid_schemas_return_errors() {
    let c = compiler();
    for s in [
        json!({"type": "string", "pattern": "^a+$"}),
        json!({"type": "integer", "minimum": 3}),
        json!({"allOf": [{"type": "string"}]}),
        json!({"not": {"type": "string"}}),
        json!({"type": "array", "uniqueItems": true}),
        json!({"type": "object", "properties": {"a": {}}, "additionalProperties": {"type": "string"}}),
        json!({"$ref": "https://example.com/x.json"}),
        json!({"type": "object", "properties": {"a": {}}, "required": ["zzz"]}),
    ] {
        assert!(
            matches!(c.compile_json_schema(&s), Err(GrammarError::Unsupported(_))),
            "{s}"
        );
    }
    for s in [
        json!(3),
        json!({"type": "wat"}),
        json!({"type": 4}),
        json!({"$ref": "#/nowhere"}),
        json!({"minLength": "x", "type": "string"}),
        json!({"enum": 1}),
    ] {
        assert!(
            matches!(c.compile_json_schema(&s), Err(GrammarError::Invalid(_))),
            "{s}"
        );
    }
    assert!(matches!(
        c.compile_json_schema(&json!(false)),
        Err(GrammarError::Unsatisfiable)
    ));
    assert!(matches!(
        c.compile_json_schema(&json!({"type": "array", "minItems": 3, "maxItems": 2})),
        Err(GrammarError::Unsatisfiable)
    ));
    assert!(matches!(
        c.compile_json_schema(&json!({"type": "array", "minItems": 100000000})),
        Err(GrammarError::TooLarge(_))
    ));
}

#[test]
fn recursion_is_bounded_not_fatal() {
    let schema = json!({
        "$defs": {"n": {"type": "object", "properties": {"next": {"$ref": "#/$defs/n"}}, "required": ["next"]}},
        "$ref": "#/$defs/n"
    });
    // Required infinite recursion has no finite document.
    assert!(matches!(
        compiler().compile_json_schema(&schema),
        Err(GrammarError::Unsatisfiable)
    ));
}

#[test]
fn utf8_split_across_tokens_stays_valid() {
    let g = compiler()
        .compile_json_schema(&json!({"type": "string", "minLength": 1, "maxLength": 3}))
        .unwrap();
    for seed in 1..=60 {
        if let Some(text) = generate(&g, seed, 30) {
            let v: Value = serde_json::from_str(&text).unwrap();
            let n = v.as_str().unwrap().chars().count();
            assert!((1..=3).contains(&n));
        }
    }
}

/// Reports mask timings; run with `cargo test --release grammar_timing -- --ignored --nocapture`.
#[test]
#[ignore]
fn grammar_timing() {
    let tok = tokenizer();
    println!("vocab {}", tok.vocab_size());
    let t = Instant::now();
    let c = GrammarCompiler::new(tok);
    println!("trie build: {:?}", t.elapsed());
    let schemas = [
        (
            "object",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"}, "age": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "address": {"type": "object", "properties": {"city": {"type": "string"}, "zip": {"type": "string"}}, "required": ["city"]}
                },
                "required": ["name", "age"]
            }),
        ),
        ("json_object", Value::Null),
    ];
    for (name, schema) in schemas {
        let t = Instant::now();
        let g = if schema.is_null() {
            c.compile_json_object().unwrap()
        } else {
            c.compile_json_schema(&schema).unwrap()
        };
        println!(
            "{name}: compile {:?}, {} NFA states",
            t.elapsed(),
            g.nfa_states()
        );
        let mut cold = Vec::new();
        let mut warm = Vec::new();
        let mut mask = vec![0u32; g.mask_words()];
        for seed in 1..=20 {
            let mut rng = Rng(seed * 77 + 1);
            let mut m = g.matcher();
            for _ in 0..300 {
                let before = g.dfa_states();
                let t = Instant::now();
                m.fill_mask(&mut mask);
                let d = t.elapsed();
                if g.dfa_states() > before || d.as_micros() > 500 {
                    cold.push(d);
                } else {
                    warm.push(d);
                }
                let all = allowed(&m);
                let cands: Vec<u32> = all
                    .into_iter()
                    .filter(|&x| Some(x) != tok.eos_id())
                    .collect();
                if cands.is_empty() {
                    break;
                }
                m.accept_token(cands[(rng.next() % cands.len() as u64) as usize])
                    .unwrap();
            }
        }
        cold.sort();
        warm.sort();
        let med = |v: &Vec<std::time::Duration>| v.get(v.len() / 2).copied();
        println!(
            "{name}: {} DFA states; fills over 50us (first visit of a state) n={} median {:?} max {:?}; other fills n={} median {:?} p99 {:?}",
            g.dfa_states(),
            cold.len(),
            med(&cold),
            cold.last(),
            warm.len(),
            med(&warm),
            warm.get(warm.len() * 99 / 100)
        );
    }
}
