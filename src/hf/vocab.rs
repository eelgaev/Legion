//! `tokenizer.json` / `tokenizer_config.json` as the `tokenizer.ggml.*` metadata llama.cpp's
//! converter writes for GPT-2 style BPE vocabularies: `get_vocab_base` / `_set_vocab_gpt2` in
//! `conversion/base.py` and `SpecialVocab` in `gguf-py/gguf/vocab.py` (llama.cpp, MIT).

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value as Json;

use super::Builder;
use crate::gguf::Value;

/// llama.cpp's token types.
const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;

/// Special token types in the order the converter checks them, with their GGUF key stems.
const SPECIAL: [(&str, &str); 7] = [
    ("bos", "bos_token_id"),
    ("eos", "eos_token_id"),
    ("unk", "unknown_token_id"),
    ("sep", "seperator_token_id"),
    ("pad", "padding_token_id"),
    ("cls", "cls_token_id"),
    ("mask", "mask_token_id"),
];

fn read_json(path: &Path) -> Result<Option<Json>, String> {
    match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// The id of the added token `content` in `dir`'s `tokenizer.json`.
pub(super) fn added_token_id(dir: &Path, content: &str) -> Result<Option<u64>, String> {
    let Some(tok) = read_json(&dir.join("tokenizer.json"))? else {
        return Ok(None);
    };
    Ok(tok
        .get("added_tokens")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .find(|a| a.get("content").and_then(Json::as_str) == Some(content))
        .and_then(|a| a.get("id").and_then(Json::as_u64)))
}

/// The converter's `does_token_look_special`.
fn looks_special(t: &str) -> bool {
    matches!(t, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (t.starts_with("<|") && t.ends_with("|>"))
        || (t.starts_with("<｜") && t.ends_with("｜>"))
        || (t.starts_with("<unused") && t.ends_with('>'))
}

/// Writes the GPT-2 BPE vocabulary of the checkpoint in `dir`. `vocab_size` is the model's
/// (`config.json`), which may exceed the tokenizer's: the gap is `[PAD{i}]` tokens.
pub(super) fn gpt2(
    b: &mut Builder,
    dir: &Path,
    config: &Json,
    vocab_size: u64,
    pre: &str,
) -> Result<(), String> {
    let tok = read_json(&dir.join("tokenizer.json"))?
        .ok_or("no tokenizer.json (only tokenizer.json vocabularies are supported)")?;
    let model = tok.get("model").ok_or("tokenizer.json: no `model`")?;
    if model.get("type").and_then(Json::as_str) != Some("BPE") {
        return Err("tokenizer.json: only BPE models are supported".into());
    }

    // id -> token: the base vocab plus added tokens (which may also sit in the base vocab).
    let mut by_id: HashMap<u64, String> = HashMap::new();
    for (t, id) in model
        .get("vocab")
        .and_then(Json::as_object)
        .ok_or("tokenizer.json: no `model.vocab`")?
    {
        let id = id.as_u64().ok_or("tokenizer.json: non-integer vocab id")?;
        by_id.insert(id, t.clone());
    }
    struct Added {
        special: bool,
    }
    let mut added: HashMap<String, Added> = HashMap::new();
    let added_list: Vec<Json> = tok
        .get("added_tokens")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    for a in &added_list {
        let (Some(id), Some(content)) = (
            a.get("id").and_then(Json::as_u64),
            a.get("content").and_then(Json::as_str),
        ) else {
            continue;
        };
        by_id.insert(id, content.to_string());
        added.insert(
            content.to_string(),
            Added {
                special: a.get("special").and_then(Json::as_bool).unwrap_or(false),
            },
        );
    }
    // transformers also registers `tokenizer_config.json`'s `added_tokens_decoder`, which can
    // hold tokens `tokenizer.json` lacks (Qwen3.5's audio/TTS markers).
    let tc = read_json(&dir.join("tokenizer_config.json"))?;
    if let Some(dec) = tc
        .as_ref()
        .and_then(|t| t.get("added_tokens_decoder"))
        .and_then(Json::as_object)
    {
        for (id, a) in dec {
            let (Ok(id), Some(content)) =
                (id.parse::<u64>(), a.get("content").and_then(Json::as_str))
            else {
                continue;
            };
            by_id.insert(id, content.to_string());
            added.insert(
                content.to_string(),
                Added {
                    special: a.get("special").and_then(Json::as_bool).unwrap_or(false),
                },
            );
        }
    }
    if let Some(max) = by_id.keys().max()
        && *max >= vocab_size
    {
        return Err(format!(
            "tokenizer has id {max}, beyond the model's vocab_size {vocab_size}"
        ));
    }

    let mut tokens = Vec::with_capacity(vocab_size as usize);
    let mut types = Vec::with_capacity(vocab_size as usize);
    for i in 0..vocab_size {
        match by_id.get(&i) {
            None => {
                tokens.push(Value::String(format!("[PAD{i}]")));
                types.push(Value::I32(UNUSED));
            }
            Some(t) => match added.get(t) {
                Some(a) if a.special || looks_special(t) => {
                    tokens.push(Value::String(t.clone()));
                    types.push(Value::I32(CONTROL));
                }
                Some(_) => {
                    tokens.push(Value::String(t.replace('\u{2581}', " ")));
                    types.push(Value::I32(USER_DEFINED));
                }
                None => {
                    tokens.push(Value::String(t.clone()));
                    types.push(Value::I32(NORMAL));
                }
            },
        }
    }
    b.set("tokenizer.ggml.model", Value::String("gpt2".into()))?;
    b.set("tokenizer.ggml.pre", Value::String(pre.into()))?;
    b.set("tokenizer.ggml.tokens", Value::Array(tokens))?;
    b.set("tokenizer.ggml.token_type", Value::Array(types))?;

    // Merges: "a b" strings, or [a, b] pairs whose spaces are encoded as U+0120's neighbour.
    let merges: Vec<Value> = match model.get("merges").and_then(Json::as_array) {
        Some(m) => m
            .iter()
            .map(|e| match e {
                Json::String(s) => Ok(s.clone()),
                Json::Array(pair) if pair.len() == 2 => {
                    let enc = |p: &Json| -> Result<String, String> {
                        let s = p.as_str().ok_or("tokenizer.json: bad merge pair")?;
                        Ok(s.chars()
                            .map(|c| {
                                if c == ' ' {
                                    char::from_u32(' ' as u32 + 256).unwrap()
                                } else {
                                    c
                                }
                            })
                            .collect())
                    };
                    Ok(format!("{} {}", enc(&pair[0])?, enc(&pair[1])?))
                }
                _ => Err("tokenizer.json: unknown merges format".to_string()),
            })
            .map(|r| r.map(Value::String))
            .collect::<Result<_, _>>()?,
        None => Vec::new(),
    };
    if !merges.is_empty() {
        b.set("tokenizer.ggml.merges", Value::Array(merges))?;
    }

    // add_bos / add_eos from the post-processor (the ByteLevel case is all this supports).
    let mut add: HashMap<&str, bool> = HashMap::new();
    if let Some(pp) = tok.get("post_processor").filter(|p| !p.is_null()) {
        let list: Vec<&Json> = match pp.get("processors").and_then(Json::as_array) {
            Some(l) => l.iter().collect(),
            None => vec![pp],
        };
        for p in list {
            match p.get("type").and_then(Json::as_str) {
                Some("ByteLevel") => {
                    add.entry("bos").or_insert(false);
                    add.entry("eos").or_insert(false);
                }
                Some("TemplateProcessing") | Some("RobertaProcessing") => {
                    return Err(
                        "tokenizer.json: TemplateProcessing/RobertaProcessing post-processors are not supported yet"
                            .into(),
                    );
                }
                _ => {}
            }
        }
    }
    normalizer(b, tok.get("normalizer"))?;

    // Special ids: tokenizer_config.json first (by content, among the added tokens), then
    // config.json (root, then text_config). The first value found for a type wins.
    let mut ids: HashMap<&str, u64> = HashMap::new();
    let set_id = |ids: &mut HashMap<&str, u64>, typ: &'static str, id: Option<u64>| {
        if let Some(id) = id
            && id < vocab_size
        {
            ids.entry(typ).or_insert(id);
        }
    };
    let mut template: Option<String> = None;
    if let Some(tc) = &tc {
        template = match tc.get("chat_template") {
            Some(Json::String(s)) => Some(s.clone()),
            _ => None,
        };
        for (typ, _) in SPECIAL {
            if let Some(v) = tc.get(format!("add_{typ}_token")).and_then(Json::as_bool) {
                add.entry(typ).or_insert(v);
            }
            let content = match tc.get(format!("{typ}_token")) {
                Some(Json::String(s)) => Some(s.as_str()),
                Some(Json::Object(o)) => o.get("content").and_then(Json::as_str),
                _ => None,
            };
            let Some(content) = content else { continue };
            let id = added_list
                .iter()
                .find(|a| a.get("content").and_then(Json::as_str) == Some(content))
                .and_then(|a| a.get("id").and_then(Json::as_u64));
            set_id(&mut ids, typ, id);
        }
    }
    for (typ, _) in SPECIAL {
        let key = format!("{typ}_token_id");
        let id = config
            .get(&key)
            .or_else(|| config.get("text_config").and_then(|t| t.get(&key)))
            .and_then(Json::as_u64);
        set_id(&mut ids, typ, id);
    }
    for (typ, stem) in SPECIAL {
        if let Some(&id) = ids.get(typ) {
            b.set(&format!("tokenizer.ggml.{stem}"), Value::U32(id as u32))?;
        }
    }
    for typ in ["bos", "eos", "sep"] {
        if let Some(&v) = add.get(typ) {
            b.set(&format!("tokenizer.ggml.add_{typ}_token"), Value::Bool(v))?;
        }
    }

    if template.is_none() {
        template = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(s) => Some(s),
            Err(_) => read_json(&dir.join("chat_template.json"))?.and_then(|j| {
                j.get("chat_template")
                    .and_then(Json::as_str)
                    .map(str::to_string)
            }),
        };
    }
    if let Some(t) = template {
        b.set("tokenizer.chat_template", Value::String(t))?;
    }
    Ok(())
}

/// The converter's `_parse_normalizer`.
fn normalizer(b: &mut Builder, n: Option<&Json>) -> Result<(), String> {
    let Some(n) = n.filter(|n| !n.is_null()) else {
        return Ok(());
    };
    let flag = |b: &mut Builder, k: &str, v: bool| {
        b.set(&format!("tokenizer.ggml.normalizer.{k}"), Value::Bool(v))
    };
    match n.get("type").and_then(Json::as_str) {
        Some("Lowercase") => flag(b, "lowercase", true)?,
        Some("NFC") => flag(b, "nfc", true)?,
        Some("StripAccents") => flag(b, "strip_accents", true)?,
        Some("BertNormalizer") => {
            if let Some(v) = n.get("lowercase").and_then(Json::as_bool) {
                flag(b, "lowercase", v)?;
            }
            if let Some(v) = n.get("strip_accents").and_then(Json::as_bool) {
                flag(b, "strip_accents", v)?;
            }
        }
        Some("Sequence") => {
            for m in n
                .get("normalizers")
                .and_then(Json::as_array)
                .into_iter()
                .flatten()
            {
                normalizer(b, Some(m))?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests;
