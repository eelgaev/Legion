mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::{Value, json};

/// The mock replies "You said: hi there" as the tokens "You", " said:", " hi", " there".
const TOKENS: [&str; 4] = ["You", " said:", " hi", " there"];

fn assert_close(v: &Value, want: f64) {
    let got = v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"));
    assert!((got - want).abs() < 1e-6, "{got} != {want}");
}

/// Every `choices[0]` of the data chunks of a chat/completions stream.
fn stream_choices(body: &str) -> Vec<Value> {
    sse_events(body)
        .into_iter()
        .filter(|(_, d)| d != "[DONE]")
        .filter_map(|(_, d)| {
            serde_json::from_str::<Value>(&d).unwrap()["choices"][0]
                .clone()
                .into()
        })
        .filter(|c: &Value| !c.is_null())
        .collect()
}

async fn chat_json(extra: Value) -> (StatusCode, Value) {
    json_of(post("/v1/chat/completions", chat(extra))).await
}

fn check_chat_entry(e: &Value, i: usize, top: usize) {
    assert_eq!(e["token"], TOKENS[i]);
    assert_close(&e["logprob"], -0.1 * (i % 5) as f64);
    let bytes: Vec<u64> = e["bytes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b.as_u64().unwrap())
        .collect();
    assert_eq!(bytes, TOKENS[i].bytes().map(u64::from).collect::<Vec<_>>());
    let tops = e["top_logprobs"].as_array().unwrap();
    assert_eq!(tops.len(), top);
    for t in tops {
        assert!(t["token"].is_string() && t["logprob"].is_number() && t["bytes"].is_array());
    }
    if top > 0 {
        assert_eq!(tops[0]["token"], TOKENS[i]);
        assert_eq!(
            tops[1.min(top - 1)]["token"],
            if top > 1 { " alt1" } else { TOKENS[i] }
        );
    }
}

#[tokio::test]
async fn chat_off_by_default() {
    let (status, v) = chat_json(json!({})).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(v["choices"][0]["logprobs"].is_null());
    let (_, v) = chat_json(json!({"logprobs": false})).await;
    assert!(v["choices"][0]["logprobs"].is_null());
}

#[tokio::test]
async fn chat_non_stream_shape() {
    let (status, v) = chat_json(json!({"logprobs": true, "top_logprobs": 3})).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let lp = &v["choices"][0]["logprobs"];
    assert!(lp["refusal"].is_null(), "{lp}");
    let content = lp["content"].as_array().unwrap();
    assert_eq!(content.len(), TOKENS.len());
    for (i, e) in content.iter().enumerate() {
        check_chat_entry(e, i, 3);
    }
    assert_close(&content[0]["top_logprobs"][1]["logprob"], -1.0);
    assert_close(&content[0]["top_logprobs"][2]["logprob"], -1.5);
    let joined: String = content
        .iter()
        .map(|e| e["token"].as_str().unwrap())
        .collect();
    assert_eq!(
        joined,
        v["choices"][0]["message"]["content"].as_str().unwrap()
    );
}

#[tokio::test]
async fn chat_logprobs_without_top() {
    // `logprobs: true` alone: the chosen token only, with an empty alternatives list.
    let (_, v) = chat_json(json!({"logprobs": true})).await;
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 4);
    check_chat_entry(&content[2], 2, 0);
    let (_, v) = chat_json(json!({"logprobs": true, "top_logprobs": 20})).await;
    check_chat_entry(&v["choices"][0]["logprobs"]["content"][3], 3, 20);
}

#[tokio::test]
async fn chat_validation() {
    let (status, v) = chat_json(json!({"logprobs": true, "top_logprobs": 21})).await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "top_logprobs");
    // top_logprobs needs logprobs: true, as at OpenAI.
    let (status, v) = chat_json(json!({"top_logprobs": 2})).await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    let (status, v) = chat_json(json!({"logprobs": false, "top_logprobs": 2})).await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_stream_shape() {
    let (status, body) = send(post(
        "/v1/chat/completions",
        chat(json!({"stream": true, "logprobs": true, "top_logprobs": 2})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK);
    let choices = stream_choices(&body);
    // The role chunk and the finish chunk carry no logprobs.
    assert!(choices[0]["logprobs"].is_null());
    assert!(choices.last().unwrap()["logprobs"].is_null());
    let mut i = 0;
    for c in &choices {
        let Some(content) = c["logprobs"]["content"].as_array() else {
            continue;
        };
        assert!(c["logprobs"]["refusal"].is_null());
        // Each chunk's logprobs describe that chunk's own text.
        let text: String = content
            .iter()
            .map(|e| e["token"].as_str().unwrap())
            .collect();
        assert_eq!(text, c["delta"]["content"].as_str().unwrap());
        for e in content {
            check_chat_entry(e, i, 2);
            i += 1;
        }
    }
    assert_eq!(i, TOKENS.len());
}

#[tokio::test]
async fn chat_stream_without_logprobs_has_none() {
    let (_, body) = send(post("/v1/chat/completions", chat(json!({"stream": true})))).await;
    assert!(
        stream_choices(&body)
            .iter()
            .all(|c| c["logprobs"].is_null())
    );
}

/// Stop strings that never match: text is held back at the end, and every token still gets
/// its entry, in order.
#[tokio::test]
async fn stop_hold_back_without_match_keeps_all_logprobs() {
    // " there" is held back after " hi" because " thereX" could still match.
    let body = chat(json!({"logprobs": true, "stop": [" thereX"]}));
    let (_, v) = json_of(post("/v1/chat/completions", body)).await;
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 4);
    assert_eq!(v["choices"][0]["message"]["content"], "You said: hi there");
}

/// A stop string that swallows whole tokens: their entries are dropped with their text.
#[tokio::test]
async fn stop_string_drops_logprobs_of_stopped_tokens() {
    let body = chat(json!({"logprobs": true, "stop": [" there"]}));
    let (_, v) = json_of(post("/v1/chat/completions", body)).await;
    let c = &v["choices"][0];
    assert_eq!(c["message"]["content"], "You said: hi");
    assert_eq!(c["finish_reason"], "stop");
    let tokens: Vec<&str> = c["logprobs"]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["token"].as_str().unwrap())
        .collect();
    assert_eq!(tokens, ["You", " said:", " hi"]);
}

/// A stop string that starts in the middle of a token: the token is kept (it contributed
/// text), the tokens after it are dropped.
#[tokio::test]
async fn stop_string_inside_a_token_keeps_the_straddling_token() {
    // Text "You said: hi there"; stop "d: h" begins inside " said:" and ends inside " hi".
    let body = chat(json!({"logprobs": true, "stop": ["d: h"]}));
    let (_, v) = json_of(post("/v1/chat/completions", body)).await;
    let c = &v["choices"][0];
    assert_eq!(c["message"]["content"], "You sai");
    let tokens: Vec<&str> = c["logprobs"]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["token"].as_str().unwrap())
        .collect();
    assert_eq!(tokens, ["You", " said:"]);
}

#[tokio::test]
async fn stream_stop_string_stays_aligned() {
    // Streaming: the held-back " hi" is released (with its logprobs) only once " there"
    // proves it is not part of the stop string, and the stopped tokens never show up.
    let body = chat(json!({"stream": true, "logprobs": true, "stop": [" thereX", "d: hi t"]}));
    let (_, body) = send(post("/v1/chat/completions", body)).await;
    let mut text = String::new();
    let mut tokens = Vec::new();
    for c in stream_choices(&body) {
        if let Some(t) = c["delta"]["content"].as_str() {
            text.push_str(t);
        }
        for e in c["logprobs"]["content"].as_array().into_iter().flatten() {
            tokens.push(e["token"].as_str().unwrap().to_string());
        }
    }
    // "d: hi t" matches inside " said:" + " hi" + " there": text ends before the "d".
    assert_eq!(text, "You sai");
    assert_eq!(tokens, ["You", " said:"]);
}

#[tokio::test]
async fn max_tokens_cut_keeps_logprobs_aligned() {
    let (_, v) = chat_json(json!({"logprobs": true, "max_tokens": 2})).await;
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "length");
    assert_eq!(c["message"]["content"], "You said:");
    assert_eq!(c["logprobs"]["content"].as_array().unwrap().len(), 2);
}

fn completion(extra: Value) -> Value {
    let mut body = json!({"model": "mock", "prompt": "hi there", "max_tokens": 16});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

#[tokio::test]
async fn completions_non_stream_shape() {
    let (status, v) = json_of(post("/v1/completions", completion(json!({"logprobs": 2})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let lp = &v["choices"][0]["logprobs"];
    assert_eq!(lp["tokens"], json!(TOKENS));
    let token_logprobs = lp["token_logprobs"].as_array().unwrap();
    for (i, l) in token_logprobs.iter().enumerate() {
        assert_close(l, -0.1 * (i % 5) as f64);
    }
    // "You said: hi there"
    assert_eq!(lp["text_offset"], json!([0, 3, 9, 12]));
    let tops = lp["top_logprobs"].as_array().unwrap();
    assert_eq!(tops.len(), 4);
    for (i, t) in tops.iter().enumerate() {
        let t = t.as_object().unwrap();
        assert_eq!(t.len(), 2, "{t:?}");
        assert_close(&t[TOKENS[i]], -0.1 * (i % 5) as f64);
        assert_close(&t[" alt1"], -1.0);
    }
    assert_eq!(v["choices"][0]["text"], "You said: hi there");
}

#[tokio::test]
async fn completions_off_zero_and_validation() {
    let (_, v) = json_of(post("/v1/completions", completion(json!({})))).await;
    assert!(v["choices"][0]["logprobs"].is_null());
    // logprobs: 0 -> the chosen token alone in each top_logprobs object.
    let (status, v) = json_of(post("/v1/completions", completion(json!({"logprobs": 0})))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    let tops = v["choices"][0]["logprobs"]["top_logprobs"]
        .as_array()
        .unwrap();
    assert_eq!(
        tops[1].as_object().unwrap().keys().collect::<Vec<_>>(),
        [" said:"]
    );
    let (status, v) = json_of(post("/v1/completions", completion(json!({"logprobs": 21})))).await;
    assert_openai_error(status, &v, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "logprobs");
    let (status, _) = json_of(post("/v1/completions", completion(json!({"logprobs": 20})))).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn completions_echo_shifts_text_offset() {
    let body = completion(json!({"logprobs": 0, "echo": true}));
    let (_, v) = json_of(post("/v1/completions", body)).await;
    let c = &v["choices"][0];
    assert_eq!(c["text"], "hi thereYou said: hi there");
    assert_eq!(c["logprobs"]["text_offset"], json!([8, 11, 17, 20]));
}

#[tokio::test]
async fn completions_stream_shape() {
    let body = completion(json!({"stream": true, "logprobs": 1}));
    let (_, body) = send(post("/v1/completions", body)).await;
    let mut tokens = Vec::new();
    let mut offsets = Vec::new();
    let mut text = String::new();
    for c in stream_choices(&body) {
        text.push_str(c["text"].as_str().unwrap());
        let lp = &c["logprobs"];
        if lp.is_null() {
            continue;
        }
        let toks = lp["tokens"].as_array().unwrap();
        assert_eq!(toks.len(), lp["token_logprobs"].as_array().unwrap().len());
        assert_eq!(toks.len(), lp["top_logprobs"].as_array().unwrap().len());
        assert_eq!(toks.len(), lp["text_offset"].as_array().unwrap().len());
        tokens.extend(toks.iter().map(|t| t.as_str().unwrap().to_string()));
        offsets.extend(
            lp["text_offset"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o.as_u64().unwrap()),
        );
    }
    assert_eq!(tokens, TOKENS);
    assert_eq!(offsets, [0, 3, 9, 12]);
    assert_eq!(text, "You said: hi there");
}

#[tokio::test]
async fn completions_stop_string_drops_stopped_tokens() {
    let body = completion(json!({"logprobs": 0, "stop": [" there"]}));
    let (_, v) = json_of(post("/v1/completions", body)).await;
    let c = &v["choices"][0];
    assert_eq!(c["text"], "You said: hi");
    assert_eq!(c["logprobs"]["tokens"], json!(["You", " said:", " hi"]));
    assert_eq!(c["logprobs"]["text_offset"], json!([0, 3, 9]));
}

#[tokio::test]
async fn responses_ignores_logprobs() {
    let body = json!({"model": "mock", "input": "hi there", "top_logprobs": 3});
    let (status, v) = json_of(post("/v1/responses", body)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
}
