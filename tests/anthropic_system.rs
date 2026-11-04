mod common;

use axum::http::StatusCode;
use common::*;
use serde_json::json;

#[tokio::test]
async fn inline_system_string_turn_accepted() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "again"},
        ]})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["text"], "You said: again");
}

#[tokio::test]
async fn inline_system_text_blocks_accepted() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [
                {"type": "text", "text": "be brief"},
                {"type": "text", "text": "and kind"},
            ]},
        ]})),
    ))
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["content"][0]["text"], "You said: hi");
}

#[tokio::test]
async fn inline_system_image_block_rejected() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [
                {"type": "text", "text": "be brief"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "eA=="}},
            ]},
        ]})),
    ))
    .await;
    assert_anthropic_error(status, &v);
    assert_eq!(v["error"]["param"], "messages[1].content");
    assert_eq!(
        v["error"]["message"],
        "system messages may only contain text"
    );
}

#[tokio::test]
async fn inline_system_tool_block_rejected() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": [
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
            ]},
        ]})),
    ))
    .await;
    assert_anthropic_error(status, &v);
    assert_eq!(v["error"]["param"], "messages[1].content");
}

#[tokio::test]
async fn unknown_role_still_rejected() {
    let (status, v) = json_of(post(
        "/v1/messages",
        messages(json!({"messages": [
            {"role": "tool", "content": "x"},
        ]})),
    ))
    .await;
    assert_anthropic_error(status, &v);
    assert_eq!(v["error"]["param"], "messages[0].role");
}
