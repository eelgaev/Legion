use super::*;

#[test]
fn added_tokens_are_found_by_content() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"added_tokens": [{"id": 7, "content": "<|endoftext|>"}, {"id": 9, "content": "<|user|>"}]}"#,
    )
    .unwrap();
    assert_eq!(added_token_id(dir.path(), "<|user|>").unwrap(), Some(9));
    assert_eq!(added_token_id(dir.path(), "<|observation|>").unwrap(), None);
    let empty = tempfile::tempdir().unwrap();
    assert_eq!(added_token_id(empty.path(), "<|user|>").unwrap(), None);
}
