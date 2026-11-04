use super::*;

#[test]
fn byte_table_matches_gpt2() {
    let t = byte_to_char_table();
    assert_eq!(t[b'A' as usize], 'A');
    assert_eq!(t[b' ' as usize], '\u{120}'); // Ġ
    assert_eq!(t[b'\n' as usize], '\u{10A}'); // Ċ
    assert_eq!(t[0], '\u{100}');
    assert_eq!(t[0xAD], '\u{143}');
}

#[test]
fn byte_token_parse() {
    assert_eq!(parse_byte_token("<0x0A>"), Some(10));
    assert_eq!(parse_byte_token("<0xZZ>"), None);
    assert_eq!(parse_byte_token("abc"), None);
}

/// The `llama4` pre-tokenizer (Muse Glimmer's): case runs split, digits by three, `/` kept with
/// the punctuation before it (with the newlines after).
#[test]
fn gpt4o_pre_tokens() {
    let (re, ignore_merges) = pre_regex("llama4").unwrap();
    assert!(!ignore_merges);
    let re = Regex::new(re).unwrap();
    let split = |t: &str| -> Vec<String> {
        re.find_iter(t)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    };
    assert_eq!(
        split("Hello WORLDwide, it's 12345 cases.//\n\nOK"),
        [
            "Hello",
            " WORLDwide",
            ",",
            " it's",
            " ",
            "123",
            "45",
            " cases",
            ".//\n\n",
            "OK"
        ]
    );
    assert_eq!(split("CamelCase"), ["Camel", "Case"]);
}

/// The `deepseek-v3` / `joyai-llm` pre-tokenizer: digits by three, CJK runs, then the third
/// expression on each piece on its own. The expected pieces are what the `tokenizers` library
/// gives with DeepSeek-V4.1-Flash's tokenizer.json (its three `Split` steps).
#[test]
fn deepseek3_pre_tokens() {
    assert_eq!(pre_regex("joyai-llm"), Some((RE_DEEPSEEK3, false)));
    assert_eq!(pre_regex("deepseek-v3"), Some((RE_DEEPSEEK3, false)));
    assert!(pre_splits("qwen35").is_empty());
    let res: Vec<Regex> = pre_splits("joyai-llm")
        .iter()
        .chain([&RE_DEEPSEEK3])
        .map(|re| Regex::new(re).unwrap())
        .collect();
    let split = |t: &str| -> Vec<String> {
        split_pieces(res.iter(), t)
            .into_iter()
            .map(|(s, e)| t[s..e].to_string())
            .collect()
    };
    let cases: [(&str, &[&str]); 12] = [
        (
            "Hello world, it's 12345 cases.\n\nOK",
            &[
                "Hello", " world", ",", " it", "'s", " ", "123", "45", " cases", ".\n\n", "OK",
            ],
        ),
        ("x  \n  y", &["x", "  \n", " ", " y"]),
        ("a\tb  c\n\n d", &["a", "\tb", " ", " c", "\n\n", " d"]),
        (
            "价格是1234元，日本語のテキスト。カタカナ",
            &[
                "价格是",
                "123",
                "4",
                "元",
                "，",
                "日本語のテキスト",
                "。",
                "カタカナ",
            ],
        ),
        (
            "\"quoted\" (paren) -dash x.y",
            &[
                "\"quoted", "\"", " (", "paren", ")", " -", "dash", " x", ".y",
            ],
        ),
        (
            "naïve café 3.14159 über",
            &["naïve", " café", " ", "3", ".", "141", "59", " über"],
        ),
        ("emoji 👍🏽 ok", &["emoji", " 👍🏽", " ok"]),
        (
            "line1\r\nline2   \n",
            &["line", "1", "\r\n", "line", "2", "   \n"],
        ),
        ("a\u{200b}b", &["a", "\u{200b}b"]),
        (
            "fn main() {\n    let x = 42;\n}\n",
            &[
                "fn", " main", "()", " {\n", "   ", " let", " x", " =", " ", "42", ";\n", "}\n",
            ],
        ),
        (
            "##Title\n- item_one: v2.0",
            &[
                "##", "Title", "\n", "-", " item", "_one", ":", " v", "2", ".", "0",
            ],
        ),
        ("", &[]),
    ];
    for (text, want) in cases {
        assert_eq!(split(text), want, "{text:?}");
    }
}
