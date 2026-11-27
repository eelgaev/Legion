use super::{DEAD, Lazy, Nfa, Node, Options, RawStep, RawText};

/// Runs `text` through the automaton: `Some(i)`: `end` completes at byte `i`; `None`: a
/// forbidden string completed first, or `end` never did.
fn ends_at(raw: &RawText<'_>, text: &[u8]) -> Option<usize> {
    let mut j = 0;
    for (i, &b) in text.iter().enumerate() {
        if !b.is_ascii() {
            j = 0;
            continue;
        }
        match raw.step(j, b) {
            RawStep::To(k) => j = k,
            RawStep::End => return Some(i),
            RawStep::Dead => return None,
        }
    }
    None
}

/// The same by searching the text directly.
fn reference(end: &[u8], forbid: &[Vec<u8>], text: &[u8]) -> Option<usize> {
    for i in 0..text.len() {
        let seen = &text[..=i];
        if seen.ends_with(end) {
            return Some(i);
        }
        if forbid.iter().any(|f| seen.ends_with(f)) {
            return None;
        }
    }
    None
}

#[test]
fn raw_text_ends_at_the_first_terminator() {
    let forbid = vec![b"</parameter>".to_vec(), b"</tool_call>".to_vec()];
    let raw = RawText::new(b"\n</parameter>", &forbid);
    let at = |s: &str| ends_at(&raw, s.as_bytes());
    assert_eq!(at("abc\n</parameter>"), Some(15));
    assert_eq!(at("\n</parameter>"), Some(12));
    // A partial terminator, then the real one.
    assert_eq!(at("a\n</param\n\n</parameter>"), Some(22));
    assert_eq!(at("x <b>y</b>\n</parameter>"), Some(22));
    // Multi-byte text in between resets the match.
    assert_eq!(at("é\n</parameter>"), Some(14));
    // Forbidden: a closing tag not on its own line, or the call's end.
    assert_eq!(at("abc</parameter>"), None);
    assert_eq!(at("a</tool_call>\n</parameter>"), None);
    assert_eq!(at("abc"), None);
}

#[test]
fn raw_text_matches_a_direct_search() {
    let forbid = vec![b"</parameter>".to_vec(), b"</tool_call>".to_vec()];
    let end = b"\n</parameter>";
    let raw = RawText::new(end, &forbid);
    // Texts over the strings' own bytes, where partial matches overlap the most.
    let alphabet = b"\n</parametrxo_cl>";
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..20_000 {
        let len = (seed % 40) as usize;
        let mut text = Vec::with_capacity(len + end.len());
        for _ in 0..len {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            text.push(alphabet[(seed % alphabet.len() as u64) as usize]);
        }
        if seed.is_multiple_of(2) {
            text.extend_from_slice(end);
        }
        assert_eq!(
            ends_at(&raw, &text),
            reference(end, &forbid, &text),
            "{:?}",
            String::from_utf8_lossy(&text)
        );
    }
}

#[test]
fn raw_text_without_forbidden_strings_is_kmp() {
    let raw = RawText::new(b">", &[]);
    assert_eq!(raw.states(), 1);
    assert_eq!(raw.step(0, b'a'), RawStep::To(0));
    assert_eq!(raw.step(0, b'>'), RawStep::End);
    let raw = RawText::new(b"aab", &[]);
    assert_eq!(ends_at(&raw, b"aaab"), Some(3));
    assert_eq!(ends_at(&raw, b"abaab"), Some(4));
}

/// What [`Node::Name`] should take: a name the tool parser reads back as it is (its trim is
/// `str::trim`) that is not excluded.
fn is_name(exclude: &[Vec<u8>], text: &[u8]) -> bool {
    let Ok(s) = std::str::from_utf8(text) else {
        return false;
    };
    !s.is_empty()
        && !s.contains(['\0', '<', '>', '\n'])
        && s.trim() == s
        && !exclude.iter().any(|e| e == text)
}

#[test]
fn name_automaton_matches_the_parsers_names() {
    let exclude: Vec<Vec<u8>> = ["w", "wx", "a w", "é", "\u{a0}x", "中😀", "<w"]
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let (nfa, start) = Nfa::build(
        &Node::Name {
            exclude: exclude.clone(),
        },
        &Options::default(),
    )
    .unwrap();
    let mut lazy = Lazy::new(&nfa, start);
    let mut takes = |text: &[u8]| {
        let mut s = 0;
        for &b in text {
            s = lazy.step(&nfa, s, b);
            if s == DEAD {
                return false;
            }
        }
        lazy.states[s as usize].accepting
    };
    // Whole characters, whitespace of every width, and stray or broken UTF-8 bytes.
    let pieces: &[&[u8]] = &[
        b"a",
        b"w",
        b"x",
        b" ",
        b"\t",
        b"\r",
        b"<",
        b">",
        b"\n",
        b"\0",
        "é".as_bytes(),
        "中".as_bytes(),
        "😀".as_bytes(),
        "\u{85}".as_bytes(),
        "\u{a0}".as_bytes(),
        "\u{1680}".as_bytes(),
        "\u{2000}".as_bytes(),
        "\u{200b}".as_bytes(),
        "\u{2029}".as_bytes(),
        "\u{3000}".as_bytes(),
        b"\xC2",
        b"\xE2\x80",
        b"\x80",
        b"\xED\xA0\x80",
        b"\xF4\x90\x80\x80",
        b"\xC0\xAF",
    ];
    for name in &exclude {
        assert!(!takes(name), "{:?}", String::from_utf8_lossy(name));
    }
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..20_000 {
        let len = (seed % 6) as usize;
        let mut text = Vec::new();
        for _ in 0..len {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            text.extend_from_slice(pieces[(seed % pieces.len() as u64) as usize]);
        }
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        assert_eq!(
            takes(&text),
            is_name(&exclude, &text),
            "{:?}",
            String::from_utf8_lossy(&text)
        );
    }
}
