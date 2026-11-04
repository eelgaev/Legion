use serde_json::json;

use super::*;

/// DeepSeek-V4.1-Flash-1017's Engram config.
fn released() -> Layout {
    Layout {
        layers: vec![1, 14],
        max_ngram: 4,
        heads: 8,
        vocab: 16_000_000,
    }
}

/// The released GGUF's `engram_primes.weight` (per layer 24 columns).
const PRIMES: [u64; 48] = [
    16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183, 16000189,
    16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321, 16000339, 16000381,
    16000393, 16000399, 16000403, 16000409, 16000447, 16000463, 16000477, 16000487, 16000499,
    16000507, 16000511, 16000573, 16000609, 16000627, 16000667, 16000669, 16000693, 16000697,
    16000711, 16000729, 16000759, 16000769, 16000781, 16000799, 16000813, 16000819, 16000841,
    16000877, 16000879, 16000889,
];

/// Its `engram_multipliers.weight` (per layer 4).
const MULTIPLIERS: [i64; 8] = [
    76632096046245,
    4839876093313,
    35959672319349,
    73987337458391,
    67716810739261,
    51510806800915,
    30921347202721,
    82619226485591,
];

#[test]
fn primes_offsets_and_multipliers_match_the_released_gguf() {
    let l = released();
    let p = primes(&l);
    assert_eq!(p.concat(), PRIMES);
    // The rows per layer are `engram_num_embeddings`.
    let rows: Vec<u64> = p.iter().map(|v| v.iter().sum()).collect();
    assert_eq!(rows, [384006168, 384016682]);
    let o = offsets(&p);
    assert_eq!(&o[0][..4], [0, 16000057, 32000136, 48000217]);
    assert_eq!(&o[1][..4], [0, 16000477, 32000964, 48001463]);
    assert_eq!(o[1][23], 368015793);
    let m = multipliers(&l, 99092).unwrap();
    assert_eq!(m.concat(), MULTIPLIERS);
}

#[test]
fn small_primes() {
    let got: Vec<u64> = (0..30).filter(|&n| is_prime(n)).collect();
    assert_eq!(got, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    assert!(is_prime(16000057) && !is_prime(16000059));
    // Buckets above a small base, none twice over n-gram sizes and layers.
    let p = primes(&Layout {
        layers: vec![0, 3],
        max_ngram: 3,
        heads: 2,
        vocab: 10,
    });
    assert_eq!(p, [vec![11, 13, 17, 19], vec![23, 29, 31, 37]]);
    assert_eq!(offsets(&p)[1], [0, 23, 52, 83]);
}

#[test]
fn normalizer_folds_case_accents_width_and_space() {
    for (text, want) in [
        ("Hello", "hello"),
        (" The", "the"),
        ("Café", "cafe"),
        ("ＡＢＣ", "abc"),
        ("ﬁ", "fi"),
        (" ", " "),
        ("   ", " "),
        ("\n\n", " "),
        ("a \t\n b", "a b"),
        ("  x  ", "x"),
        ("\u{e000}", " "),
        ("ΣΑΣ", "σασ"),
    ] {
        assert_eq!(normalize(text), want, "{text:?}");
    }
}

#[test]
fn byte_level_alphabet() {
    let chars = byte_level_chars();
    assert_eq!(chars.len(), 256);
    assert_eq!(chars[&'Ġ'], b' ');
    assert_eq!(chars[&'Ċ'], b'\n');
    assert_eq!(chars[&'a'], b'a');
    assert_eq!(byte_level_decode("ĠThe", &chars), " The");
    // `é` as two byte-level chars, and half of it.
    assert_eq!(byte_level_decode("Ã©", &chars), "é");
    assert_eq!(byte_level_decode("Ã", &chars), "\u{fffd}");
    // A char outside the alphabet keeps the token's UTF-8.
    assert_eq!(byte_level_decode("<｜pad｜>", &chars), "<｜pad｜>");
}

#[test]
fn token_map_merges_tokens_that_normalize_alike() {
    let tok = json!({
        "model": {"type": "BPE", "vocab": {
            "<s>": 0, "ĠThe": 1, "the": 2, "THE": 3, "Ã": 4, "Ġ": 5, "ĠĠ": 6, "Ċ": 7,
            "Ã©": 8, "e": 9, "©": 10, "Â": 11
        }},
        "added_tokens": [
            {"id": 0, "content": "<｜begin▁of▁sentence｜>", "special": true},
            {"id": 13, "content": "<think>", "special": false}
        ],
        "decoder": {"type": "ByteLevel"}
    });
    let (map, n) = token_map(&tok).unwrap();
    // 0 begin; 1-3 "the"; 4 raw "Ã" (half a char); 5-7 " "; 8 "é" -> "e" = 9; 10 "©"
    // (U+00A9 decodes from byte 0xA9 alone: invalid UTF-8 -> raw "©"); 11 raw "Â";
    // 12 missing -> ""; 13 "<think>".
    assert_eq!(map, [0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8]);
    assert_eq!(n, 9);
}

#[test]
fn token_map_needs_a_byte_level_decoder() {
    let tok = json!({
        "model": {"type": "BPE", "vocab": {"a": 0}},
        "decoder": {"type": "Metaspace"}
    });
    assert!(token_map(&tok).is_err());
}

/// The real tokenizer's map against the released GGUF's `engram_token_map.weight` (its FNV-1a
/// over the I32 values). `LEGION_TEST_DS41_HF=<dir with tokenizer.json> cargo test --release
/// --lib released_token_map -- --ignored`.
#[test]
#[ignore = "needs the DeepSeek-V4.1-Flash tokenizer.json"]
fn released_token_map() {
    let dir = std::env::var("LEGION_TEST_DS41_HF").expect("LEGION_TEST_DS41_HF");
    let text = std::fs::read_to_string(std::path::Path::new(&dir).join("tokenizer.json")).unwrap();
    let tok: Json = serde_json::from_str(&text).unwrap();
    let (map, n) = token_map(&tok).unwrap();
    assert_eq!((map.len(), n), (129280, 99092));
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in map.iter().flat_map(|v| v.to_le_bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    assert_eq!(h, 0x6b5c_fa68_a5f5_7ba6);
}
