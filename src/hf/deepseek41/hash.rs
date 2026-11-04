//! DeepSeek-V4.1-Flash's Engram hash parameters, made from the checkpoint's own files as
//! DeepSeek's reference `inference/engram.py` (MIT) makes them at start-up: the checkpoint
//! ships the tables but not the parameters that index them.
//!
//! - [`primes`]: per Engram layer, n-gram size and head a bucket count, the primes above
//!   `engram_vocab_size - 1` in order, none handed out twice (`EngramLayout.from_args`);
//! - [`offsets`]: per layer the running sum of its primes (where each column's rows start);
//! - [`multipliers`]: per layer `max_ngram_size` odd multipliers from numpy's
//!   `default_rng(10007 * layer_id).integers(0, bound)` (`compute_hash_multipliers`), here a
//!   port of numpy's `SeedSequence`, `PCG64` and 64-bit Lemire bounded integers (BSD-3-Clause
//!   numpy, re-implemented from its documented algorithms);
//! - [`token_map`]: every token id onto the compressed id of its normalized text
//!   (`build_compressed_token_map`): the token decoded as the `tokenizers` byte-level decoder
//!   does, then NFKC, NFD, accents stripped, lowercased, whitespace runs made one space and
//!   trimmed (a lone space kept); partial UTF-8 tokens keyed by their raw form. Ids are given
//!   in first-seen order; their count must be `engram_compressed_vocab_size`, from which the
//!   multipliers derive.

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use serde_json::Value as Json;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

/// The Engram layout from the text config: layer ids, longest n-gram, heads, the bucket base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub layers: Vec<u64>,
    pub max_ngram: usize,
    pub heads: usize,
    pub vocab: u64,
}

/// Whether `n` is prime (trial division: the buckets are around `engram_vocab_size`).
fn is_prime(n: u64) -> bool {
    if n < 4 {
        return n >= 2;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }
    let mut d = 5u64;
    while d.saturating_mul(d) <= n {
        if n.is_multiple_of(d) || n.is_multiple_of(d + 2) {
            return false;
        }
        d += 6;
    }
    true
}

/// Per layer, its `(max_ngram - 1) × heads` bucket primes (n-gram size major, head minor).
pub fn primes(l: &Layout) -> Vec<Vec<u64>> {
    let mut last = 0u64;
    let mut out = Vec::with_capacity(l.layers.len());
    for _ in &l.layers {
        let mut layer = Vec::with_capacity((l.max_ngram - 1) * l.heads);
        for _ in 1..l.max_ngram {
            // Every n-gram size searches up from `vocab - 1`, skipping the primes handed out:
            // the primes come out in one increasing run.
            let mut current = (l.vocab.saturating_sub(1)).max(last);
            for _ in 0..l.heads {
                current += 1;
                while !is_prime(current) {
                    current += 1;
                }
                layer.push(current);
            }
            last = current;
        }
        out.push(layer);
    }
    out
}

/// Per layer, where each column's rows start: the running sum of its primes.
pub fn offsets(primes: &[Vec<u64>]) -> Vec<Vec<u64>> {
    primes
        .iter()
        .map(|p| {
            p.iter()
                .scan(0u64, |at, &n| {
                    let o = *at;
                    *at += n;
                    Some(o)
                })
                .collect()
        })
        .collect()
}

/// numpy's `SeedSequence(entropy)` pool (4 words) and `generate_state(n_words, uint32)`.
struct SeedSequence {
    pool: [u32; 4],
}

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

impl SeedSequence {
    fn new(entropy: u64) -> Self {
        // The integer as little-endian 32-bit words (one word for zero).
        let mut words = vec![entropy as u32];
        if entropy >> 32 != 0 {
            words.push((entropy >> 32) as u32);
        }
        let mut hash_const = INIT_A;
        let mut hashmix = |v: u32| {
            let mut v = v ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_A);
            v = v.wrapping_mul(hash_const);
            v ^ (v >> XSHIFT)
        };
        let mix = |x: u32, y: u32| {
            let r = MIX_MULT_L
                .wrapping_mul(x)
                .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
            r ^ (r >> XSHIFT)
        };
        let mut pool = [0u32; 4];
        for (i, p) in pool.iter_mut().enumerate() {
            *p = hashmix(words.get(i).copied().unwrap_or(0));
        }
        for src in 0..4 {
            for dst in 0..4 {
                if src != dst {
                    let h = hashmix(pool[src]);
                    pool[dst] = mix(pool[dst], h);
                }
            }
        }
        for &w in words.iter().skip(4) {
            for p in pool.iter_mut() {
                let h = hashmix(w);
                *p = mix(*p, h);
            }
        }
        Self { pool }
    }

    fn generate_state(&self, n_words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..n_words)
            .map(|i| {
                let mut v = self.pool[i % 4] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^ (v >> XSHIFT)
            })
            .collect()
    }
}

/// numpy's `PCG64` (PCG XSL-RR 128/64), seeded from a [`SeedSequence`].
struct Pcg64 {
    state: u128,
    inc: u128,
}

const PCG_MULT: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

impl Pcg64 {
    fn new(seed: u64) -> Self {
        let w = SeedSequence::new(seed).generate_state(8);
        let u64s: Vec<u64> = w
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| c[0] as u64 | (c[1] as u64) << 32)
            .collect();
        let initstate = (u64s[0] as u128) << 64 | u64s[1] as u128;
        let initseq = (u64s[2] as u128) << 64 | u64s[3] as u128;
        let mut r = Self {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        r.step();
        r.state = r.state.wrapping_add(initstate);
        r.step();
        r
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        ((s >> 64) as u64 ^ s as u64).rotate_right((s >> 122) as u32)
    }

    /// `Generator.integers(0, high)` for int64 and `high > 2^32`: Lemire's bounded integers.
    fn below(&mut self, high: u64) -> u64 {
        let rng = high - 1;
        let excl = high;
        let mut m = self.next_u64() as u128 * excl as u128;
        let mut leftover = m as u64;
        if leftover < excl {
            let threshold = (u64::MAX - rng) % excl;
            while leftover < threshold {
                m = self.next_u64() as u128 * excl as u128;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }
}

/// Per layer `max_ngram` odd multipliers for a compressed vocabulary of `compressed` ids.
pub fn multipliers(l: &Layout, compressed: u64) -> Result<Vec<Vec<i64>>, String> {
    let bound = ((i64::MAX as u64 / compressed.max(1)) / 2).max(1);
    if bound <= 1u64 << 32 {
        return Err(format!(
            "Engram: a compressed vocabulary of {compressed} gives multipliers below 2^32 (numpy's 32-bit path, not supported)"
        ));
    }
    Ok(l.layers
        .iter()
        .map(|&layer| {
            let mut g = Pcg64::new(10007 * layer);
            (0..l.max_ngram)
                .map(|_| (g.below(bound) * 2 + 1) as i64)
                .collect()
        })
        .collect())
}

/// The byte-level BPE alphabet: byte -> char (GPT-2's `bytes_to_unicode`), inverted.
fn byte_level_chars() -> HashMap<char, u8> {
    let mut out = HashMap::with_capacity(256);
    let mut n = 0u32;
    for b in 0..=255u8 {
        let printable = matches!(b, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff);
        let c = if printable {
            b as u32
        } else {
            n += 1;
            255 + n
        };
        out.insert(char::from_u32(c).expect("a byte-level char"), b);
    }
    out
}

/// The `tokenizers` `ByteLevel` decoder on one token: its chars as bytes when all are in the
/// byte-level alphabet, else its UTF-8 as is; then lossy UTF-8.
fn byte_level_decode(token: &str, chars: &HashMap<char, u8>) -> String {
    let bytes: Option<Vec<u8>> = token.chars().map(|c| chars.get(&c).copied()).collect();
    match bytes {
        Some(b) => String::from_utf8_lossy(&b).into_owned(),
        None => token.to_string(),
    }
}

/// `build_compressed_token_map`'s normalizer: NFKC, NFD, strip accents, lowercase, whitespace
/// runs as one space, a lone space kept, trimmed.
pub fn normalize(text: &str) -> String {
    let s: String = text
        .nfkc()
        .nfd()
        .filter(|&c| !is_combining_mark(c))
        .flat_map(char::to_lowercase)
        .collect();
    let mut collapsed = String::with_capacity(s.len());
    let mut in_run = false;
    for c in s.chars() {
        if matches!(c, ' ' | '\t' | '\r' | '\n') {
            if !in_run {
                collapsed.push(' ');
            }
            in_run = true;
        } else {
            collapsed.push(c);
            in_run = false;
        }
    }
    // `^ $` -> a private-use sentinel, strip, the sentinel back to a space (also any the text
    // had).
    if collapsed == " " {
        collapsed = SENTINEL.to_string();
    }
    collapsed.trim().replace(SENTINEL, " ")
}

const SENTINEL: char = '\u{e000}';

/// The compressed token map of `tokenizer.json` (parsed) and its number of compressed ids.
pub fn token_map(tok: &Json) -> Result<(Vec<i32>, usize), String> {
    let model = tok.get("model").ok_or("tokenizer.json: no `model`")?;
    let vocab = model
        .get("vocab")
        .and_then(Json::as_object)
        .ok_or("tokenizer.json: no `model.vocab`")?;
    let decoder = tok
        .get("decoder")
        .and_then(|d| d.get("type"))
        .and_then(Json::as_str);
    if decoder != Some("ByteLevel") {
        return Err(format!(
            "Engram token map: the tokenizer's decoder is {decoder:?}, only ByteLevel is supported"
        ));
    }
    let mut by_id: HashMap<u64, &str> = HashMap::with_capacity(vocab.len());
    for (t, id) in vocab {
        by_id.insert(
            id.as_u64().ok_or("tokenizer.json: non-integer vocab id")?,
            t,
        );
    }
    // Added tokens come first when decoding an id.
    for a in tok
        .get("added_tokens")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        if let (Some(id), Some(c)) = (
            a.get("id").and_then(Json::as_u64),
            a.get("content").and_then(Json::as_str),
        ) {
            by_id.insert(id, c);
        }
    }
    let n = by_id.keys().max().map_or(0, |&m| m as usize + 1);
    let chars = byte_level_chars();
    let mut keys: HashMap<String, i32> = HashMap::new();
    let mut map = Vec::with_capacity(n);
    for id in 0..n as u64 {
        let raw = by_id.get(&id).copied();
        let text = raw
            .map(|t| byte_level_decode(t, &chars))
            .unwrap_or_default();
        let key = if text.contains('\u{fffd}') {
            raw.unwrap_or_default().to_string()
        } else {
            let normalized = normalize(&text);
            if normalized.is_empty() {
                text
            } else {
                normalized
            }
        };
        let next = keys.len() as i32;
        map.push(*keys.entry(key).or_insert(next));
    }
    Ok((map, keys.len()))
}
