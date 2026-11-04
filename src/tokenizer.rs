//! BPE tokenizer (`tokenizer.ggml.model = gpt2` or `gemma4`) built from GGUF metadata.
//!
//! The behaviour follows llama.cpp (MIT): `src/llama-vocab.cpp` (`llm_tokenizer_bpe`,
//! `tokenizer_st_partition`, `llama_vocab::impl::tokenize`, `token_to_piece`) and `src/unicode.cpp`
//! (GPT-2 byte <-> unicode mapping). Pre-tokenizer regexes are copied from `llm_tokenizer_bpe`'s
//! constructor in `src/llama-vocab.cpp`.
//!
//! Why `fancy-regex` plus our own merge loop instead of the `tokenizers` crate: the pre-tokenizer
//! patterns need a look-ahead (`\s+(?!\S)`), which `fancy-regex` supports on pure Rust; `tokenizers`
//! pulls in a large dependency tree (onig, rayon, ...) and its BPE differs from llama.cpp in
//! special-token handling and merge tie-breaks, which would put byte-identical output at risk.
//! The merge loop below is a direct port of llama.cpp's bigram queue.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use fancy_regex::Regex;

use crate::gguf::{GgufFile, Value};

mod gemma4;
mod parallel;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("missing GGUF key `{0}`")]
    MissingKey(&'static str),
    #[error("GGUF key `{0}` has an unexpected type")]
    BadKey(&'static str),
    #[error("unsupported tokenizer model `{0}` (only `gpt2` and `gemma4` BPE are implemented)")]
    UnsupportedModel(String),
    #[error("unsupported pre-tokenizer `{0}`")]
    UnsupportedPre(String),
    #[error("invalid pre-tokenizer regex: {0}")]
    Regex(String),
}

/// Token kinds, numbered like llama.cpp's `llama_token_type` (`tokenizer.ggml.token_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenKind {
    fn from_gguf(v: i64) -> Self {
        match v {
            1 => TokenKind::Normal,
            2 => TokenKind::Unknown,
            3 => TokenKind::Control,
            4 => TokenKind::UserDefined,
            5 => TokenKind::Unused,
            6 => TokenKind::Byte,
            _ => TokenKind::Undefined,
        }
    }
}

// Source: llama.cpp src/llama-vocab.cpp, llm_tokenizer_bpe (LLAMA_VOCAB_PRE_TYPE_QWEN2).
const RE_QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
// Source: same place (LLAMA_VOCAB_PRE_TYPE_QWEN35): letters also take combining marks.
const RE_QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// Source: same place (LLAMA_VOCAB_PRE_TYPE_LLAMA3): digits in runs of up to three.
const RE_LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// Source: same place (LLAMA_VOCAB_PRE_TYPE_GPT4O), the regex of the original tokenizer.json
// that llama.cpp adapts for its own regex engine (the two agree on ASCII text).
const RE_GPT4O: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

// Source: same place (LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM), three expressions applied one after
// the other; the same three `Split` steps are in DeepSeek-V4.1-Flash's tokenizer.json, whose
// GGUF names its pre-tokenizer `joyai-llm`. The last one is `RE_DEEPSEEK3`, the two before it
// are `DEEPSEEK3_SPLITS`.
const DEEPSEEK3_SPLITS: [&str; 2] = [r"\p{N}{1,3}", r"[一-龥぀-ゟ゠-ヿ]+"];
const RE_DEEPSEEK3: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

/// Expressions that cut the text before the pre-tokenizer regex (llama.cpp's `regex_exprs`
/// before the last); empty for the single-regex pre-tokenizers.
fn pre_splits(pre: &str) -> &'static [&'static str] {
    match pre {
        "deepseek-v3" | "joyai-llm" => &DEEPSEEK3_SPLITS,
        _ => &[],
    }
}

/// Pre-tokenizer regex for a `tokenizer.ggml.pre` value (same aliases as llama.cpp), and
/// whether a pre-token that is itself in the vocabulary skips the merges (llama.cpp's
/// `ignore_merges`).
fn pre_regex(pre: &str) -> Option<(&'static str, bool)> {
    match pre {
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "megrez" => Some((RE_QWEN2, false)),
        "qwen35" => Some((RE_QWEN35, false)),
        // Nemotron-H's tokenizer is `pixtral`.
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "pixtral" | "midm-2.0" | "lfm2"
        | "jina-v5-nano" | "glm4" | "glm5" => Some((RE_LLAMA3, true)),
        // Muse Glimmer's tokenizer is `llama4`.
        "gpt-4o" | "llama4" | "kanana2" | "talkie" => Some((RE_GPT4O, false)),
        // The last of three expressions (`pre_splits`).
        "deepseek-v3" | "joyai-llm" => Some((RE_DEEPSEEK3, false)),
        _ => None,
    }
}

/// Byte ranges of `text` after cutting it with each expression in turn: every piece, on its own
/// (look-arounds see the piece's ends as the text's), becomes its matches and the text between
/// them. A regex error (a backtracking limit) leaves the rest of that piece whole.
fn split_pieces<'a>(res: impl Iterator<Item = &'a Regex>, text: &str) -> Vec<(usize, usize)> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut pieces = vec![(0, text.len())];
    for re in res {
        let mut next = Vec::with_capacity(pieces.len() * 2);
        for &(start, end) in &pieces {
            let mut pos = start;
            for m in re.find_iter(&text[start..end]) {
                let Ok(m) = m else { break };
                let (s, e) = (start + m.start(), start + m.end());
                if s == e {
                    continue;
                }
                if s > pos {
                    next.push((pos, s));
                }
                next.push((s, e));
                pos = e;
            }
            if pos < end {
                next.push((pos, end));
            }
        }
        pieces = next;
    }
    pieces
}

/// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest to U+0100 and up in order.
fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for (b, slot) in table.iter_mut().enumerate() {
        let printable = matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        *slot = if printable {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(next).unwrap();
            next += 1;
            c
        };
    }
    table
}

/// Pending merges: `(rank, left, right, left_len, right_len)`, smallest first.
type Queue = BinaryHeap<Reverse<(u32, u32, u32, u32, u32)>>;

#[derive(Clone, Copy)]
struct Symbol {
    start: usize,
    len: usize,
    prev: isize,
    next: isize,
}

pub struct Tokenizer {
    /// Token text as stored in the GGUF (GPT-2 byte encoding or Gemma 4 raw UTF-8).
    token_text: Vec<String>,
    /// Bytes each token stands for.
    pieces: Vec<Vec<u8>>,
    kinds: Vec<TokenKind>,
    token_to_id: HashMap<String, u32>,
    /// `"left right"` (as in the GGUF merges list) -> rank.
    ranks: HashMap<String, u32>,
    regex: Regex,
    /// Expressions applied before `regex` (`pre_splits`); empty for most pre-tokenizers.
    splits: Vec<Regex>,
    /// A pre-token found whole in the vocabulary is that token, merges or not.
    ignore_merges: bool,
    gemma4: bool,
    byte_to_char: [char; 256],
    /// Special-token candidates for `parse_special`, longest text first (llama.cpp's special cache).
    specials: Vec<u32>,
    bos: Option<u32>,
    eos: Option<u32>,
    eot: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
    add_bos: bool,
    add_eos: bool,
}

enum Fragment {
    Text(usize, usize),
    Token(u32),
}

impl Tokenizer {
    /// Build the tokenizer from `tokenizer.ggml.*` metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let model = gguf
            .get_str("tokenizer.ggml.model")
            .ok_or(TokenizerError::MissingKey("tokenizer.ggml.model"))?;
        let gemma4 = model == "gemma4";
        if model != "gpt2" && !gemma4 {
            return Err(TokenizerError::UnsupportedModel(model));
        }
        let pre = gguf.get_str("tokenizer.ggml.pre").unwrap_or_default();
        let (re, ignore_merges) = if gemma4 {
            (gemma4::REGEX, false)
        } else {
            pre_regex(&pre).ok_or_else(|| TokenizerError::UnsupportedPre(pre.clone()))?
        };
        let compile = |re: &str| Regex::new(re).map_err(|e| TokenizerError::Regex(e.to_string()));
        let regex = compile(re)?;
        let splits = if gemma4 {
            Vec::new()
        } else {
            pre_splits(&pre)
                .iter()
                .map(|re| compile(re))
                .collect::<Result<_, _>>()?
        };

        let tokens = string_array(gguf, "tokenizer.ggml.tokens")?;
        let merges = string_array(gguf, "tokenizer.ggml.merges")?;
        let n = tokens.len();

        let kinds: Vec<TokenKind> = match gguf.get("tokenizer.ggml.token_type") {
            Some(Value::Array(a)) if a.len() >= n => a
                .iter()
                .take(n)
                .map(|v| TokenKind::from_gguf(as_i64(v).unwrap_or(0)))
                .collect(),
            Some(Value::Array(_)) | Some(_) => {
                return Err(TokenizerError::BadKey("tokenizer.ggml.token_type"));
            }
            None => vec![TokenKind::Normal; n],
        };

        let mut token_text = Vec::with_capacity(n);
        let mut token_to_id = HashMap::with_capacity(n);
        for (i, t) in tokens.into_iter().enumerate() {
            // llama.cpp renames empty tokens so they stay addressable.
            let t = if t.is_empty() {
                format!("[EMPTY_{i}]")
            } else {
                t
            };
            token_to_id.insert(t.clone(), i as u32); // later duplicates win, like llama.cpp
            token_text.push(t);
        }

        // llama.cpp splits each merge at the first space after position 0; a merge without one
        // becomes the pair ("", ""), which can never match, so it is skipped here.
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.into_iter().enumerate() {
            if m.len() > 1 && m.as_bytes()[1..].contains(&b' ') {
                ranks.entry(m).or_insert(i as u32);
            }
        }

        let byte_to_char = byte_to_char_table();
        let char_to_byte: HashMap<char, u8> = byte_to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        let pieces = token_text
            .iter()
            .zip(&kinds)
            .map(|(t, k)| match k {
                TokenKind::Normal if gemma4 => t.replace('▁', " ").into_bytes(),
                TokenKind::Normal => decode_bytes(t, &char_to_byte),
                TokenKind::Byte => parse_byte_token(t)
                    .map(|b| vec![b])
                    .unwrap_or_else(|| t.as_bytes().to_vec()),
                _ => t.as_bytes().to_vec(),
            })
            .collect();

        // Defaults for the "gpt2" model in llama.cpp: bos = eos = 11 when in range, unset otherwise.
        let id_key = |key: &'static str| -> Option<u32> {
            let id = gguf.get_u64(key)?;
            (id < n as u64).then_some(id as u32)
        };
        let legacy = (!gemma4 && n > 11).then_some(11u32);
        let bos = id_key("tokenizer.ggml.bos_token_id").or(legacy);
        let eos = id_key("tokenizer.ggml.eos_token_id").or(legacy);
        let pad = id_key("tokenizer.ggml.padding_token_id");
        let unk = id_key("tokenizer.ggml.unknown_token_id");
        let eot = id_key("tokenizer.ggml.eot_token_id").or_else(|| {
            [
                "<|eot_id|>",
                "<|im_end|>",
                "<|end|>",
                "<end_of_turn>",
                "<|endoftext|>",
                "<|end_of_text|>",
            ]
            .iter()
            .find_map(|t| token_to_id.get(*t).copied())
        });
        let flag = |key: &str| gguf.get(key).and_then(|v| v.as_bool());
        // llama.cpp src/llama-vocab.cpp forces add_bos for Gemma 4, even if metadata says false.
        let add_bos = gemma4 || flag("tokenizer.ggml.add_bos_token").unwrap_or(false);
        let add_eos = flag("tokenizer.ggml.add_eos_token").unwrap_or(false);

        let mut specials: Vec<u32> = (0..n as u32)
            .filter(|&i| {
                matches!(
                    kinds[i as usize],
                    TokenKind::Control | TokenKind::UserDefined | TokenKind::Unknown
                )
            })
            .collect();
        specials.sort_by_key(|&i| Reverse(token_text[i as usize].len()));

        Ok(Tokenizer {
            token_text,
            pieces,
            kinds,
            token_to_id,
            ranks,
            regex,
            splits,
            ignore_merges,
            gemma4,
            byte_to_char,
            specials,
            bos,
            eos,
            eot,
            pad,
            unk,
            add_bos,
            add_eos,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    /// End-of-turn token (`<|im_end|>` for Qwen).
    pub fn eot_id(&self) -> Option<u32> {
        self.eot
    }

    pub fn pad_id(&self) -> Option<u32> {
        self.pad
    }

    pub fn unk_id(&self) -> Option<u32> {
        self.unk
    }

    /// Whether `encode(.., true)` prepends BOS (always enabled for Gemma 4).
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }

    /// Whether `encode(.., true)` appends EOS (`tokenizer.ggml.add_eos_token`).
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }

    pub fn token_kind(&self, id: u32) -> Option<TokenKind> {
        self.kinds.get(id as usize).copied()
    }

    /// True for control, user-defined and unknown tokens: the ones `encode` can match as a single
    /// token in the text when special parsing is on. Out-of-range ids are not special.
    pub fn is_special(&self, id: u32) -> bool {
        matches!(
            self.token_kind(id),
            Some(TokenKind::Control | TokenKind::UserDefined | TokenKind::Unknown)
        )
    }

    /// The token's text as stored in the GGUF (GPT-2 byte encoding or Gemma 4 raw UTF-8).
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.token_text.get(id as usize).map(String::as_str)
    }

    /// Id of the token whose GGUF text is exactly `text`.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    /// The bytes a token stands for. Control tokens return their text, so callers decide whether
    /// to show them (`decode` hides them). Out-of-range ids give an empty slice.
    pub fn token_to_piece(&self, id: u32) -> &[u8] {
        self.pieces.get(id as usize).map_or(&[], Vec::as_slice)
    }

    /// Encode with special-token parsing tied to `add_special`: when true, BOS/EOS are added as the
    /// model asks and `<|im_start|>`-style tokens in the text become single ids.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        self.encode_with(text, add_special, add_special)
    }

    /// Like llama.cpp's `llama_tokenize(text, add_special, parse_special)`.
    pub fn encode_with(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if add_special && self.add_bos {
            out.extend(self.bos);
        }
        let mut scratch = Scratch::default();
        for frag in self.partition(text, parse_special) {
            match frag {
                Fragment::Token(id) => out.push(id),
                Fragment::Text(s, e) => self.encode_text(&text[s..e], &mut scratch, &mut out),
            }
        }
        if add_special && self.add_eos {
            out.extend(self.eos);
        }
        out
    }

    /// Port of `tokenizer_st_partition`: cut the text at special tokens, longest token first.
    /// Control and unknown tokens only match when `parse_special` is on.
    fn partition(&self, text: &str, parse_special: bool) -> Vec<Fragment> {
        let mut frags = Vec::new();
        if !text.is_empty() {
            frags.push(Fragment::Text(0, text.len()));
        }
        if self.specials.is_empty() || text.is_empty() {
            return frags;
        }
        for &id in &self.specials {
            let kind = self.kinds[id as usize];
            if !parse_special && matches!(kind, TokenKind::Control | TokenKind::Unknown) {
                continue;
            }
            let needle = self.token_text[id as usize].as_str();
            if !text.contains(needle) {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len() + 2);
            for frag in frags {
                let Fragment::Text(start, end) = frag else {
                    next.push(frag);
                    continue;
                };
                let mut pos = start;
                while let Some(off) = text[pos..end].find(needle) {
                    let at = pos + off;
                    if at > pos {
                        next.push(Fragment::Text(pos, at));
                    }
                    next.push(Fragment::Token(id));
                    pos = at + needle.len();
                }
                if pos < end {
                    next.push(Fragment::Text(pos, end));
                }
            }
            frags = next;
        }
        frags
    }

    fn encode_text(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<u32>) {
        if self.gemma4 {
            self.encode_gemma4(text, scratch, out);
            return;
        }
        if !self.splits.is_empty() {
            self.encode_split(text, scratch, out);
            return;
        }
        if text.len() >= parallel::MIN_BYTES {
            let threads = std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .min(parallel::MAX_THREADS);
            if let Some(ids) = self.encode_text_parallel(text, threads, parallel::CHUNK_BYTES) {
                out.extend(ids);
                return;
            }
        }
        for m in self.regex.find_iter(text) {
            // The patterns only fail on backtracking limits; llama.cpp has no such failure mode,
            // so fall back to treating the rest of the text as one word.
            let (word, last) = match m {
                Ok(m) => (m.as_str(), false),
                Err(_) => (text, true),
            };
            self.encode_word(word, scratch, out);
            if last {
                break;
            }
        }
    }

    /// Several expressions one after the other, like llama.cpp's `unicode_regex_split` and the
    /// `Split { behavior: Isolated }` steps of a tokenizer.json: each expression, `regex` last,
    /// cuts every piece on its own into its matches and the text between them, and the pieces
    /// left at the end are the pre-tokens. Long texts merge their pre-tokens on several threads.
    fn encode_split(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<u32>) {
        let words = self.split_words(text);
        let threads = if text.len() >= parallel::MIN_BYTES {
            std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .min(parallel::MAX_THREADS)
        } else {
            1
        };
        if threads < 2 {
            for &(s, e) in &words {
                self.encode_word(&text[s..e], scratch, out);
            }
            return;
        }
        let per = words.len().div_ceil(threads).max(1);
        let parts: Vec<Vec<u32>> = std::thread::scope(|sc| {
            let jobs: Vec<_> = words
                .chunks(per)
                .map(|ws| {
                    sc.spawn(move || {
                        let mut scratch = Scratch::default();
                        let mut ids = Vec::with_capacity(ws.len() * 2);
                        for &(s, e) in ws {
                            self.encode_word(&text[s..e], &mut scratch, &mut ids);
                        }
                        ids
                    })
                })
                .collect();
            jobs.into_iter()
                .map(|j| j.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect()
        });
        for part in parts {
            out.extend(part);
        }
    }

    /// The byte ranges of `encode_split`'s pre-tokens.
    fn split_words(&self, text: &str) -> Vec<(usize, usize)> {
        split_pieces(self.splits.iter().chain([&self.regex]), text)
    }

    /// The ids of one pre-token: the whole-word token when merges are skipped for it, else BPE.
    fn encode_word(&self, word: &str, scratch: &mut Scratch, out: &mut Vec<u32>) {
        scratch.word.clear();
        scratch
            .word
            .extend(word.bytes().map(|b| self.byte_to_char[b as usize]));
        match self
            .token_to_id
            .get(scratch.word.as_str())
            .filter(|_| self.ignore_merges)
        {
            Some(&id) => out.push(id),
            None => self.bpe(scratch, out),
        }
    }

    /// Merge the characters of `scratch.word` by rank, like `llm_tokenizer_bpe_session::tokenize`.
    fn bpe(&self, s: &mut Scratch, out: &mut Vec<u32>) {
        let word = s.word.as_str();
        s.symbols.clear();
        let mut off = 0;
        for c in word.chars() {
            let len = c.len_utf8();
            let i = s.symbols.len() as isize;
            s.symbols.push(Symbol {
                start: off,
                len,
                prev: i - 1,
                next: i + 1,
            });
            off += len;
        }
        if let Some(last) = s.symbols.last_mut() {
            last.next = -1;
        }

        s.queue.clear();
        for i in 1..s.symbols.len() {
            self.add_bigram(
                word,
                &s.symbols,
                i as isize - 1,
                i as isize,
                &mut s.key,
                &mut s.queue,
            );
        }
        // Smallest rank first, then leftmost, as llama.cpp's priority queue orders them.
        while let Some(Reverse((_, left, right, llen, rlen))) = s.queue.pop() {
            let (l, r) = (left as usize, right as usize);
            if s.symbols[l].len == 0 || s.symbols[r].len == 0 {
                continue;
            }
            if s.symbols[l].len != llen as usize || s.symbols[r].len != rlen as usize {
                continue; // stale: one side was merged since this bigram was queued
            }
            s.symbols[l].len += s.symbols[r].len;
            s.symbols[r].len = 0;
            let rnext = s.symbols[r].next;
            s.symbols[l].next = rnext;
            if rnext >= 0 {
                s.symbols[rnext as usize].prev = left as isize;
            }
            let (lprev, lnext) = (s.symbols[l].prev, s.symbols[l].next);
            self.add_bigram(
                word,
                &s.symbols,
                lprev,
                left as isize,
                &mut s.key,
                &mut s.queue,
            );
            self.add_bigram(
                word,
                &s.symbols,
                left as isize,
                lnext,
                &mut s.key,
                &mut s.queue,
            );
        }

        for sym in s.symbols.iter().filter(|sym| sym.len > 0) {
            let text = &word[sym.start..sym.start + sym.len];
            match self.token_to_id.get(text) {
                Some(&id) => out.push(id),
                None if self.gemma4 => self.gemma4_byte_fallback(text, out),
                None => {
                    // Fall back to single characters (the byte tokens); unknown ones are dropped.
                    let mut buf = [0u8; 4];
                    out.extend(
                        text.chars()
                            .filter_map(|c| self.token_to_id.get(c.encode_utf8(&mut buf) as &str))
                            .copied(),
                    );
                }
            }
        }
    }

    fn add_bigram(
        &self,
        word: &str,
        symbols: &[Symbol],
        left: isize,
        right: isize,
        key: &mut String,
        queue: &mut Queue,
    ) {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (&symbols[left as usize], &symbols[right as usize]);
        key.clear();
        key.push_str(&word[l.start..l.start + l.len]);
        key.push(' ');
        key.push_str(&word[r.start..r.start + r.len]);
        if let Some(&rank) = self.ranks.get(key.as_str()) {
            queue.push(Reverse((
                rank,
                left as u32,
                right as u32,
                l.len as u32,
                r.len as u32,
            )));
        }
    }

    /// Ids to text. Control and unknown tokens are skipped (llama.cpp's `special = false`);
    /// bytes that are not valid UTF-8 become U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.decode_with(ids, false)
    }

    /// Like [`decode`](Self::decode); with `render_special` control tokens print their text.
    pub fn decode_with(&self, ids: &[u32], render_special: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if render_special || !self.hidden(id) {
                bytes.extend_from_slice(self.token_to_piece(id));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn hidden(&self, id: u32) -> bool {
        matches!(
            self.token_kind(id),
            Some(TokenKind::Control | TokenKind::Unknown)
        )
    }

    /// Incremental decoder for a token stream.
    pub fn decode_stream(&self) -> DecodeStream<'_> {
        DecodeStream::new(self)
    }
}

#[derive(Default)]
struct Scratch {
    word: String,
    key: String,
    symbols: Vec<Symbol>,
    queue: Queue,
}

/// Turns token ids into valid UTF-8 text a few at a time. A multi-byte character that is split
/// across tokens is held back until its last byte arrives.
pub struct DecodeStream<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
    render_special: bool,
}

impl<'a> DecodeStream<'a> {
    pub fn new(tok: &'a Tokenizer) -> Self {
        DecodeStream {
            tok,
            pending: Vec::new(),
            render_special: false,
        }
    }

    /// Also print control tokens such as `<|im_end|>` (hidden by default).
    pub fn render_special(mut self, on: bool) -> Self {
        self.render_special = on;
        self
    }

    /// Feed one token; returns the text that is now complete (possibly empty).
    pub fn push(&mut self, id: u32) -> String {
        if self.render_special || !self.tok.hidden(id) {
            self.pending.extend_from_slice(self.tok.token_to_piece(id));
        }
        self.take(false)
    }

    /// Flush bytes held back at the end of the stream (an unfinished character becomes U+FFFD).
    pub fn finish(&mut self) -> String {
        self.take(true)
    }

    fn take(&mut self, flush: bool) -> String {
        let mut text = String::new();
        let mut consumed = 0;
        loop {
            match std::str::from_utf8(&self.pending[consumed..]) {
                Ok(s) => {
                    text.push_str(s);
                    consumed = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    text.push_str(
                        std::str::from_utf8(&self.pending[consumed..consumed + valid]).unwrap(),
                    );
                    consumed += valid;
                    match e.error_len() {
                        Some(bad) => {
                            text.push('\u{FFFD}');
                            consumed += bad;
                        }
                        None if flush => {
                            text.push('\u{FFFD}');
                            consumed = self.pending.len();
                            break;
                        }
                        None => break, // incomplete sequence at the end: wait for more bytes
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        text
    }
}

fn string_array(gguf: &GgufFile, key: &'static str) -> Result<Vec<String>, TokenizerError> {
    match gguf.get(key) {
        None => Err(TokenizerError::MissingKey(key)),
        Some(Value::Array(a)) => a
            .into_iter()
            .map(|v| match v {
                Value::String(s) => Ok(s),
                _ => Err(TokenizerError::BadKey(key)),
            })
            .collect(),
        Some(_) => Err(TokenizerError::BadKey(key)),
    }
}

fn as_i64(v: &Value) -> Option<i64> {
    match *v {
        Value::I8(x) => Some(x.into()),
        Value::I16(x) => Some(x.into()),
        Value::I32(x) => Some(x.into()),
        Value::I64(x) => Some(x),
        _ => v.as_u64().and_then(|x| i64::try_from(x).ok()),
    }
}

/// Undo the GPT-2 byte-to-unicode mapping. Characters outside the table (never produced by a
/// well-formed vocab) are kept as their own UTF-8 bytes.
fn decode_bytes(text: &str, char_to_byte: &HashMap<char, u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for c in text.chars() {
        match char_to_byte.get(&c) {
            Some(&b) => out.push(b),
            None => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out
}

/// `<0xXX>` byte tokens.
fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2).then(|| u8::from_str_radix(hex, 16).ok())?
}

#[cfg(test)]
mod tests;
