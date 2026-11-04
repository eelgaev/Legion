//! Gemma 4's SPM-style BPE rules, ported from llama.cpp (MIT),
//! `src/llama-vocab.cpp`: `llm_tokenizer_bpe` (LLAMA_VOCAB_PRE_TYPE_GEMMA4),
//! `llm_tokenizer_bpe_session::tokenize`, and `llama_vocab::impl::tokenize`.

use super::{Scratch, Tokenizer};

pub(super) const REGEX: &str = r"[^\n]+|[\n]+";

impl Tokenizer {
    pub(super) fn encode_gemma4(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<u32>) {
        for word in self.regex.find_iter(text) {
            // This pattern has no backtracking or lookaround.
            let word = word
                .expect("Gemma 4 newline split cannot backtrack")
                .as_str();
            // Newline-only words use a whole-word token when one exists, even without merges.
            if word.starts_with('\n')
                && let Some(&id) = self.token_to_id.get(word)
            {
                out.push(id);
                continue;
            }
            scratch.word.clear();
            scratch
                .word
                .extend(word.chars().map(|c| if c == ' ' { '▁' } else { c }));
            self.bpe(scratch, out);
        }
    }

    pub(super) fn gemma4_byte_fallback(&self, text: &str, out: &mut Vec<u32>) {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        for byte in text.bytes() {
            let token = [
                b'<',
                b'0',
                b'x',
                HEX[(byte >> 4) as usize],
                HEX[(byte & 15) as usize],
                b'>',
            ];
            let token = std::str::from_utf8(&token).expect("hex byte token is ASCII");
            if let Some(&id) = self.token_to_id.get(token) {
                out.push(id);
            }
        }
    }
}
