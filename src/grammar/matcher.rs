//! Byte NFA, lazy DFA and the token-level [`Matcher`].

use std::collections::HashMap;
use std::sync::Arc;

use super::schema::{Node, Prop, lit};
use super::{Grammar, GrammarError, Options, TokenTable};

#[derive(Clone, Copy)]
struct ByteSet([u64; 4]);

impl ByteSet {
    fn range(lo: u8, hi: u8) -> Self {
        let mut s = ByteSet([0; 4]);
        for b in lo..=hi {
            s.add(b);
        }
        s
    }

    fn of(bytes: &[u8]) -> Self {
        let mut s = ByteSet([0; 4]);
        for &b in bytes {
            s.add(b);
        }
        s
    }

    fn add(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1 << (b & 63);
    }

    fn with(mut self, lo: u8, hi: u8) -> Self {
        for b in lo..=hi {
            self.add(b);
        }
        self
    }

    fn contains(&self, b: u8) -> bool {
        self.0[(b >> 6) as usize] >> (b & 63) & 1 != 0
    }
}

#[derive(Clone, Copy)]
enum St {
    Set(ByteSet, u32),
    Split(u32, u32),
    Match,
}

/// Thompson NFA over bytes.
pub(crate) struct Nfa {
    states: Vec<St>,
}

type R<T> = Result<T, GrammarError>;

struct Builder<'a> {
    states: Vec<St>,
    opts: &'a Options,
}

impl Nfa {
    pub(crate) fn build(node: &Node, opts: &Options) -> R<(Nfa, u32)> {
        let mut b = Builder {
            states: Vec::new(),
            opts,
        };
        let end = b.add(St::Match)?;
        let start = b.build(node, end)?;
        Ok((Nfa { states: b.states }, start))
    }

    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }
}

impl Builder<'_> {
    fn add(&mut self, s: St) -> R<u32> {
        if self.states.len() >= self.opts.max_nfa_states {
            return Err(GrammarError::TooLarge(self.opts.max_nfa_states));
        }
        self.states.push(s);
        Ok((self.states.len() - 1) as u32)
    }

    fn set(&mut self, set: ByteSet, next: u32) -> R<u32> {
        self.add(St::Set(set, next))
    }

    fn byte(&mut self, b: u8, next: u32) -> R<u32> {
        self.set(ByteSet::of(&[b]), next)
    }

    fn split(&mut self, a: u32, b: u32) -> R<u32> {
        self.add(St::Split(a, b))
    }

    fn hole(&mut self) -> R<u32> {
        self.add(St::Split(0, 0))
    }

    fn alt(&mut self, starts: &[u32]) -> R<u32> {
        let (last, rest) = starts.split_last().expect("alternation is not empty");
        let mut acc = *last;
        for &s in rest.iter().rev() {
            acc = self.split(s, acc)?;
        }
        Ok(acc)
    }

    fn literal(&mut self, bytes: &[u8], mut next: u32) -> R<u32> {
        for &b in bytes.iter().rev() {
            next = self.byte(b, next)?;
        }
        Ok(next)
    }

    /// Optional JSON whitespace.
    fn ws(&mut self, next: u32) -> R<u32> {
        let w = ByteSet::of(b" \n\t\r");
        let mut tail = next;
        for _ in 0..self.opts.max_whitespace {
            let c = self.set(w, tail)?;
            tail = self.split(c, next)?;
        }
        Ok(tail)
    }

    /// `body{min,max}` followed by `next`.
    fn repeat(
        &mut self,
        min: usize,
        max: Option<usize>,
        body: &mut dyn FnMut(&mut Self, u32) -> R<u32>,
        next: u32,
    ) -> R<u32> {
        let mut tail = match max {
            None => {
                let l = self.hole()?;
                let b = body(self, l)?;
                self.states[l as usize] = St::Split(b, next);
                l
            }
            Some(max) => {
                let mut tail = next;
                for _ in min..max {
                    let b = body(self, tail)?;
                    tail = self.split(b, next)?;
                }
                tail
            }
        };
        for _ in 0..min {
            tail = body(self, tail)?;
        }
        Ok(tail)
    }

    /// One character of a JSON string body (valid UTF-8, no lone surrogate escapes).
    fn json_char(&mut self, next: u32) -> R<u32> {
        let plain = ByteSet::range(0x20, 0x21).with(0x23, 0x5B).with(0x5D, 0x7F);
        let mut alts = vec![self.set(plain, next)?];
        self.utf8_multi(&mut alts, next)?;
        // Escapes.
        let hex = ByteSet::range(b'0', b'9').with(b'a', b'f').with(b'A', b'F');
        let h1 = self.set(hex, next)?;
        let h2 = self.set(hex, h1)?;
        let h3 = self.set(hex, h2)?;
        let first = ByteSet::range(b'0', b'9')
            .with(b'a', b'c')
            .with(b'A', b'C')
            .with(b'e', b'f')
            .with(b'E', b'F');
        let a = self.set(first, h3)?;
        let d2 = self.set(ByteSet::range(b'0', b'7'), h2)?;
        let d = self.set(ByteSet::of(b"dD"), d2)?;
        let u_body = self.alt(&[a, d])?;
        let u = self.byte(b'u', u_body)?;
        let simple = self.set(ByteSet::of(b"\"\\/bfnrt"), next)?;
        let esc_body = self.split(simple, u)?;
        alts.push(self.byte(b'\\', esc_body)?);
        self.alt(&alts)
    }

    /// The starts of every valid multi-byte UTF-8 character (no surrogates), each followed by
    /// `next`, pushed onto `alts`.
    fn utf8_multi(&mut self, alts: &mut Vec<u32>, next: u32) -> R<()> {
        let cont = ByteSet::range(0x80, 0xBF);
        let c1 = self.set(cont, next)?;
        let c2 = self.set(cont, c1)?;
        let c3 = self.set(cont, c2)?;
        alts.push(self.set(ByteSet::range(0xC2, 0xDF), c1)?);
        let e0 = self.set(ByteSet::range(0xA0, 0xBF), c1)?;
        alts.push(self.byte(0xE0, e0)?);
        alts.push(self.set(ByteSet::range(0xE1, 0xEC).with(0xEE, 0xEF), c2)?);
        let ed = self.set(ByteSet::range(0x80, 0x9F), c1)?;
        alts.push(self.byte(0xED, ed)?);
        let f0 = self.set(ByteSet::range(0x90, 0xBF), c2)?;
        alts.push(self.byte(0xF0, f0)?);
        alts.push(self.set(ByteSet::range(0xF1, 0xF3), c3)?);
        let f4 = self.set(ByteSet::range(0x80, 0x8F), c2)?;
        alts.push(self.byte(0xF4, f4)?);
        Ok(())
    }

    /// UTF-8 text (no NUL) that contains neither `end` nor a `forbid` string, then `end`, then
    /// `next`: an Aho-Corasick automaton over the bytes ([`RawText`]). A multi-byte character
    /// (never part of the ASCII strings) goes back to the start.
    fn raw_until(&mut self, end: &[u8], forbid: &[Vec<u8>], next: u32) -> R<u32> {
        let raw = RawText::new(end, forbid);
        let mut holes = Vec::with_capacity(raw.states());
        for _ in 0..raw.states() {
            holes.push(self.hole()?);
        }
        let mut multi = Vec::new();
        self.utf8_multi(&mut multi, holes[0])?;
        for (j, &hole) in holes.iter().enumerate() {
            // ASCII bytes grouped by where they lead (a forbidden string: nowhere).
            let mut targets: Vec<(RawStep, ByteSet)> = Vec::new();
            for b in 1u8..=0x7F {
                let to = raw.step(j, b);
                match targets.iter_mut().find(|t| t.0 == to) {
                    Some(t) => t.1.add(b),
                    None => targets.push((to, ByteSet::of(&[b]))),
                }
            }
            let mut alts = multi.clone();
            for (to, set) in targets {
                let to = match to {
                    RawStep::To(k) => holes[k],
                    RawStep::End => next,
                    RawStep::Dead => continue,
                };
                alts.push(self.set(set, to)?);
            }
            let s = self.alt(&alts)?;
            self.states[hole as usize] = St::Split(s, s);
        }
        Ok(holes[0])
    }

    fn digits1(&mut self, next: u32) -> R<u32> {
        let digit = ByteSet::range(b'0', b'9');
        let l = self.hole()?;
        let first = self.set(digit, l)?;
        let more = self.set(digit, l)?;
        self.states[l as usize] = St::Split(more, next);
        Ok(first)
    }

    fn number(&mut self, integer: bool, next: u32) -> R<u32> {
        let after_int = if integer {
            next
        } else {
            // exponent: [eE][+-]?[0-9]{1,2}
            let d = ByteSet::range(b'0', b'9');
            let d2 = self.split(0, next)?; // patched below
            let e_d2 = self.set(d, next)?;
            self.states[d2 as usize] = St::Split(e_d2, next);
            let e_d1 = self.set(d, d2)?;
            let sign = self.set(ByteSet::of(b"+-"), e_d1)?;
            let e_body = self.split(sign, e_d1)?;
            let exp = self.set(ByteSet::of(b"eE"), e_body)?;
            let after_frac = self.split(next, exp)?;
            let frac_digits = self.digits1(after_frac)?;
            let frac = self.byte(b'.', frac_digits)?;
            self.alt(&[next, exp, frac])?
        };
        let zero = self.byte(b'0', after_int)?;
        let l = self.hole()?;
        self.states[l as usize] = St::Split(0, after_int);
        let more = self.set(ByteSet::range(b'0', b'9'), l)?;
        self.states[l as usize] = St::Split(more, after_int);
        let nonzero = self.set(ByteSet::range(b'1', b'9'), l)?;
        let body = self.split(zero, nonzero)?;
        let neg = self.byte(b'-', body)?;
        self.split(body, neg)
    }

    fn kv(&mut self, p: &Prop, next: u32) -> R<u32> {
        let value = self.build(&p.value, next)?;
        let after_colon = self.ws(value)?;
        let colon = self.byte(b':', after_colon)?;
        let after_key = self.ws(colon)?;
        let key = serde_json::to_vec(&p.key).expect("a string serializes");
        self.literal(&key, after_key)
    }

    fn object(&mut self, props: &[Prop], next: u32) -> R<u32> {
        let close = self.byte(b'}', next)?;
        let n = props.len();
        // rec[i][e]: before property i, `e` = a property was already written.
        let mut rec = vec![[close; 2]; n + 1];
        for i in (0..n).rev() {
            let after = self.ws(rec[i + 1][1])?;
            let kv = self.kv(&props[i], after)?;
            let comma_ws = self.ws(kv)?;
            let with_comma = self.byte(b',', comma_ws)?;
            let mut row = rec[i];
            for (e, slot) in row.iter_mut().enumerate() {
                let enter = if e == 1 { with_comma } else { kv };
                *slot = if props[i].required {
                    enter
                } else {
                    self.split(enter, rec[i + 1][e])?
                };
            }
            rec[i] = row;
        }
        let inner = self.ws(rec[0][0])?;
        self.byte(b'{', inner)
    }

    fn map(&mut self, value: &Node, next: u32) -> R<u32> {
        let close = self.byte(b'}', next)?;
        let l = self.hole()?;
        let after_value = self.ws(l)?;
        let v = self.build(value, after_value)?;
        let after_colon = self.ws(v)?;
        let colon = self.byte(b':', after_colon)?;
        let after_key = self.ws(colon)?;
        let key = self.build(&Node::Str { min: 0, max: None }, after_key)?;
        let comma_ws = self.ws(key)?;
        let comma = self.byte(b',', comma_ws)?;
        self.states[l as usize] = St::Split(comma, close);
        let first = self.split(close, key)?;
        let inner = self.ws(first)?;
        self.byte(b'{', inner)
    }

    fn array(&mut self, item: &Node, min: usize, max: Option<usize>, next: u32) -> R<u32> {
        let close = self.byte(b']', next)?;
        let k = max.unwrap_or(min.max(1));
        if k == 0 {
            let inner = self.ws(close)?;
            return self.byte(b'[', inner);
        }
        let mut after = vec![0u32; k + 1];
        for slot in after.iter_mut().skip(1) {
            *slot = self.hole()?;
        }
        let mut items = vec![0u32; k + 1];
        for i in 1..=k {
            let a = self.ws(after[i])?;
            items[i] = self.build(item, a)?;
        }
        for i in 1..=k {
            let mut alts = Vec::new();
            if i < k || max.is_none() {
                let target = items[(i + 1).min(k)];
                let w = self.ws(target)?;
                alts.push(self.byte(b',', w)?);
            }
            if i >= min {
                alts.push(close);
            }
            self.states[after[i] as usize] = {
                let s = self.alt(&alts)?;
                // Re-point the placeholder at the alternation.
                St::Split(s, s)
            };
        }
        let first = if min == 0 {
            self.split(close, items[1])?
        } else {
            items[1]
        };
        let inner = self.ws(first)?;
        self.byte(b'[', inner)
    }

    fn build(&mut self, node: &Node, next: u32) -> R<u32> {
        match node {
            Node::Never => Err(GrammarError::Unsatisfiable),
            Node::Lit(b) => self.literal(b, next),
            Node::Alt(v) => {
                let mut starts = Vec::new();
                for x in v {
                    starts.push(self.build(x, next)?);
                }
                self.alt(&starts)
            }
            Node::Str { min, max } => {
                let close = self.byte(b'"', next)?;
                let body = self.repeat(*min, *max, &mut |b, n| b.json_char(n), close)?;
                self.byte(b'"', body)
            }
            Node::Number { integer } => self.number(*integer, next),
            Node::Object(props) => self.object(props, next),
            Node::Map(v) => self.map(v, next),
            Node::Array { item, min, max } => self.array(item, *min, *max, next),
            Node::Any(depth) => {
                let mut alts = vec![
                    Node::Str { min: 0, max: None },
                    Node::Number { integer: false },
                    lit(&serde_json::Value::Bool(true)),
                    lit(&serde_json::Value::Bool(false)),
                    lit(&serde_json::Value::Null),
                ];
                if *depth > 0 {
                    alts.push(Node::Map(Box::new(Node::Any(depth - 1))));
                    alts.push(Node::Array {
                        item: Box::new(Node::Any(depth - 1)),
                        min: 0,
                        max: None,
                    });
                }
                self.build(&Node::Alt(alts), next)
            }
            Node::Seq(v) => {
                let mut next = next;
                for x in v.iter().rev() {
                    next = self.build(x, next)?;
                }
                Ok(next)
            }
            Node::Opt(x) => {
                let s = self.build(x, next)?;
                self.split(s, next)
            }
            Node::Star(x) => {
                let l = self.hole()?;
                let b = self.build(x, l)?;
                self.states[l as usize] = St::Split(b, next);
                Ok(l)
            }
            Node::RawUntil { end, forbid } => self.raw_until(end, forbid, next),
        }
    }
}

/// The automaton of [`Node::RawUntil`]'s text. State `j` is `prefixes[j]`: the longest end of
/// the text so far that starts one of the strings (state 0: none, the empty prefix). Any
/// string that ends at a byte is an end of that state followed by the byte.
pub(crate) struct RawText<'a> {
    end: &'a [u8],
    forbid: &'a [Vec<u8>],
    prefixes: Vec<&'a [u8]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RawStep {
    To(usize),
    /// `end` is complete.
    End,
    /// A forbidden string is complete.
    Dead,
}

impl<'a> RawText<'a> {
    pub(crate) fn new(end: &'a [u8], forbid: &'a [Vec<u8>]) -> Self {
        let strings = std::iter::once(end).chain(forbid.iter().map(Vec::as_slice));
        let mut prefixes: Vec<&[u8]> = vec![&[]];
        for s in strings {
            assert!(
                !s.is_empty() && s.is_ascii() && !s.contains(&0),
                "raw text's strings are non-empty ASCII without NUL"
            );
            for i in 1..s.len() {
                if !prefixes.contains(&&s[..i]) {
                    prefixes.push(&s[..i]);
                }
            }
        }
        Self {
            end,
            forbid,
            prefixes,
        }
    }

    pub(crate) fn states(&self) -> usize {
        self.prefixes.len()
    }

    /// Where state `j` goes on ASCII byte `b`. `end` wins over a forbidden string ending
    /// with it.
    pub(crate) fn step(&self, j: usize, b: u8) -> RawStep {
        let mut text = self.prefixes[j].to_vec();
        text.push(b);
        if text.ends_with(self.end) {
            return RawStep::End;
        }
        if self.forbid.iter().any(|f| text.ends_with(f)) {
            return RawStep::Dead;
        }
        let to = (0..=text.len())
            .find_map(|k| self.prefixes.iter().position(|p| *p == &text[k..]))
            .expect("the empty prefix is a state");
        RawStep::To(to)
    }
}

const UNKNOWN: u32 = u32::MAX;
const DEAD: u32 = u32::MAX - 1;

struct DState {
    /// Sorted `Set`/`Match` NFA states.
    nfa: Box<[u32]>,
    next: Box<[u32; 256]>,
    accepting: bool,
}

/// On-demand subset construction plus the per-state mask cache.
pub(crate) struct Lazy {
    states: Vec<DState>,
    index: HashMap<Box<[u32]>, u32>,
    masks: Vec<Option<Arc<[u32]>>>,
    seen: Vec<u32>,
    stamp: u32,
    stack: Vec<u32>,
}

impl Lazy {
    pub(crate) fn new(nfa: &Nfa, start: u32) -> Self {
        let mut l = Lazy {
            states: Vec::new(),
            index: HashMap::new(),
            masks: Vec::new(),
            seen: vec![0; nfa.len()],
            stamp: 0,
            stack: Vec::new(),
        };
        let s = l.closure(nfa, &[start]);
        debug_assert_eq!(s, 0);
        l
    }

    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }

    /// DFA state for the epsilon closure of `seeds`, or `DEAD`.
    fn closure(&mut self, nfa: &Nfa, seeds: &[u32]) -> u32 {
        self.stamp += 1;
        let stamp = self.stamp;
        let mut out: Vec<u32> = Vec::new();
        self.stack.clear();
        self.stack.extend_from_slice(seeds);
        while let Some(s) = self.stack.pop() {
            if self.seen[s as usize] == stamp {
                continue;
            }
            self.seen[s as usize] = stamp;
            match nfa.states[s as usize] {
                St::Split(a, b) => {
                    self.stack.push(a);
                    self.stack.push(b);
                }
                _ => out.push(s),
            }
        }
        if out.is_empty() {
            return DEAD;
        }
        out.sort_unstable();
        let key: Box<[u32]> = out.into_boxed_slice();
        if let Some(&id) = self.index.get(&key) {
            return id;
        }
        let id = self.states.len() as u32;
        let accepting = key
            .iter()
            .any(|&s| matches!(nfa.states[s as usize], St::Match));
        self.states.push(DState {
            nfa: key.clone(),
            next: Box::new([UNKNOWN; 256]),
            accepting,
        });
        self.masks.push(None);
        self.index.insert(key, id);
        id
    }

    #[inline]
    fn step(&mut self, nfa: &Nfa, s: u32, b: u8) -> u32 {
        let t = self.states[s as usize].next[b as usize];
        if t != UNKNOWN {
            return t;
        }
        let mut seeds = Vec::new();
        for &n in self.states[s as usize].nfa.iter() {
            if let St::Set(set, to) = nfa.states[n as usize]
                && set.contains(b)
            {
                seeds.push(to);
            }
        }
        let t = if seeds.is_empty() {
            DEAD
        } else {
            self.closure(nfa, &seeds)
        };
        self.states[s as usize].next[b as usize] = t;
        t
    }

    fn mask(&mut self, g: &Grammar, s: u32) -> Arc<[u32]> {
        if let Some(m) = &self.masks[s as usize] {
            return m.clone();
        }
        let table: &TokenTable = &g.inner.table;
        let nfa = &g.inner.nfa;
        let mut mask = vec![0u32; table.mask_words()];
        let mut stack = vec![(0u32, s)];
        while let Some((node, st)) = stack.pop() {
            for e in table.edges(node) {
                let (byte, child) = table.edge(e);
                let t = self.step(nfa, st, byte);
                if t == DEAD {
                    continue;
                }
                for &id in table.tokens_at(child) {
                    mask[(id >> 5) as usize] |= 1 << (id & 31);
                }
                stack.push((child, t));
            }
        }
        if self.states[s as usize].accepting {
            for &id in g.inner.stop.iter() {
                mask[(id >> 5) as usize] |= 1 << (id & 31);
            }
        }
        let m: Arc<[u32]> = mask.into();
        self.masks[s as usize] = Some(m.clone());
        m
    }
}

/// The token was not allowed in the current state; the matcher did not move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("token {0} is not allowed by the grammar here")]
pub struct Rejected(pub u32);

/// Walks a generation through a [`Grammar`]. Cheap to clone (a state id and an `Arc`).
#[derive(Clone)]
pub struct Matcher {
    grammar: Grammar,
    state: u32,
    finished: bool,
}

impl Matcher {
    pub(crate) fn new(grammar: Grammar) -> Self {
        Matcher {
            grammar,
            state: 0,
            finished: false,
        }
    }

    fn lazy(&self) -> std::sync::MutexGuard<'_, Lazy> {
        self.grammar
            .inner
            .lazy
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Words `fill_mask` writes: `ceil(vocab / 32)`.
    pub fn mask_words(&self) -> usize {
        self.grammar.mask_words()
    }

    /// Write the allowed-token bitset (bit `id % 32` of word `id / 32`). Words past
    /// `mask_words()` are cleared. Panics if `mask` is shorter than `mask_words()`.
    pub fn fill_mask(&self, mask: &mut [u32]) {
        let words = self.mask_words();
        assert!(
            mask.len() >= words,
            "mask has {} words, the vocabulary needs {words}",
            mask.len()
        );
        match self.shared_mask() {
            Some(m) => {
                mask[..words].copy_from_slice(&m);
                mask[words..].fill(0);
            }
            None => mask.fill(0),
        }
    }

    /// The allowed-token bitset ([`Self::fill_mask`]'s layout, `mask_words()` long) as the
    /// grammar caches it, without a copy; `None` once finished (nothing is allowed). Safe to
    /// call from another thread on a clone of the matcher.
    pub fn shared_mask(&self) -> Option<Arc<[u32]>> {
        if self.finished {
            return None;
        }
        Some(self.lazy().mask(&self.grammar, self.state))
    }

    /// Whether the text so far is a complete document (EOS is allowed).
    pub fn is_complete(&self) -> bool {
        self.finished || self.lazy().states[self.state as usize].accepting
    }

    /// True once an EOS/EOT token was accepted.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Advance by one token, or leave the state alone and say why not.
    pub fn accept_token(&mut self, id: u32) -> Result<(), Rejected> {
        if self.finished || id as usize >= self.grammar.inner.table.vocab() {
            return Err(Rejected(id));
        }
        let table = &self.grammar.inner.table;
        if self.grammar.inner.stop.contains(&id) {
            if self.is_complete() {
                self.finished = true;
                return Ok(());
            }
            return Err(Rejected(id));
        }
        let piece = table.piece(id);
        if piece.is_empty() {
            return Err(Rejected(id));
        }
        let nfa = &self.grammar.inner.nfa;
        let mut lazy = self
            .grammar
            .inner
            .lazy
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut s = self.state;
        for &b in piece {
            s = lazy.step(nfa, s, b);
            if s == DEAD {
                return Err(Rejected(id));
            }
        }
        self.state = s;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
