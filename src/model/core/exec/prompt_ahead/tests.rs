use super::*;

fn seq<'b>(tokens: &'b [u32], n_past: usize) -> SeqInput<'b> {
    SeqInput {
        tokens,
        n_past,
        blocks: &[],
        ring: 0,
        logits: Logits::None,
        media: None,
    }
}

fn chunk<'b>(tokens: &'b [u32], n_past: usize) -> Group<'b> {
    Group {
        slot: 3,
        seqs: vec![seq(tokens, n_past)],
        verify: false,
    }
}

fn next(tokens: usize, logits: Logits) -> PromptNext {
    PromptNext { tokens, logits }
}

#[test]
fn the_next_chunk_starts_where_this_one_ends() {
    let toks = [7u32; 4096];
    let groups = [chunk(&toks, 8192)];
    let ahead = next_groups(&groups, next(4096, Logits::None), None).expect("a prompt chunk");
    assert_eq!(ahead.len(), 1);
    assert_eq!(ahead[0].slot, 3);
    assert!(!ahead[0].verify);
    let [s] = &ahead[0].seqs[..] else {
        panic!("one sequence")
    };
    assert_eq!(s.n_past, 8192 + 4096);
    assert_eq!(s.tokens.len(), 4096);
    assert_eq!(s.logits, Logits::None);
}

#[test]
fn a_shorter_last_chunk_takes_its_count_and_logits() {
    let toks = [7u32; 4096];
    let groups = [chunk(&toks, 0)];
    let ahead = next_groups(&groups, next(1500, Logits::Last), None).expect("a prompt chunk");
    let s = &ahead[0].seqs[0];
    assert_eq!(
        (s.n_past, s.tokens.len(), s.logits),
        (4096, 1500, Logits::Last)
    );
}

#[test]
fn a_longer_or_empty_next_chunk_is_not_built() {
    let toks = [7u32; 1024];
    let groups = [chunk(&toks, 0)];
    assert!(next_groups(&groups, next(2048, Logits::None), None).is_none());
    assert!(next_groups(&groups, next(0, Logits::None), None).is_none());
}

#[test]
fn only_one_prompt_chunk_alone() {
    let toks = [7u32; 1024];
    let one = [1u32; 1];
    let two = [chunk(&toks, 0), chunk(&one, 50)];
    assert!(next_groups(&two, next(1024, Logits::None), None).is_none());
    let both = [Group {
        slot: 0,
        seqs: vec![seq(&toks, 0), seq(&toks, 0)],
        verify: false,
    }];
    assert!(next_groups(&both, next(1024, Logits::None), None).is_none());
    let verify = [Group {
        verify: true,
        ..chunk(&toks, 0)
    }];
    assert!(next_groups(&verify, next(1024, Logits::None), None).is_none());
    assert!(next_groups(&[], next(1024, Logits::None), None).is_none());
}

#[test]
fn the_next_chunk_must_fit_the_block_table() {
    let toks = [7u32; 1024];
    let groups = [chunk(&toks, 0)];
    let w = (2048usize).div_ceil(KV_BLOCK);
    assert!(next_groups(&groups, next(1024, Logits::None), Some(w)).is_some());
    assert!(next_groups(&groups, next(1024, Logits::None), Some(w - 1)).is_none());
}
