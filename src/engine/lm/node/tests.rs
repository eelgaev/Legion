use super::*;

#[test]
fn each_tensor_loads_on_one_rank() {
    let n = 8;
    for (name, r0, r1) in [
        ("token_embd.weight", true, false),
        ("rope_freqs.weight", true, true),
        ("output_norm.weight", false, true),
        ("output.weight", false, true),
        ("blk.0.attn_q.weight", true, false),
        ("blk.2.ffn_down_exps.weight", true, false),
        ("blk.3.ffn_down_exps.weight", false, true),
        ("blk.7.attn_norm.weight", false, true),
        // An MTP head's layer, past the last: no MTP over two nodes.
        ("blk.8.nextn.eh_proj.weight", false, false),
    ] {
        assert_eq!(owns(0, 3, n, name), r0, "{name} on rank 0");
        assert_eq!(owns(1, 3, n, name), r1, "{name} on rank 1");
    }
}

#[test]
fn balanced_split_evens_the_bytes() {
    // Rank 0 has the embeddings, rank 1 the head.
    let bytes = [10u64; 8];
    assert_eq!(balanced_split(&bytes, 0, 0), 4);
    assert_eq!(balanced_split(&bytes, 20, 0), 3);
    assert_eq!(balanced_split(&bytes, 0, 20), 5);
}

#[test]
fn fitting_split_or_none() {
    let bytes = [10u64; 8];
    assert_eq!(fitting_split(&bytes, 0, 0, 40), Some(4));
    // The head on rank 1 leaves it room for 3 layers: only a split at 5 fits.
    assert_eq!(fitting_split(&bytes, 0, 15, 50), Some(5));
    // Several fit: the most even one.
    assert_eq!(fitting_split(&bytes, 0, 0, 60), Some(4));
    assert_eq!(fitting_split(&bytes, 0, 0, 30), None);
}

#[test]
fn shares_take_rank_bytes() {
    let bytes = [1u64, 2, 3, 4];
    assert_eq!(rank_bytes(&bytes, 10, 20, 2), (13, 27));
}
