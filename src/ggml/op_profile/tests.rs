//! The op profile's weight test and column count.

use super::{act_tokens, weight_name};

#[test]
fn weights_are_named_like_model_tensors() {
    for name in [
        "blk.3.attn_q.weight",
        "blk.0.ffn_down_exps.scale",
        "output.weight",
        "token_embd.weight",
        "ssd.3.gate",
    ] {
        assert!(weight_name(name.as_bytes()), "{name}");
    }
    for name in [
        "cache_k-3",
        "cache_v-12",
        "cache_idx-0",
        "host_cache_k-3",
        "ssm_state-14",
        "conv_state-2",
        "ssm_landing-5",
        "ple_state-1",
        "pool_k-0",
        "spec_q-7",
        "mail0",
        "view_room-2",
    ] {
        assert!(!weight_name(name.as_bytes()), "{name}");
    }
}

#[test]
fn columns_from_the_activation() {
    // A matvec's input [k, n].
    assert_eq!(act_tokens([2560, 1, 1, 1], 2560), Some(1));
    assert_eq!(act_tokens([2560, 8, 1, 1], 2560), Some(8));
    // The hc mixer's streams [n_embd, hc, n] against its down weight's rows of n_embd * hc.
    assert_eq!(act_tokens([2560, 4, 1, 1], 10240), Some(1));
    assert_eq!(act_tokens([2560, 4, 8, 1], 10240), Some(8));
    assert_eq!(act_tokens([2560, 1, 4, 2], 2560), Some(8));
    // No leading run of dimensions is one row.
    assert_eq!(act_tokens([2560, 4, 8, 1], 320), None);
    assert_eq!(act_tokens([2560, 3, 8, 1], 10240), None);
    assert_eq!(act_tokens([81920, 1, 1, 1], 10240), None);
}
