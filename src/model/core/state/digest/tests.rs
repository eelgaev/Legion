use super::super::tests::{fill, hparams};
use super::*;
use crate::ggml::Backend;

#[test]
fn fnv1a_reference_values() {
    assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
}

/// A slot's digests follow its own bytes only: a copied slot has the same ones, and a byte
/// changed in one tensor of the slot changes that tensor's digest alone.
#[test]
fn slot_digests_follow_the_slots_bytes() {
    let hp = hparams();
    let mut backend = Backend::cpu().unwrap();
    let buft = backend.buffer_type();
    let pool = StatePool::new(&hp, buft, 3).unwrap();
    fill(&pool, 5);
    StatePool::copy_slots(&mut backend, &[(&pool, 0, &pool, 2)]).unwrap();
    let d0 = pool.slot_digests(0);
    let names: Vec<_> = d0.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["conv_state-0", "ssm_state-0", "tokens"]);
    assert_eq!(pool.slot_digests(2), d0);
    assert_ne!(pool.slot_digests(1), d0);

    // One byte of slot 2's delta-rule state.
    let ssm = pool.ssm(0);
    let at = 2 * ssm.nb()[3] + 5;
    let old = ssm.bytes_at(at, 1)[0];
    ssm.set_bytes_at(at, &[old ^ 1]);
    let d2 = pool.slot_digests(2);
    assert_eq!(d2[0], d0[0]);
    assert_ne!(d2[1], d0[1]);
    assert_eq!(d2[2], d0[2]);
}
