//! Digests of one slot's recurrent state, for exactness checks on a GPU (`legion serve
//! --prefix-state-digests`, #46): the bits of a state saved inside a prompt step against one
//! saved at a step's end, without dumping hundreds of MB.

use super::StatePool;

/// A 64-bit FNV-1a digest of `bytes`.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl StatePool {
    /// The digest ([`fnv1a`]) of each of `slot`'s state tensors, named by the tensor (per
    /// recurrent layer its conv history `conv_state-<il>` and delta-rule state
    /// `ssm_state-<il>`, then the per-layer embedding's history), and of its token history
    /// (`tokens`). Reads the slot back (from the device): the caller orders this after the work
    /// that writes it.
    pub fn slot_digests(&self, slot: usize) -> Vec<(String, u64)> {
        assert!(slot < self.n_slots, "slot {slot} out of range");
        let mut out: Vec<(String, u64)> = self
            .tensors()
            .map(|t| {
                // Slots in dim 3, each contiguous.
                let per = t.nb()[3];
                (t.name(), fnv1a(&t.bytes_at(slot * per, per)))
            })
            .collect();
        let tokens = self.peek_tokens(slot);
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        out.push(("tokens".to_string(), fnv1a(&bytes)));
        out
    }
}

#[cfg(test)]
mod tests;
