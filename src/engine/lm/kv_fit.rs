//! Whether one request fits the KV cache at all (#125): a sequence reserves KV blocks for its
//! prompt and up to `max_tokens` generated tokens, so a request that needs more than the whole
//! pool can never run. That is the client's error (HTTP 400, as a prompt over the context),
//! refused before the request queues; the scheduler's admission checks the same numbers.

use crate::engine::EngineError;
use crate::model::core::KV_BLOCK;

/// KV blocks a sequence of `prompt` tokens generating up to `max_tokens` reserves: its whole
/// length, at most `max_ctx`, and the `spec_k` positions a verify step writes past the last
/// one it keeps.
pub fn blocks(prompt: usize, max_tokens: u32, max_ctx: usize, spec_k: usize) -> usize {
    let max_len = (prompt + max_tokens as usize).min(max_ctx);
    (max_len + spec_k).div_ceil(KV_BLOCK)
}

/// [`blocks`] against a pool of `pool_blocks`: [`EngineError::KvOverflow`] when it needs more.
pub fn check(
    prompt: usize,
    max_tokens: u32,
    max_ctx: usize,
    spec_k: usize,
    pool_blocks: usize,
) -> Result<(), EngineError> {
    let need = blocks(prompt, max_tokens, max_ctx, spec_k);
    if need > pool_blocks {
        return Err(EngineError::KvOverflow {
            need: need * KV_BLOCK,
            holds: pool_blocks * KV_BLOCK,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
