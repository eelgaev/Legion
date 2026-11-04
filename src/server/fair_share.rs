//! `--fair-share` (#625): whom a request counts against in the engine's waiting queue
//! ([`crate::engine::queue`]). Only a hash of the identity ([`ShareKey`]) leaves this module.

use crate::engine::queue::ShareKey;

/// `--fair-share`: what identifies a client for fair share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum FairShare {
    /// No fair share: requests are admitted in the scheduling policy's order alone.
    #[default]
    Off,
    /// The request's API key (`Authorization: Bearer` or `x-api-key`): with `--api-key`, the
    /// accepted key; without, whatever key the client presents (e.g. a proxy's per-user keys).
    ApiKey,
    /// The request's end user: OpenAI's `user` field, Anthropic's `metadata.user_id`.
    User,
}

/// The caller's API key, hashed: the auth middleware puts it in the request's extensions
/// under `--fair-share api-key`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallerKey(pub ShareKey);

/// The fair-share key of a request under `mode`: the caller's API key (from the auth
/// middleware) or the request's `user`. `None`, one shared key, when fair share is off or the
/// request carries no such identity.
pub fn share_key(
    mode: FairShare,
    caller: Option<CallerKey>,
    user: Option<&str>,
) -> Option<ShareKey> {
    match mode {
        FairShare::Off => None,
        FairShare::ApiKey => caller.map(|c| c.0),
        FairShare::User => user.filter(|u| !u.is_empty()).map(ShareKey::of),
    }
}

#[cfg(test)]
mod tests;
