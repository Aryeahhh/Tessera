//! The model-runtime boundary.
//!
//! [`ModelRuntime`] is the single seam between the engine and whatever produces
//! next-token predictions — a deterministic mock today, a real tensor backend
//! later. The engine references only this trait, never a concrete runtime, so
//! the dependency direction stays acyclic (see `CLAUDE.md` §2.1).

use crate::{SeqId, TokenId};

/// Errors a runtime can return from a prefill or decode call.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// The runtime has no remaining continuation for this sequence.
    #[error("sequence {0:?} has no remaining output")]
    Exhausted(SeqId),

    /// The sequence was decoded before it was prefilled.
    #[error("sequence {0:?} was decoded before prefill")]
    NotPrefilled(SeqId),
}

/// Produces next-token predictions for the engine.
///
/// The contract is split into the two passes the engine distinguishes:
/// [`prefill`](ModelRuntime::prefill) consumes a full prompt for a newly
/// admitted sequence, and [`decode`](ModelRuntime::decode) advances an already
/// running sequence by one token. Both return the next token to append.
///
/// Layer 0 keeps the contract per-sequence and KV-agnostic. Block-table and
/// batched signatures (per `docs/SPEC.md §6`) arrive with the engine loop in
/// Layer 2; the prefill/decode split is fixed here so runtimes can be built and
/// tested against it now.
pub trait ModelRuntime {
    /// Run the full prompt for a freshly admitted sequence and return its first
    /// generated token.
    fn prefill(&mut self, seq: SeqId, prompt: &[TokenId]) -> Result<TokenId, RuntimeError>;

    /// Advance a running sequence by one token, given its most recent token.
    fn decode(&mut self, seq: SeqId, last: TokenId) -> Result<TokenId, RuntimeError>;
}
