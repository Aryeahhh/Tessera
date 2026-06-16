//! Strongly-typed identifiers shared across the engine.
//!
//! Newtypes keep illegal mixing of identifiers unrepresentable: a token value
//! can never be passed where a sequence id is expected, and vice versa.

/// Identifier for a vocabulary token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TokenId(pub u32);

/// Identifier for an in-flight sequence (one generation request).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(pub u64);
