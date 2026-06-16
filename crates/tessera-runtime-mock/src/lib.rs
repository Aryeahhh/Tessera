//! A deterministic [`ModelRuntime`] for building and testing the engine with
//! zero ML dependency.
//!
//! [`MockRuntime`] replays a script: each sequence is given a fixed list of
//! tokens to emit, in order — one per prefill/decode call. Because output is
//! fully scripted, the scheduler and allocator can be exercised reproducibly
//! before any real model exists (see `CLAUDE.md` §2.2, Layer 0).
#![forbid(unsafe_code)]

use std::collections::HashMap;

use tessera_core::{ModelRuntime, RuntimeError, SeqId, TokenId};

/// A runtime that emits pre-scripted tokens per sequence.
#[derive(Debug, Default)]
pub struct MockRuntime {
    scripts: HashMap<SeqId, Script>,
}

#[derive(Debug)]
struct Script {
    tokens: Vec<TokenId>,
    cursor: usize,
    prefilled: bool,
}

impl MockRuntime {
    /// Create an empty runtime with no scripted sequences.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the tokens `seq` should emit, in order. The first token is
    /// returned by [`prefill`](ModelRuntime::prefill); each subsequent token by
    /// a [`decode`](ModelRuntime::decode) call.
    pub fn script(&mut self, seq: SeqId, tokens: impl IntoIterator<Item = TokenId>) {
        self.scripts.insert(
            seq,
            Script {
                tokens: tokens.into_iter().collect(),
                cursor: 0,
                prefilled: false,
            },
        );
    }

    fn advance(&mut self, seq: SeqId) -> Result<TokenId, RuntimeError> {
        let script = self
            .scripts
            .get_mut(&seq)
            .ok_or(RuntimeError::Exhausted(seq))?;
        let token = script
            .tokens
            .get(script.cursor)
            .copied()
            .ok_or(RuntimeError::Exhausted(seq))?;
        script.cursor += 1;
        Ok(token)
    }
}

impl ModelRuntime for MockRuntime {
    fn prefill(&mut self, seq: SeqId, _prompt: &[TokenId]) -> Result<TokenId, RuntimeError> {
        if let Some(script) = self.scripts.get_mut(&seq) {
            script.prefilled = true;
        }
        self.advance(seq)
    }

    fn decode(&mut self, seq: SeqId, _last: TokenId) -> Result<TokenId, RuntimeError> {
        match self.scripts.get(&seq) {
            Some(script) if !script.prefilled => Err(RuntimeError::NotPrefilled(seq)),
            _ => self.advance(seq),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: u64) -> SeqId {
        SeqId(n)
    }

    fn tok(n: u32) -> TokenId {
        TokenId(n)
    }

    #[test]
    fn replays_scripted_tokens_in_order() {
        let mut rt = MockRuntime::new();
        rt.script(seq(1), [tok(10), tok(11), tok(12)]);

        assert_eq!(rt.prefill(seq(1), &[tok(1), tok(2)]).unwrap(), tok(10));
        assert_eq!(rt.decode(seq(1), tok(10)).unwrap(), tok(11));
        assert_eq!(rt.decode(seq(1), tok(11)).unwrap(), tok(12));
    }

    #[test]
    fn exhausted_script_errors() {
        let mut rt = MockRuntime::new();
        rt.script(seq(1), [tok(10)]);

        assert_eq!(rt.prefill(seq(1), &[]).unwrap(), tok(10));
        assert!(matches!(
            rt.decode(seq(1), tok(10)),
            Err(RuntimeError::Exhausted(_))
        ));
    }

    #[test]
    fn decode_before_prefill_errors() {
        let mut rt = MockRuntime::new();
        rt.script(seq(1), [tok(10), tok(11)]);

        assert!(matches!(
            rt.decode(seq(1), tok(0)),
            Err(RuntimeError::NotPrefilled(_))
        ));
    }

    #[test]
    fn determinism_two_runs_match() {
        let run = || {
            let mut rt = MockRuntime::new();
            rt.script(seq(42), [tok(7), tok(8), tok(9)]);
            let a = rt.prefill(seq(42), &[tok(1)]).unwrap();
            let b = rt.decode(seq(42), a).unwrap();
            let c = rt.decode(seq(42), b).unwrap();
            (a, b, c)
        };
        assert_eq!(run(), run());
    }
}
