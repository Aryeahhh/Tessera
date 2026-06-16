//! Tessera core: allocator, scheduler, and the engine step loop.
//!
//! This crate owns the serving engine's authoritative state and the traits that
//! decouple it from any concrete model runtime or scheduling policy. It depends
//! on **no** runtime crate — runtimes are injected through the [`ModelRuntime`]
//! trait defined here, keeping the dependency graph acyclic (see `CLAUDE.md`
//! §2.1).
//!
//! Build status: Layer 0 (foundation). Only the runtime contract and shared
//! identifier newtypes exist so far; the allocator, sequence, scheduler, and
//! engine modules arrive in later layers.
#![forbid(unsafe_code)]

mod ids;
pub mod runtime;

pub use ids::{SeqId, TokenId};
pub use runtime::{ModelRuntime, RuntimeError};
