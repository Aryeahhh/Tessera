//! Tessera core: allocator, scheduler, and the engine step loop.
//!
//! This crate owns the serving engine's authoritative state and the traits that
//! decouple it from any concrete model runtime or scheduling policy. It depends
//! on **no** runtime crate — runtimes are injected through the [`ModelRuntime`]
//! trait defined here, keeping the dependency graph acyclic (see `CLAUDE.md`
//! §2.1).
//!
//! Build status: Layer 1 (memory). The paged-KV [`BlockAllocator`] and
//! [`Sequence`] block table now exist alongside the runtime contract; the
//! scheduler and engine loop arrive in later layers.
#![forbid(unsafe_code)]

pub mod block;
mod ids;
pub mod runtime;
pub mod sequence;

pub use block::{AllocError, BlockAllocator, PhysicalBlockId, BLOCK_SIZE};
pub use ids::{SeqId, TokenId};
pub use runtime::{ModelRuntime, RuntimeError};
pub use sequence::{SamplingParams, SeqState, Sequence};
