//! Tessera core: allocator, scheduler, and the engine step loop.
//!
//! This crate owns the serving engine's authoritative state and the traits that
//! decouple it from any concrete model runtime or scheduling policy. It depends
//! on **no** runtime crate — runtimes are injected through the [`ModelRuntime`]
//! trait defined here, keeping the dependency graph acyclic (see `CLAUDE.md`
//! §2.1).
//!
//! Build status: Layer 4 (prefix sharing). On top of safety-checked admission
//! and preemption, the engine now shares identical prompt-prefix blocks across
//! sequences via a refcounted prefix cache, with copy-on-write on divergent
//! writes ([`BlockAllocator::copy_on_write`], [`Sequence::fork`]). A real model
//! runtime arrives in the next layer.
#![forbid(unsafe_code)]

pub mod block;
pub mod engine;
mod ids;
pub mod metrics;
pub mod runtime;
pub mod scheduler;
pub mod sequence;

pub use block::{AllocError, BlockAllocator, BlockHash, PhysicalBlockId, BLOCK_SIZE};
pub use engine::{Engine, EngineConfig, EngineError, StepTrace};
pub use ids::{SeqId, TokenId};
pub use metrics::EngineMetrics;
pub use runtime::{ModelRuntime, RuntimeError};
pub use scheduler::{Fcfs, Priority, SchedulerPolicy, StepPlan};
pub use sequence::{SamplingParams, SeqState, Sequence};
