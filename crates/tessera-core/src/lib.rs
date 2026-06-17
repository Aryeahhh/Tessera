//! Tessera core: allocator, scheduler, and the engine step loop.
//!
//! This crate owns the serving engine's authoritative state and the traits that
//! decouple it from any concrete model runtime or scheduling policy. It depends
//! on **no** runtime crate — runtimes are injected through the [`ModelRuntime`]
//! trait defined here, keeping the dependency graph acyclic (see `CLAUDE.md`
//! §2.1).
//!
//! Build status: Layer 2 (execution loop). The continuous-batching [`Engine`]
//! drives the paged-KV [`BlockAllocator`] and [`Sequence`] block table under a
//! pluggable [`SchedulerPolicy`] (FCFS today). Scheduling refinements, prefix
//! sharing, and a real runtime arrive in later layers.
#![forbid(unsafe_code)]

pub mod block;
pub mod engine;
mod ids;
pub mod metrics;
pub mod runtime;
pub mod scheduler;
pub mod sequence;

pub use block::{AllocError, BlockAllocator, PhysicalBlockId, BLOCK_SIZE};
pub use engine::{Engine, EngineConfig, EngineError, StepTrace};
pub use ids::{SeqId, TokenId};
pub use metrics::EngineMetrics;
pub use runtime::{ModelRuntime, RuntimeError};
pub use scheduler::{Fcfs, SchedulerPolicy, StepPlan};
pub use sequence::{SamplingParams, SeqState, Sequence};
