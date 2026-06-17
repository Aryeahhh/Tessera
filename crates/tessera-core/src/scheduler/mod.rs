//! Scheduling policies: deciding the per-step plan under the block budget.
//!
//! A [`SchedulerPolicy`] is pure decision-making — it inspects the waiting queue,
//! the running set, and the free-block count, and returns a [`StepPlan`]. It does
//! not touch the runtime, allocate blocks, or mutate sequences; the engine owns
//! all of that (see `CLAUDE.md` §2.2 and `docs/SPEC.md §5.3`).

mod fcfs;

use std::collections::VecDeque;

pub use fcfs::Fcfs;

use crate::block::BlockAllocator;
use crate::ids::SeqId;
use crate::sequence::Sequence;

/// The work a single engine step should perform.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StepPlan {
    /// Newly admitted sequences to run prefill for this step.
    pub prefill: Vec<SeqId>,
    /// Running sequences to advance by one token this step.
    pub decode: Vec<SeqId>,
    /// Sequences to evict this step. Always empty until preemption (Layer 3).
    pub preempt: Vec<SeqId>,
}

/// A pluggable scheduling policy.
///
/// Given the waiting queue, the running set, and the free-block budget, a policy
/// returns the [`StepPlan`] for the next step. Implementations must never plan
/// allocations that exceed the budget — the engine treats overspend as a bug.
pub trait SchedulerPolicy {
    /// Produce the plan for the next step.
    fn plan(
        &mut self,
        waiting: &VecDeque<Sequence>,
        running: &[Sequence],
        alloc: &BlockAllocator,
    ) -> StepPlan;
}
