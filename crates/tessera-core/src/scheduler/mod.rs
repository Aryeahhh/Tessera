//! Scheduling policies: deciding the per-step plan under the block budget.
//!
//! A [`SchedulerPolicy`] is pure decision-making — it inspects the waiting queue,
//! the running set, and the free-block count, and returns a [`StepPlan`]. It does
//! not touch the runtime, allocate blocks, or mutate sequences; the engine owns
//! all of that (see `CLAUDE.md` §2.2 and `docs/SPEC.md §5.3`).

mod fcfs;
mod priority;

use std::collections::VecDeque;

pub use fcfs::Fcfs;
pub use priority::Priority;

use crate::block::BlockAllocator;
use crate::ids::SeqId;
use crate::sequence::Sequence;

/// The work a single engine step should perform.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StepPlan {
    /// Sequences to admit and run (fresh prefill, or resume of a preempted one).
    pub prefill: Vec<SeqId>,
    /// Running sequences to advance by one token this step.
    pub decode: Vec<SeqId>,
    /// Running sequences to evict this step to reclaim their blocks.
    pub preempt: Vec<SeqId>,
}

/// A pluggable scheduling policy.
///
/// Given the waiting queue, the running set, and the free-block budget, a policy
/// returns the [`StepPlan`] for the next step. Implementations must never plan
/// allocations that exceed the pool — the engine treats overspend as a bug.
pub trait SchedulerPolicy {
    /// Produce the plan for the next step.
    fn plan(
        &mut self,
        waiting: &VecDeque<Sequence>,
        running: &[Sequence],
        alloc: &BlockAllocator,
    ) -> StepPlan;
}

/// A scheduling candidate: an index into the running set or the waiting list.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Candidate {
    Running(usize),
    Waiting(usize),
}

/// Select the active set greedily in `order`, packing post-step block footprints
/// into `total_blocks`, and classify the result into a [`StepPlan`].
///
/// Candidates are visited highest-value first; each is selected while the
/// cumulative footprint still fits, and the first that does not fit stops
/// selection (head-of-line). A running candidate left unselected becomes a
/// preemption, so a higher-value waiting candidate can displace a lower-value
/// running one. Because `Σ selected footprints ≤ total_blocks` and the engine
/// frees preempted blocks before allocating, applying the plan never exceeds the
/// pool.
pub(crate) fn select_plan(
    running: &[Sequence],
    waiting: &[&Sequence],
    order: &[Candidate],
    total_blocks: usize,
) -> StepPlan {
    let mut used = 0usize;
    let mut run_selected = vec![false; running.len()];
    let mut wait_selected = vec![false; waiting.len()];

    for &cand in order {
        let footprint = match cand {
            Candidate::Running(i) => running[i].footprint_after_next(),
            Candidate::Waiting(i) => waiting[i].footprint_after_next(),
        };
        if used + footprint <= total_blocks {
            used += footprint;
            match cand {
                Candidate::Running(i) => run_selected[i] = true,
                Candidate::Waiting(i) => wait_selected[i] = true,
            }
        } else {
            break;
        }
    }

    let mut decode = Vec::new();
    let mut preempt = Vec::new();
    for (i, seq) in running.iter().enumerate() {
        if run_selected[i] {
            decode.push(seq.id());
        } else {
            preempt.push(seq.id());
        }
    }
    let prefill = waiting
        .iter()
        .enumerate()
        .filter(|(i, _)| wait_selected[*i])
        .map(|(_, seq)| seq.id())
        .collect();

    StepPlan {
        prefill,
        decode,
        preempt,
    }
}
