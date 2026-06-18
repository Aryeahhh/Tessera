//! First-come, first-served scheduling.

use std::collections::VecDeque;

use super::{select_plan, Candidate, SchedulerPolicy, StepPlan};
use crate::block::BlockAllocator;
use crate::sequence::Sequence;

/// Admits and retains sequences in arrival order. Under memory pressure the
/// newest (latest-arrived) running sequences are preempted first, so the oldest
/// requests are protected and the queue drains in order. Admission stops at the
/// first request that does not fit (head-of-line blocking).
#[derive(Debug, Default, Clone, Copy)]
pub struct Fcfs;

impl SchedulerPolicy for Fcfs {
    fn plan(
        &mut self,
        waiting: &VecDeque<Sequence>,
        running: &[Sequence],
        alloc: &BlockAllocator,
    ) -> StepPlan {
        let wvec: Vec<&Sequence> = waiting.iter().collect();
        let mut order: Vec<Candidate> = (0..running.len())
            .map(Candidate::Running)
            .chain((0..wvec.len()).map(Candidate::Waiting))
            .collect();
        // Oldest arrival first; on a tie prefer the already-running sequence.
        order.sort_by_key(|&cand| match cand {
            Candidate::Running(i) => (running[i].arrival(), false),
            Candidate::Waiting(i) => (wvec[i].arrival(), true),
        });
        select_plan(running, &wvec, &order, alloc.total_blocks())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{SeqId, TokenId};
    use crate::sequence::SamplingParams;

    fn waiting_seq(id: u64, prompt_len: usize) -> Sequence {
        let prompt = (0..prompt_len as u32).map(TokenId).collect();
        Sequence::new(SeqId(id), prompt, SamplingParams::default(), 0)
    }

    #[test]
    fn admits_in_order_until_budget_exhausted() {
        // Pool of 2 blocks; three one-block prompts queued.
        let alloc = BlockAllocator::new(2);
        let mut waiting = VecDeque::new();
        waiting.push_back(waiting_seq(1, 4));
        waiting.push_back(waiting_seq(2, 4));
        waiting.push_back(waiting_seq(3, 4));

        let plan = Fcfs.plan(&waiting, &[], &alloc);

        assert_eq!(plan.prefill, vec![SeqId(1), SeqId(2)]);
        assert!(plan.decode.is_empty());
        assert!(plan.preempt.is_empty());
    }

    #[test]
    fn stops_at_first_oversized_prompt() {
        // One free block, but the head prompt spans two — admit nothing.
        let alloc = BlockAllocator::new(1);
        let mut waiting = VecDeque::new();
        waiting.push_back(waiting_seq(1, 20));

        let plan = Fcfs.plan(&waiting, &[], &alloc);
        assert!(plan.prefill.is_empty());
    }
}
