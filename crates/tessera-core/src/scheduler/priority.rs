//! Priority scheduling with preemption.

use std::cmp::Reverse;
use std::collections::VecDeque;

use super::{select_plan, Candidate, SchedulerPolicy, StepPlan};
use crate::block::BlockAllocator;
use crate::sequence::Sequence;

/// Admits the highest-priority sequences first and preempts the lowest-priority
/// running sequences first, so a high-priority arrival can displace a
/// lower-priority running sequence under memory pressure. Ties fall back to
/// arrival order (FCFS), preferring sequences already running to limit churn.
#[derive(Debug, Default, Clone, Copy)]
pub struct Priority;

impl SchedulerPolicy for Priority {
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
        // Highest priority first; tie -> already running first; tie -> oldest first.
        order.sort_by_key(|&cand| match cand {
            Candidate::Running(i) => (Reverse(running[i].priority()), false, running[i].arrival()),
            Candidate::Waiting(i) => (Reverse(wvec[i].priority()), true, wvec[i].arrival()),
        });
        select_plan(running, &wvec, &order, alloc.total_blocks())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{SeqId, TokenId};
    use crate::sequence::SamplingParams;

    fn seq(id: u64, prompt_len: usize, priority: u8) -> Sequence {
        let prompt = (0..prompt_len as u32).map(TokenId).collect();
        Sequence::new(SeqId(id), prompt, SamplingParams::default(), priority)
    }

    #[test]
    fn high_priority_admitted_first_under_budget() {
        // Room for one block; the second prompt outranks the first.
        let alloc = BlockAllocator::new(1);
        let mut waiting = VecDeque::new();
        waiting.push_back(seq(1, 4, 0));
        waiting.push_back(seq(2, 4, 9));

        let plan = Priority.plan(&waiting, &[], &alloc);
        assert_eq!(plan.prefill, vec![SeqId(2)]);
    }

    #[test]
    fn high_priority_waiting_preempts_low_priority_running() {
        // One block of capacity, one low-priority sequence running in it. A
        // higher-priority arrival must preempt it to be admitted this step.
        let alloc = BlockAllocator::new(1);
        let mut running_seq = seq(1, 4, 0);
        running_seq.set_state(crate::sequence::SeqState::Running);
        running_seq.push_output(TokenId(50));
        let running = [running_seq];

        let mut waiting = VecDeque::new();
        waiting.push_back(seq(2, 4, 9));

        let plan = Priority.plan(&waiting, &running, &alloc);
        assert_eq!(plan.prefill, vec![SeqId(2)]);
        assert_eq!(plan.preempt, vec![SeqId(1)]);
    }
}
