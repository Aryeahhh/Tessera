//! First-come, first-served admission.

use std::collections::VecDeque;

use super::{SchedulerPolicy, StepPlan};
use crate::block::BlockAllocator;
use crate::sequence::Sequence;

/// Admits waiting sequences in arrival order while the block budget allows,
/// after first reserving room for every running sequence to grow one block this
/// step.
///
/// Admission stops at the first waiting sequence that does not fit
/// (head-of-line blocking), so queue order is strictly honored. Growth safety
/// margins and preemption are added in Layer 3; this policy only guarantees the
/// budget is not exceeded *this* step.
#[derive(Debug, Default, Clone, Copy)]
pub struct Fcfs;

impl SchedulerPolicy for Fcfs {
    fn plan(
        &mut self,
        waiting: &VecDeque<Sequence>,
        running: &[Sequence],
        alloc: &BlockAllocator,
    ) -> StepPlan {
        let mut budget = alloc.free_blocks();

        // Reserve growth for the running set first: every running sequence may
        // need a fresh block to hold the token it produces this step.
        let mut decode = Vec::with_capacity(running.len());
        for seq in running {
            decode.push(seq.id());
            budget = budget.saturating_sub(seq.blocks_needed_for_next());
        }

        // Admit from the front of the queue while prefill fits the remainder.
        let mut prefill = Vec::new();
        for seq in waiting {
            let need = seq.blocks_needed_for_next();
            if need <= budget {
                prefill.push(seq.id());
                budget -= need;
            } else {
                break;
            }
        }

        StepPlan {
            prefill,
            decode,
            preempt: Vec::new(),
        }
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
