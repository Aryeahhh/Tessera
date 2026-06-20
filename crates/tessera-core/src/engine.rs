//! The single-threaded, synchronous engine step loop.
//!
//! The [`Engine`] owns all authoritative serving state — the waiting queue, the
//! running set, the [`BlockAllocator`], the injected runtime, and the scheduling
//! policy — and advances it one deterministic step at a time:
//! `schedule → preempt → admit → execute → reconcile` (see `docs/SPEC.md §4.1`).
//! There is no shared mutable state and no locking; determinism is a feature, so
//! the loop is replayable from a script (the sim harness relies on this).

use std::collections::{HashSet, VecDeque};

use crate::block::{AllocError, BlockAllocator};
use crate::ids::{SeqId, TokenId};
use crate::metrics::EngineMetrics;
use crate::runtime::{ModelRuntime, RuntimeError};
use crate::scheduler::SchedulerPolicy;
use crate::sequence::{SeqState, Sequence};

/// Engine configuration knobs.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    /// Token that ends generation when produced. It is not appended to output.
    pub eos_token: Option<TokenId>,

    /// Preemption recovery threshold: a victim with `len() <= swap_threshold_tokens`
    /// is recovered by **recompute**, a longer one by **swap**. The block lifecycle
    /// is identical for both today (blocks are reclaimed and rebuilt on resume);
    /// the distinction drives the recovery accounting that a real KV runtime
    /// (Layer 5) turns into recompute-vs-copy. Defaults to [`usize::MAX`]
    /// (always recompute).
    pub swap_threshold_tokens: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            eos_token: None,
            swap_threshold_tokens: usize::MAX,
        }
    }
}

/// Errors raised while stepping the engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The runtime failed to produce a token.
    #[error(transparent)]
    Runtime(#[from] RuntimeError),

    /// The KV pool was exhausted mid-step. With a correct plan this cannot
    /// happen; it signals a scheduler bug.
    #[error(transparent)]
    Alloc(#[from] AllocError),

    /// A waiting sequence's footprint exceeds the whole pool, so it can never be
    /// scheduled — the engine would otherwise spin forever making no progress.
    #[error("a waiting sequence needs {needed} blocks but the pool holds only {capacity}")]
    Unschedulable { needed: usize, capacity: usize },
}

/// A record of what one step did, for tracing and tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StepTrace {
    /// Step index, starting at 0.
    pub step: u64,
    /// Sequences admitted this step (fresh prefill or resumed preemption).
    pub admitted: Vec<SeqId>,
    /// Sequences decoded this step.
    pub decoded: Vec<SeqId>,
    /// Sequences preempted this step.
    pub preempted: Vec<SeqId>,
    /// Sequences that finished this step.
    pub finished: Vec<SeqId>,
    /// Running-set size after reconciliation.
    pub running_after: usize,
    /// Free blocks after reconciliation.
    pub free_blocks_after: usize,
}

/// The serving engine: owns all authoritative state and drives the step loop.
#[derive(Debug)]
pub struct Engine<R, P> {
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    finished: Vec<Sequence>,
    allocator: BlockAllocator,
    runtime: R,
    policy: P,
    config: EngineConfig,
    metrics: EngineMetrics,
    step_count: u64,
}

impl<R, P> Engine<R, P>
where
    R: ModelRuntime,
    P: SchedulerPolicy,
{
    /// Create an engine over a KV pool, runtime, and scheduling policy.
    pub fn new(allocator: BlockAllocator, runtime: R, policy: P, config: EngineConfig) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
            finished: Vec::new(),
            allocator,
            runtime,
            policy,
            config,
            metrics: EngineMetrics::default(),
            step_count: 0,
        }
    }

    /// Queue a sequence for admission.
    pub fn submit(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    /// True when no work remains (nothing waiting or running).
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Snapshot of cumulative metrics.
    #[must_use]
    pub fn metrics(&self) -> EngineMetrics {
        self.metrics
    }

    /// Sequences that have completed, in completion order.
    #[must_use]
    pub fn finished(&self) -> &[Sequence] {
        &self.finished
    }

    /// Number of running sequences.
    #[must_use]
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Number of waiting sequences.
    #[must_use]
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Free blocks currently available in the KV pool.
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.allocator.free_blocks()
    }

    /// Run one engine step: schedule, preempt, admit, execute, reconcile.
    pub fn step(&mut self) -> Result<StepTrace, EngineError> {
        let plan = self
            .policy
            .plan(&self.waiting, &self.running, &self.allocator);

        let admit: HashSet<SeqId> = plan.prefill.iter().copied().collect();
        let decode: HashSet<SeqId> = plan.decode.iter().copied().collect();

        // Free victims' blocks first so admission/growth always fits the pool.
        let preempted = self.preempt(&plan.preempt);
        let admitted = self.admit(&admit);
        self.metrics.sequences_admitted += admitted.len() as u64;

        if self.running.is_empty() && !self.waiting.is_empty() {
            let needed = self
                .waiting
                .iter()
                .map(Sequence::footprint_after_next)
                .min()
                .unwrap_or(0);
            return Err(EngineError::Unschedulable {
                needed,
                capacity: self.allocator.total_blocks(),
            });
        }

        let decoded = self.execute(&admit, &decode)?;
        let finished = self.reconcile();
        self.metrics.sequences_finished += finished.len() as u64;

        let trace = StepTrace {
            step: self.step_count,
            admitted,
            decoded,
            preempted,
            finished,
            running_after: self.running.len(),
            free_blocks_after: self.allocator.free_blocks(),
        };
        self.step_count += 1;
        self.metrics.steps += 1;
        Ok(trace)
    }

    /// Step until the engine is idle, returning the trace of every step.
    pub fn run_to_idle(&mut self) -> Result<Vec<StepTrace>, EngineError> {
        let mut traces = Vec::new();
        while !self.is_idle() {
            traces.push(self.step()?);
        }
        Ok(traces)
    }

    /// Evict the victim sequences: reclaim their blocks and return them to the
    /// waiting queue as `Preempted`, so a later step can resume them.
    fn preempt(&mut self, victims: &[SeqId]) -> Vec<SeqId> {
        if victims.is_empty() {
            return Vec::new();
        }
        let vset: HashSet<SeqId> = victims.iter().copied().collect();

        let mut kept = Vec::with_capacity(self.running.len());
        let mut evicted = Vec::new();
        for seq in self.running.drain(..) {
            if vset.contains(&seq.id()) {
                evicted.push(seq);
            } else {
                kept.push(seq);
            }
        }
        self.running = kept;

        let mut ids = Vec::with_capacity(evicted.len());
        for mut seq in evicted {
            self.metrics.preemptions += 1;
            if seq.len() <= self.config.swap_threshold_tokens {
                self.metrics.preemptions_recompute += 1;
            } else {
                self.metrics.preemptions_swap += 1;
            }
            seq.free_blocks(&mut self.allocator);
            seq.set_state(SeqState::Preempted);
            ids.push(seq.id());
            self.waiting.push_front(seq);
        }
        ids
    }

    /// Move admitted sequences from the waiting queue into the running set,
    /// preserving queue order for those left behind.
    fn admit(&mut self, admit: &HashSet<SeqId>) -> Vec<SeqId> {
        if admit.is_empty() {
            return Vec::new();
        }
        let mut admitted = Vec::with_capacity(admit.len());
        let mut remaining = VecDeque::with_capacity(self.waiting.len());
        for mut seq in self.waiting.drain(..) {
            if admit.contains(&seq.id()) {
                seq.set_state(SeqState::Running);
                admitted.push(seq.id());
                self.running.push(seq);
            } else {
                remaining.push_back(seq);
            }
        }
        self.waiting = remaining;
        admitted
    }

    /// Run prefill for fresh admissions, decode for resumed and running ones.
    fn execute(
        &mut self,
        admit: &HashSet<SeqId>,
        decode: &HashSet<SeqId>,
    ) -> Result<Vec<SeqId>, EngineError> {
        let mut decoded = Vec::new();
        for idx in 0..self.running.len() {
            let id = self.running[idx].id();
            if admit.contains(&id) {
                if self.running[idx].output_tokens().is_empty() {
                    // Fresh admission: share any cached prompt-prefix blocks,
                    // then prefill the prompt and emit the first token. The clone
                    // is once per sequence, off the decode hot path.
                    let shared = self.running[idx].share_prefix(&mut self.allocator)?;
                    self.metrics.blocks_shared += shared as u64;
                    let prompt = self.running[idx].prompt_tokens().to_vec();
                    let token = self.runtime.prefill(id, &prompt)?;
                    self.apply_token(idx, token)?;
                } else {
                    // Resume a preempted sequence: its blocks were reclaimed, so
                    // the next decode rebuilds them and continues generation
                    // exactly where it left off (output is unchanged).
                    let last = self.last_token(idx);
                    let token = self.runtime.decode(id, last)?;
                    self.apply_token(idx, token)?;
                }
            } else if decode.contains(&id) {
                let last = self.last_token(idx);
                let token = self.runtime.decode(id, last)?;
                self.apply_token(idx, token)?;
                decoded.push(id);
            }
        }
        Ok(decoded)
    }

    /// Reclaim the blocks of finished sequences, preserving running-set order.
    fn reconcile(&mut self) -> Vec<SeqId> {
        let (done, still): (Vec<Sequence>, Vec<Sequence>) = self
            .running
            .drain(..)
            .partition(|seq| seq.state() == SeqState::Finished);
        self.running = still;

        let mut finished_ids = Vec::with_capacity(done.len());
        for mut seq in done {
            seq.free_blocks(&mut self.allocator);
            finished_ids.push(seq.id());
            self.finished.push(seq);
        }
        finished_ids
    }

    /// Most recent token of running sequence `idx` (its last output token).
    fn last_token(&self, idx: usize) -> TokenId {
        self.running[idx]
            .output_tokens()
            .last()
            .copied()
            .expect("a decoded or resumed sequence has at least one prior token")
    }

    /// Apply a produced token to running sequence `idx`: detect EOS / max-tokens,
    /// append, and grow the block table for the new token.
    fn apply_token(&mut self, idx: usize, token: TokenId) -> Result<(), AllocError> {
        if self.config.eos_token == Some(token) {
            self.running[idx].set_state(SeqState::Finished);
            return Ok(());
        }

        let seq = &mut self.running[idx];
        seq.push_output(token);
        seq.reserve(&mut self.allocator)?;
        self.metrics.tokens_generated += 1;

        let max = seq.sampling().max_tokens;
        if max > 0 && seq.output_tokens().len() >= max {
            seq.set_state(SeqState::Finished);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::Fcfs;
    use crate::sequence::SamplingParams;

    /// A runtime that always emits the same token — handy for exercising the
    /// completion paths without scripting.
    struct ConstRuntime(TokenId);

    impl ModelRuntime for ConstRuntime {
        fn prefill(&mut self, _seq: SeqId, _prompt: &[TokenId]) -> Result<TokenId, RuntimeError> {
            Ok(self.0)
        }

        fn decode(&mut self, _seq: SeqId, _last: TokenId) -> Result<TokenId, RuntimeError> {
            Ok(self.0)
        }
    }

    #[test]
    fn max_tokens_bounds_generation() {
        let mut engine = Engine::new(
            BlockAllocator::new(8),
            ConstRuntime(TokenId(42)),
            Fcfs,
            EngineConfig::default(),
        );
        let sampling = SamplingParams {
            max_tokens: 3,
            ..SamplingParams::default()
        };
        engine.submit(Sequence::new(SeqId(1), vec![TokenId(1)], sampling, 0));

        engine.run_to_idle().unwrap();

        assert!(engine.is_idle());
        assert_eq!(engine.finished().len(), 1);
        assert_eq!(engine.finished()[0].output_tokens().len(), 3);
        assert_eq!(engine.metrics().tokens_generated, 3);
        assert_eq!(engine.free_blocks(), 8, "blocks reclaimed on completion");
    }

    #[test]
    fn eos_token_finishes_without_appending() {
        let mut engine = Engine::new(
            BlockAllocator::new(8),
            ConstRuntime(TokenId(7)),
            Fcfs,
            EngineConfig {
                eos_token: Some(TokenId(7)),
                ..Default::default()
            },
        );
        engine.submit(Sequence::new(
            SeqId(1),
            vec![TokenId(1)],
            SamplingParams::default(),
            0,
        ));

        engine.run_to_idle().unwrap();

        assert_eq!(engine.finished().len(), 1);
        assert!(engine.finished()[0].output_tokens().is_empty());
        assert_eq!(engine.metrics().tokens_generated, 0);
    }

    #[test]
    fn oversized_request_is_unschedulable() {
        // A prompt that needs more blocks than the whole pool can never run.
        let mut engine = Engine::new(
            BlockAllocator::new(1),
            ConstRuntime(TokenId(9)),
            Fcfs,
            EngineConfig::default(),
        );
        let big_prompt = (0..40u32).map(TokenId).collect();
        engine.submit(Sequence::new(
            SeqId(1),
            big_prompt,
            SamplingParams::default(),
            0,
        ));

        assert!(matches!(
            engine.step(),
            Err(EngineError::Unschedulable { capacity: 1, .. })
        ));
    }
}
