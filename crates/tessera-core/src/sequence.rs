//! Sequences and their logical→physical block mapping.
//!
//! A [`Sequence`] is one in-flight generation request. It owns its tokens and a
//! **block table** — a `Vec<PhysicalBlockId>` mapping logical block index to a
//! physical block in the KV pool. Growth and release go through the
//! [`BlockAllocator`], which remains the sole owner of refcounts (see
//! `docs/SPEC.md §5.1`).

use std::time::Instant;

use crate::block::{blocks_for_len, AllocError, BlockAllocator, PhysicalBlockId, BLOCK_SIZE};
use crate::ids::{SeqId, TokenId};

/// Lifecycle state of a sequence.
///
/// Transitions are owned by the engine (Layer 2); this type only records the
/// current state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqState {
    /// Queued, not yet admitted to the running set.
    Waiting,
    /// Admitted and advancing one token per engine step.
    Running,
    /// Evicted under memory pressure; eligible for readmission.
    Preempted,
    /// Completed (EOS, max tokens, or stop condition).
    Finished,
}

/// Token-sampling configuration for a request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature; `0.0` selects greedy decoding.
    pub temperature: f32,
    /// Nucleus (top-p) cutoff in `(0.0, 1.0]`.
    pub top_p: f32,
    /// Maximum tokens to generate; `0` means unbounded (caller enforces).
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            max_tokens: 0,
        }
    }
}

/// A single generation request and its KV block mapping.
#[derive(Debug)]
pub struct Sequence {
    id: SeqId,
    state: SeqState,
    prompt_tokens: Vec<TokenId>,
    output_tokens: Vec<TokenId>,
    block_table: Vec<PhysicalBlockId>,
    priority: u8,
    sampling: SamplingParams,
    arrival: Instant,
    first_token_at: Option<Instant>,
}

impl Sequence {
    /// Create a waiting sequence from its prompt.
    #[must_use]
    pub fn new(
        id: SeqId,
        prompt_tokens: Vec<TokenId>,
        sampling: SamplingParams,
        priority: u8,
    ) -> Self {
        Self {
            id,
            state: SeqState::Waiting,
            prompt_tokens,
            output_tokens: Vec::new(),
            block_table: Vec::new(),
            priority,
            sampling,
            arrival: Instant::now(),
            first_token_at: None,
        }
    }

    /// Sequence identifier.
    #[must_use]
    pub fn id(&self) -> SeqId {
        self.id
    }

    /// Current lifecycle state.
    #[must_use]
    pub fn state(&self) -> SeqState {
        self.state
    }

    /// Prompt tokens (the input prefix).
    #[must_use]
    pub fn prompt_tokens(&self) -> &[TokenId] {
        &self.prompt_tokens
    }

    /// Generated output tokens, in order.
    #[must_use]
    pub fn output_tokens(&self) -> &[TokenId] {
        &self.output_tokens
    }

    /// The block table: logical block index → physical block.
    #[must_use]
    pub fn block_table(&self) -> &[PhysicalBlockId] {
        &self.block_table
    }

    /// Scheduling priority (higher preempts lower).
    #[must_use]
    pub fn priority(&self) -> u8 {
        self.priority
    }

    /// Sampling configuration.
    #[must_use]
    pub fn sampling(&self) -> SamplingParams {
        self.sampling
    }

    /// Arrival time, used for FCFS ordering and TTFT.
    #[must_use]
    pub fn arrival(&self) -> Instant {
        self.arrival
    }

    /// Time the first output token was produced, if any (for TTFT).
    #[must_use]
    pub fn first_token_at(&self) -> Option<Instant> {
        self.first_token_at
    }

    /// Total logical token length (prompt + generated) — what the KV holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// True when the sequence holds no tokens at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Set the lifecycle state. Engine-owned in Layer 2; minimal here.
    pub fn set_state(&mut self, state: SeqState) {
        self.state = state;
    }

    /// Append a generated token, recording time-to-first-token once.
    pub fn push_output(&mut self, token: TokenId) {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(Instant::now());
        }
        self.output_tokens.push(token);
    }

    /// Physical block and in-block offset for logical token position `pos`,
    /// or `None` if the block table does not yet cover that position.
    #[must_use]
    pub fn locate(&self, pos: usize) -> Option<(PhysicalBlockId, usize)> {
        let block = *self.block_table.get(pos / BLOCK_SIZE)?;
        Some((block, pos % BLOCK_SIZE))
    }

    /// Additional blocks the table needs to cover `len` tokens.
    #[must_use]
    pub fn blocks_needed(&self, len: usize) -> usize {
        blocks_for_len(len).saturating_sub(self.block_table.len())
    }

    /// Additional blocks required for the one token this sequence will produce
    /// next step (covering logical length `len() + 1`).
    ///
    /// Used by the scheduler to size the per-step budget uniformly across newly
    /// admitted (prefill) and already running (decode) sequences.
    #[must_use]
    pub fn blocks_needed_for_next(&self) -> usize {
        blocks_for_len(self.len() + 1).saturating_sub(self.block_table.len())
    }

    /// Total blocks this sequence will hold after producing its next token
    /// (covering logical length `len() + 1`), independent of what it holds now.
    ///
    /// The scheduler packs these footprints into the pool to size the active set;
    /// it is the same value whether the sequence is running, freshly waiting, or
    /// a preempted sequence whose blocks were reclaimed.
    #[must_use]
    pub fn footprint_after_next(&self) -> usize {
        blocks_for_len(self.len() + 1)
    }

    /// Grow the block table to cover the sequence's current length, allocating
    /// from `alloc`. Returns the number of blocks added.
    ///
    /// All-or-nothing: if the pool cannot satisfy the full requirement the table
    /// is left untouched and [`AllocError::Exhausted`] is returned, so a caller
    /// deferring admission never leaks partially allocated blocks.
    pub fn reserve(&mut self, alloc: &mut BlockAllocator) -> Result<usize, AllocError> {
        let needed = self.blocks_needed(self.len());
        if needed == 0 {
            return Ok(0);
        }
        if alloc.free_blocks() < needed {
            return Err(AllocError::Exhausted);
        }
        for _ in 0..needed {
            // Availability was checked above, so this allocation cannot fail.
            let block = alloc.allocate_block()?;
            self.block_table.push(block);
        }
        Ok(needed)
    }

    /// Return all of this sequence's blocks to the allocator and clear the table.
    pub fn free_blocks(&mut self, alloc: &mut BlockAllocator) {
        for &block in &self.block_table {
            alloc.free_block(block);
        }
        self.block_table.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(n: usize) -> Vec<TokenId> {
        (0..n as u32).map(TokenId).collect()
    }

    #[test]
    fn locate_maps_position_to_block_and_offset() {
        let mut alloc = BlockAllocator::new(8);
        let mut seq = Sequence::new(
            SeqId(1),
            tokens(BLOCK_SIZE + 3),
            SamplingParams::default(),
            0,
        );
        seq.reserve(&mut alloc).unwrap();
        assert_eq!(seq.block_table().len(), 2);

        assert_eq!(seq.locate(0), Some((seq.block_table()[0], 0)));
        assert_eq!(seq.locate(BLOCK_SIZE), Some((seq.block_table()[1], 0)));
        assert_eq!(seq.locate(BLOCK_SIZE + 2), Some((seq.block_table()[1], 2)));
        assert_eq!(seq.locate(2 * BLOCK_SIZE), None);
    }

    #[test]
    fn allocate_grow_free_round_trip() {
        let total = 16;
        let mut alloc = BlockAllocator::new(total);
        let mut seq = Sequence::new(SeqId(7), tokens(BLOCK_SIZE), SamplingParams::default(), 0);

        // A one-block prompt reserves exactly one block.
        assert_eq!(seq.reserve(&mut alloc).unwrap(), 1);
        assert_eq!(seq.block_table().len(), 1);
        assert_eq!(alloc.allocated_blocks(), 1);

        // Generate a block's worth of tokens, then grow by one block.
        for _ in 0..BLOCK_SIZE {
            seq.push_output(TokenId(99));
        }
        assert_eq!(seq.blocks_needed(seq.len()), 1);
        assert_eq!(seq.reserve(&mut alloc).unwrap(), 1);
        assert_eq!(seq.block_table().len(), 2);
        assert_eq!(alloc.allocated_blocks(), 2);

        // Freeing returns every block to the pool.
        seq.free_blocks(&mut alloc);
        assert!(seq.block_table().is_empty());
        assert_eq!(alloc.free_blocks(), total);
        assert_eq!(alloc.allocated_blocks(), 0);
    }

    #[test]
    fn reserve_is_all_or_nothing_on_exhaustion() {
        let mut alloc = BlockAllocator::new(1);
        let mut seq = Sequence::new(
            SeqId(1),
            tokens(BLOCK_SIZE * 3),
            SamplingParams::default(),
            0,
        );

        assert_eq!(seq.blocks_needed(seq.len()), 3);
        assert_eq!(seq.reserve(&mut alloc), Err(AllocError::Exhausted));
        assert!(seq.block_table().is_empty());
        assert_eq!(
            alloc.free_blocks(),
            1,
            "no blocks consumed on a failed reserve"
        );
    }

    #[test]
    fn first_token_time_recorded_once() {
        let mut seq = Sequence::new(SeqId(1), tokens(2), SamplingParams::default(), 0);
        assert!(seq.first_token_at().is_none());

        seq.push_output(TokenId(5));
        let first = seq.first_token_at().unwrap();
        seq.push_output(TokenId(6));

        assert_eq!(seq.first_token_at(), Some(first));
        assert_eq!(seq.output_tokens().len(), 2);
        assert_eq!(seq.len(), 4);
    }
}
