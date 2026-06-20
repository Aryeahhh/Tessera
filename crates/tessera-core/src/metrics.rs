//! Engine-internal counters.
//!
//! Transport-agnostic by design: the engine only increments these, and the API
//! layer (Layer 6) maps them onto a metrics exposition format. Keeping the type
//! here means the core has no dependency on any metrics/HTTP crate.

/// Cumulative engine counters, snapshotted by value.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EngineMetrics {
    /// Engine steps executed.
    pub steps: u64,
    /// Sequences admitted from the waiting queue (prefill starts).
    pub sequences_admitted: u64,
    /// Sequences that reached completion.
    pub sequences_finished: u64,
    /// Output tokens generated across all sequences.
    pub tokens_generated: u64,
    /// Preemptions performed (running sequences evicted under memory pressure).
    pub preemptions: u64,
    /// Preemptions recovered by recompute (drop blocks, rebuild on resume).
    pub preemptions_recompute: u64,
    /// Preemptions recovered by swap (evict blocks to host, restore on resume).
    pub preemptions_swap: u64,
    /// Prompt blocks shared with an existing sequence via prefix caching.
    pub blocks_shared: u64,
}
