//! Block allocator and physical-block identifiers for the paged KV-cache.
//!
//! The KV pool is a flat arena of fixed-size blocks. A [`BlockAllocator`] hands
//! out [`PhysicalBlockId`]s from a free list and reference-counts them, so that
//! prefix sharing and copy-on-write (Layer 4) can let several sequences point at
//! the same physical block. This module is the *only* place block refcounts are
//! mutated (see `CLAUDE.md` §2.4 and `docs/SPEC.md §5.1`).

/// Number of token slots of KV stored in one physical block.
///
/// Logical token position `p` lives in block `p / BLOCK_SIZE` at offset
/// `p % BLOCK_SIZE`; the last block of a sequence may be partially filled.
pub const BLOCK_SIZE: usize = 16;

/// Identifier of a physical block in the KV-pool arena (`0..total_blocks`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalBlockId(pub u32);

/// Error returned when the KV pool cannot satisfy an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AllocError {
    /// No free blocks remain; the scheduler must apply backpressure rather than
    /// let the engine OOM.
    #[error("KV pool exhausted: no free blocks remain")]
    Exhausted,
}

/// Number of blocks required to hold `len` tokens (rounding up).
#[must_use]
pub fn blocks_for_len(len: usize) -> usize {
    len.div_ceil(BLOCK_SIZE)
}

/// Owns the free list and per-block reference counts for the KV pool.
///
/// Invariant: a block is in the free list **iff** its refcount is zero, and the
/// free list never contains duplicates. Therefore
/// `free_blocks() + allocated_blocks() == total_blocks()` always holds.
#[derive(Debug)]
pub struct BlockAllocator {
    free_list: Vec<PhysicalBlockId>,
    refcounts: Vec<u16>,
    total_blocks: usize,
}

impl BlockAllocator {
    /// Create an allocator managing `total_blocks` blocks, all initially free.
    #[must_use]
    pub fn new(total_blocks: usize) -> Self {
        // Push ids in reverse so the stack hands out low ids first — purely for
        // deterministic, readable test traces; correctness does not depend on it.
        let free_list = (0..total_blocks as u32)
            .rev()
            .map(PhysicalBlockId)
            .collect();
        Self {
            free_list,
            refcounts: vec![0; total_blocks],
            total_blocks,
        }
    }

    /// Total number of blocks under management.
    #[must_use]
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Number of blocks currently available to allocate.
    #[must_use]
    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Number of blocks currently held (refcount > 0).
    #[must_use]
    pub fn allocated_blocks(&self) -> usize {
        self.total_blocks - self.free_list.len()
    }

    /// Current reference count of `block` (0 means free).
    #[must_use]
    pub fn refcount(&self, block: PhysicalBlockId) -> u16 {
        self.refcounts[block.0 as usize]
    }

    /// Allocate one free block, returning its id with refcount 1.
    ///
    /// Returns [`AllocError::Exhausted`] when the pool is empty.
    pub fn allocate_block(&mut self) -> Result<PhysicalBlockId, AllocError> {
        let block = self.free_list.pop().ok_or(AllocError::Exhausted)?;
        debug_assert_eq!(
            self.refcounts[block.0 as usize], 0,
            "free block {block:?} had a nonzero refcount"
        );
        self.refcounts[block.0 as usize] = 1;
        Ok(block)
    }

    /// Increment the refcount of an already-held block (prefix sharing / CoW).
    pub fn incref(&mut self, block: PhysicalBlockId) {
        let rc = &mut self.refcounts[block.0 as usize];
        debug_assert!(*rc > 0, "incref on free block {block:?}");
        *rc = rc.checked_add(1).expect("block refcount overflow");
    }

    /// Decrement `block`'s refcount, returning it to the free list at zero.
    ///
    /// Idempotent on an already-free block (a no-op in release builds; a
    /// `debug_assert` failure in debug builds, since that signals a double-free).
    pub fn free_block(&mut self, block: PhysicalBlockId) {
        let idx = block.0 as usize;
        debug_assert!(self.refcounts[idx] > 0, "double free of block {block:?}");
        if self.refcounts[idx] == 0 {
            return;
        }
        self.refcounts[idx] -= 1;
        if self.refcounts[idx] == 0 {
            self.free_list.push(block);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_for_len_rounds_up() {
        assert_eq!(blocks_for_len(0), 0);
        assert_eq!(blocks_for_len(1), 1);
        assert_eq!(blocks_for_len(BLOCK_SIZE), 1);
        assert_eq!(blocks_for_len(BLOCK_SIZE + 1), 2);
    }

    #[test]
    fn allocate_then_free_returns_block_to_pool() {
        let mut alloc = BlockAllocator::new(2);
        assert_eq!(alloc.free_blocks(), 2);

        let a = alloc.allocate_block().unwrap();
        let b = alloc.allocate_block().unwrap();
        assert_ne!(a, b);
        assert_eq!(alloc.free_blocks(), 0);
        assert_eq!(alloc.allocate_block(), Err(AllocError::Exhausted));

        alloc.free_block(a);
        assert_eq!(alloc.free_blocks(), 1);
        assert_eq!(alloc.refcount(a), 0);
    }

    #[test]
    fn refcount_holds_block_until_zero() {
        let mut alloc = BlockAllocator::new(1);
        let a = alloc.allocate_block().unwrap();

        alloc.incref(a);
        assert_eq!(alloc.refcount(a), 2);

        alloc.free_block(a);
        assert_eq!(alloc.refcount(a), 1);
        assert_eq!(
            alloc.free_blocks(),
            0,
            "block still held by remaining reference"
        );

        alloc.free_block(a);
        assert_eq!(alloc.refcount(a), 0);
        assert_eq!(
            alloc.free_blocks(),
            1,
            "block returned to the pool at refcount 0"
        );
    }

    #[test]
    fn conservation_holds_after_each_step() {
        let total = 4;
        let mut alloc = BlockAllocator::new(total);
        let mut held = Vec::new();
        for _ in 0..total {
            held.push(alloc.allocate_block().unwrap());
            assert_eq!(alloc.free_blocks() + alloc.allocated_blocks(), total);
        }
        for b in held {
            alloc.free_block(b);
            assert_eq!(alloc.free_blocks() + alloc.allocated_blocks(), total);
        }
        assert_eq!(alloc.free_blocks(), total);
    }
}
