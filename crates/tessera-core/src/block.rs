//! Block allocator and physical-block identifiers for the paged KV-cache.
//!
//! The KV pool is a flat arena of fixed-size blocks. A [`BlockAllocator`] hands
//! out [`PhysicalBlockId`]s from a free list and reference-counts them, so that
//! prefix sharing and copy-on-write can let several sequences point at the same
//! physical block. This module is the *only* place block refcounts and the
//! prefix index are mutated (see `CLAUDE.md` §2.4 and `docs/SPEC.md §5.1`–§5.2).

use std::collections::HashMap;

use crate::ids::TokenId;

/// Number of token slots of KV stored in one physical block.
///
/// Logical token position `p` lives in block `p / BLOCK_SIZE` at offset
/// `p % BLOCK_SIZE`; the last block of a sequence may be partially filled.
pub const BLOCK_SIZE: usize = 16;

/// Identifier of a physical block in the KV-pool arena (`0..total_blocks`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhysicalBlockId(pub u32);

/// Content hash of a full block, chained over the preceding prefix.
///
/// Two sequences share a physical block only when every token up to and
/// including that block matches, so the hash chains the previous block's hash
/// into the current one (the KV of a block depends on the whole prefix).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub u64);

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

/// Chained content hashes for each *complete* block of `tokens`.
///
/// Returns one [`BlockHash`] per full `BLOCK_SIZE`-token block; a trailing
/// partial block is not hashable (its content is not yet fixed) and is omitted.
/// Each hash folds in the previous block's hash, so equal hashes imply equal
/// prefixes. A production cache verifies token equality on a hit to guard
/// against the astronomically unlikely collision; with the deterministic mock
/// workloads here, hash equality stands in for content equality.
#[must_use]
pub fn chained_block_hashes(tokens: &[TokenId]) -> Vec<BlockHash> {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let complete = tokens.len() / BLOCK_SIZE;
    let mut hashes = Vec::with_capacity(complete);
    let mut acc = FNV_OFFSET;
    for block in tokens.chunks_exact(BLOCK_SIZE) {
        for &TokenId(token) in block {
            for byte in token.to_le_bytes() {
                acc ^= u64::from(byte);
                acc = acc.wrapping_mul(FNV_PRIME);
            }
        }
        hashes.push(BlockHash(acc));
    }
    hashes
}

/// Owns the free list, per-block reference counts, and the prefix-share index.
///
/// Invariant: a block is in the free list **iff** its refcount is zero, and the
/// free list never contains duplicates. Therefore
/// `free_blocks() + allocated_blocks() == total_blocks()` always holds. A block
/// is present in the prefix index only while its refcount is positive.
#[derive(Debug)]
pub struct BlockAllocator {
    free_list: Vec<PhysicalBlockId>,
    refcounts: Vec<u16>,
    total_blocks: usize,
    prefix_index: HashMap<BlockHash, PhysicalBlockId>,
    block_hash: Vec<Option<BlockHash>>,
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
            prefix_index: HashMap::new(),
            block_hash: vec![None; total_blocks],
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

    /// Allocate a block for the prefix identified by `hash`, sharing an existing
    /// physical block when the prefix is already cached.
    ///
    /// Returns the block and whether it was **shared** (an existing block whose
    /// refcount was bumped) rather than freshly allocated. A freshly allocated
    /// block is registered so later sequences with the same prefix can share it.
    pub fn allocate_shared(
        &mut self,
        hash: BlockHash,
    ) -> Result<(PhysicalBlockId, bool), AllocError> {
        if let Some(&block) = self.prefix_index.get(&hash) {
            // Cached blocks always have a positive refcount (evicted on free).
            self.incref(block);
            return Ok((block, true));
        }
        let block = self.allocate_block()?;
        self.block_hash[block.0 as usize] = Some(hash);
        self.prefix_index.insert(hash, block);
        Ok((block, false))
    }

    /// Increment the refcount of an already-held block (prefix sharing / CoW).
    pub fn incref(&mut self, block: PhysicalBlockId) {
        let rc = &mut self.refcounts[block.0 as usize];
        debug_assert!(*rc > 0, "incref on free block {block:?}");
        *rc = rc.checked_add(1).expect("block refcount overflow");
    }

    /// Privatize a shared block before a divergent write (copy-on-write).
    ///
    /// If `block` is uniquely owned this is a no-op returning `block` itself.
    /// Otherwise a fresh private block is allocated and this owner's share of the
    /// original is released; the caller repoints its block table at the copy. A
    /// real KV runtime copies the block's tensor contents here — the mock holds
    /// no bytes, so privatization is purely the block-table swap.
    pub fn copy_on_write(&mut self, block: PhysicalBlockId) -> Result<PhysicalBlockId, AllocError> {
        if self.refcount(block) <= 1 {
            return Ok(block);
        }
        let private = self.allocate_block()?;
        self.free_block(block);
        Ok(private)
    }

    /// Decrement `block`'s refcount, returning it to the free list at zero.
    ///
    /// Idempotent on an already-free block (a no-op in release builds; a
    /// `debug_assert` failure in debug builds, since that signals a double-free).
    /// When the block becomes free, any prefix-index entry for it is evicted.
    pub fn free_block(&mut self, block: PhysicalBlockId) {
        let idx = block.0 as usize;
        debug_assert!(self.refcounts[idx] > 0, "double free of block {block:?}");
        if self.refcounts[idx] == 0 {
            return;
        }
        self.refcounts[idx] -= 1;
        if self.refcounts[idx] == 0 {
            if let Some(hash) = self.block_hash[idx].take() {
                if self.prefix_index.get(&hash) == Some(&block) {
                    self.prefix_index.remove(&hash);
                }
            }
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

    #[test]
    fn equal_prefixes_hash_equal_and_diverge_on_difference() {
        let a: Vec<TokenId> = (0..2 * BLOCK_SIZE as u32).map(TokenId).collect();
        let mut b = a.clone();
        b[BLOCK_SIZE + 1] = TokenId(9999); // diverge inside the second block

        let ha = chained_block_hashes(&a);
        let hb = chained_block_hashes(&b);
        assert_eq!(ha.len(), 2);
        assert_eq!(ha[0], hb[0], "identical first block hashes");
        assert_ne!(ha[1], hb[1], "second block diverges");
    }

    #[test]
    fn allocate_shared_reuses_cached_prefix_block() {
        let mut alloc = BlockAllocator::new(4);
        let hashes = chained_block_hashes(&(0..BLOCK_SIZE as u32).map(TokenId).collect::<Vec<_>>());
        let hash = hashes[0];

        let (first, shared_a) = alloc.allocate_shared(hash).unwrap();
        let (second, shared_b) = alloc.allocate_shared(hash).unwrap();

        assert!(!shared_a, "first request allocates the block");
        assert!(shared_b, "second request shares it");
        assert_eq!(first, second);
        assert_eq!(alloc.refcount(first), 2);
        assert_eq!(alloc.allocated_blocks(), 1, "only one physical block used");
    }

    #[test]
    fn freeing_evicts_prefix_entry() {
        let mut alloc = BlockAllocator::new(2);
        let hash =
            chained_block_hashes(&(0..BLOCK_SIZE as u32).map(TokenId).collect::<Vec<_>>())[0];

        let (block, _) = alloc.allocate_shared(hash).unwrap();
        alloc.free_block(block);

        // The hash no longer maps to a (now-free) block: a new request allocates.
        let (again, shared) = alloc.allocate_shared(hash).unwrap();
        assert!(!shared, "a freed prefix block is no longer shareable");
        assert_eq!(again, block, "the freed block id is reused");
    }

    #[test]
    fn copy_on_write_privatizes_only_when_shared() {
        let mut alloc = BlockAllocator::new(4);
        let block = alloc.allocate_block().unwrap();

        // Uniquely owned: CoW is a no-op.
        assert_eq!(alloc.copy_on_write(block).unwrap(), block);
        assert_eq!(alloc.refcount(block), 1);

        // Shared: CoW hands back a private copy and releases this share.
        alloc.incref(block);
        assert_eq!(alloc.refcount(block), 2);
        let private = alloc.copy_on_write(block).unwrap();
        assert_ne!(private, block);
        assert_eq!(alloc.refcount(block), 1, "other owner keeps the original");
        assert_eq!(alloc.refcount(private), 1, "writer owns the copy");
    }
}
