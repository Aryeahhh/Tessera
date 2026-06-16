//! Property tests for the block allocator.
//!
//! Under arbitrary sequences of allocate / increment-reference / free operations
//! the allocator must always uphold:
//!
//! * **Conservation** — `free_blocks + allocated_blocks == total_blocks`.
//! * **No double allocation** — a held block is never handed out again.
//! * **Refcount correctness** — the allocator's refcount matches a reference
//!   model after every operation.
//! * **No leak** — releasing every reference returns the pool to full.

use std::collections::HashMap;

use proptest::prelude::*;
use tessera_core::block::{AllocError, BlockAllocator, PhysicalBlockId};

#[derive(Debug, Clone)]
enum Op {
    /// Allocate a fresh block.
    Alloc,
    /// Free the held block at `index % held.len()`.
    Free(usize),
    /// Increment the refcount of the held block at `index % held.len()`.
    Incref(usize),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        Just(Op::Alloc),
        any::<usize>().prop_map(Op::Free),
        any::<usize>().prop_map(Op::Incref),
    ]
}

proptest! {
    #[test]
    fn allocator_upholds_invariants(
        total in 1usize..=32,
        ops in proptest::collection::vec(op_strategy(), 0..200),
    ) {
        let mut alloc = BlockAllocator::new(total);

        // Reference model: distinct held blocks and their expected refcounts.
        let mut held: Vec<PhysicalBlockId> = Vec::new();
        let mut refs: HashMap<PhysicalBlockId, u32> = HashMap::new();

        for op in ops {
            match op {
                Op::Alloc => {
                    if alloc.free_blocks() == 0 {
                        prop_assert_eq!(alloc.allocate_block(), Err(AllocError::Exhausted));
                    } else {
                        let block = alloc.allocate_block().expect("free block available");
                        prop_assert!(!held.contains(&block), "allocated a block already held");
                        held.push(block);
                        refs.insert(block, 1);
                    }
                }
                Op::Free(i) => {
                    if !held.is_empty() {
                        let idx = i % held.len();
                        let block = held[idx];
                        alloc.free_block(block);
                        let rc = refs.get_mut(&block).expect("held block has a refcount");
                        *rc -= 1;
                        if *rc == 0 {
                            refs.remove(&block);
                            held.swap_remove(idx);
                        }
                    }
                }
                Op::Incref(i) => {
                    if !held.is_empty() {
                        let idx = i % held.len();
                        let block = held[idx];
                        alloc.incref(block);
                        *refs.get_mut(&block).expect("held block has a refcount") += 1;
                    }
                }
            }

            // Conservation and refcount agreement after every operation.
            prop_assert_eq!(alloc.free_blocks() + held.len(), total);
            prop_assert_eq!(alloc.allocated_blocks(), held.len());
            for (&block, &rc) in &refs {
                prop_assert_eq!(alloc.refcount(block), u16::try_from(rc).unwrap());
            }
        }

        // No leak: drop every outstanding reference and expect a full pool.
        let outstanding: Vec<(PhysicalBlockId, u32)> =
            refs.iter().map(|(&block, &rc)| (block, rc)).collect();
        for (block, rc) in outstanding {
            for _ in 0..rc {
                alloc.free_block(block);
            }
        }
        prop_assert_eq!(alloc.free_blocks(), total);
        prop_assert_eq!(alloc.allocated_blocks(), 0);
    }
}
