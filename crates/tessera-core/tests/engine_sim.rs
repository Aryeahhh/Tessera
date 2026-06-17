//! Deterministic simulation harness for the continuous-batching engine.
//!
//! Drives the engine with scripted requests through the real [`MockRuntime`] and
//! asserts the Layer 2 properties: varied-length requests complete with correct
//! output, the decode batch composition changes step to step, and FCFS staggers
//! admission when the block budget is tight — all without ever exceeding the
//! pool or deadlocking on cooperative input.

use tessera_core::{
    BlockAllocator, Engine, EngineConfig, Fcfs, SamplingParams, SeqId, Sequence, TokenId,
};
use tessera_runtime_mock::MockRuntime;

const EOS: TokenId = TokenId(0);
const POOL_BLOCKS: usize = 64;

/// Prompt tokens `1..=len` (kept distinct from generated and EOS tokens).
fn prompt(len: usize) -> Vec<TokenId> {
    (1..=len as u32).map(TokenId).collect()
}

/// The tokens sequence `gen_len` will emit: `100, 101, ...` then EOS.
fn generated(gen_len: usize) -> Vec<TokenId> {
    (0..gen_len).map(|i| TokenId(100 + i as u32)).collect()
}

/// Script `id` to emit `gen_len` tokens then EOS, and return its sequence.
fn scripted(rt: &mut MockRuntime, id: u64, prompt_len: usize, gen_len: usize) -> Sequence {
    let mut script = generated(gen_len);
    script.push(EOS);
    rt.script(SeqId(id), script);
    Sequence::new(SeqId(id), prompt(prompt_len), SamplingParams::default(), 0)
}

fn engine_with(rt: MockRuntime, pool: usize, seqs: Vec<Sequence>) -> Engine<MockRuntime, Fcfs> {
    let mut engine = Engine::new(
        BlockAllocator::new(pool),
        rt,
        Fcfs,
        EngineConfig {
            eos_token: Some(EOS),
        },
    );
    for seq in seqs {
        engine.submit(seq);
    }
    engine
}

#[test]
fn varied_length_requests_all_complete() {
    // (prompt_len, gen_len) per request, id = index + 1.
    let specs = [(5, 3), (10, 1), (1, 8), (20, 4), (16, 16)];

    let mut rt = MockRuntime::new();
    let seqs: Vec<Sequence> = specs
        .iter()
        .enumerate()
        .map(|(i, &(p, g))| scripted(&mut rt, (i + 1) as u64, p, g))
        .collect();

    let mut engine = engine_with(rt, POOL_BLOCKS, seqs);
    let traces = engine.run_to_idle().unwrap();

    assert!(engine.is_idle());
    assert_eq!(engine.finished().len(), specs.len());

    // Every sequence produced exactly its scripted tokens, EOS excluded.
    for fin in engine.finished() {
        let id = fin.id().0 as usize;
        let (_, gen_len) = specs[id - 1];
        assert_eq!(
            fin.output_tokens(),
            generated(gen_len).as_slice(),
            "sequence {id} output mismatch"
        );
    }

    let metrics = engine.metrics();
    assert_eq!(metrics.sequences_admitted as usize, specs.len());
    assert_eq!(metrics.sequences_finished as usize, specs.len());
    assert_eq!(
        metrics.tokens_generated as usize,
        specs.iter().map(|(_, g)| g).sum::<usize>()
    );

    // The pool is whole again once everyone finishes.
    assert_eq!(traces.last().unwrap().free_blocks_after, POOL_BLOCKS);
}

#[test]
fn decode_batch_composition_changes_per_step() {
    let specs = [(4, 2), (4, 6), (4, 1), (4, 9)];

    let mut rt = MockRuntime::new();
    let seqs: Vec<Sequence> = specs
        .iter()
        .enumerate()
        .map(|(i, &(p, g))| scripted(&mut rt, (i + 1) as u64, p, g))
        .collect();

    let mut engine = engine_with(rt, POOL_BLOCKS, seqs);
    let traces = engine.run_to_idle().unwrap();

    // Step 0 is pure prefill: everyone admitted, no decodes yet.
    assert_eq!(traces[0].admitted.len(), specs.len());
    assert!(traces[0].decoded.is_empty());

    // The decode set is not constant — sequences drop out as they finish.
    let changed = traces.windows(2).any(|w| w[0].decoded != w[1].decoded);
    assert!(
        changed,
        "decode batch composition should change across steps"
    );

    // It also shrinks: the largest decode batch precedes the smallest non-empty one.
    let max_decode = traces.iter().map(|t| t.decoded.len()).max().unwrap();
    let min_nonempty = traces
        .iter()
        .map(|t| t.decoded.len())
        .filter(|&n| n > 0)
        .min()
        .unwrap();
    assert!(
        max_decode > min_nonempty,
        "decode batch should shrink over time"
    );

    assert_eq!(engine.finished().len(), specs.len());
}

#[test]
fn fcfs_staggers_admission_under_tight_budget() {
    // Six one-block requests, only four blocks. Short generations keep each
    // request within a single block, so blocks free up as requests finish and
    // the queued ones are admitted later — never exceeding the budget.
    let pool = 4;
    let specs = [(10, 2), (10, 3), (10, 2), (10, 4), (10, 3), (10, 2)];

    let mut rt = MockRuntime::new();
    let seqs: Vec<Sequence> = specs
        .iter()
        .enumerate()
        .map(|(i, &(p, g))| scripted(&mut rt, (i + 1) as u64, p, g))
        .collect();

    let mut engine = engine_with(rt, pool, seqs);
    let traces = engine.run_to_idle().unwrap();

    // Not everyone fit at step 0: some admissions happened later.
    let admitted_after_step0: usize = traces.iter().skip(1).map(|t| t.admitted.len()).sum();
    assert!(
        admitted_after_step0 > 0,
        "tight budget should defer some admissions past step 0"
    );

    // The budget was always honored.
    let max_running = traces.iter().map(|t| t.running_after).max().unwrap();
    assert!(
        max_running <= pool,
        "running set never exceeds the block budget"
    );
    for trace in &traces {
        assert!(trace.free_blocks_after <= pool);
    }

    // Everyone still completes, and the pool is whole again.
    assert_eq!(engine.finished().len(), specs.len());
    assert_eq!(engine.free_blocks(), pool);
}
