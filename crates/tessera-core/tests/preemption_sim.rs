//! Simulation harness for Layer 3: admission safety and preemption.
//!
//! Asserts the exit criteria: under sustained overload the engine never exceeds
//! the pool and every request still completes; a preempted sequence resumes to
//! byte-identical output; the priority policy lets a high-priority arrival
//! displace a low-priority running sequence; and the recompute-vs-swap recovery
//! choice is driven by sequence length.

use tessera_core::{
    BlockAllocator, Engine, EngineConfig, Fcfs, Priority, SamplingParams, SchedulerPolicy, SeqId,
    Sequence, TokenId,
};
use tessera_runtime_mock::MockRuntime;

const EOS: TokenId = TokenId(0);

fn prompt(len: usize) -> Vec<TokenId> {
    (1..=len as u32).map(TokenId).collect()
}

fn generated(gen_len: usize) -> Vec<TokenId> {
    (0..gen_len).map(|i| TokenId(100 + i as u32)).collect()
}

/// Script `id` to emit `gen_len` tokens then EOS, returning its sequence.
fn scripted(
    rt: &mut MockRuntime,
    id: u64,
    prompt_len: usize,
    gen_len: usize,
    priority: u8,
) -> Sequence {
    let mut script = generated(gen_len);
    script.push(EOS);
    rt.script(SeqId(id), script);
    Sequence::new(
        SeqId(id),
        prompt(prompt_len),
        SamplingParams::default(),
        priority,
    )
}

/// Build and run a workload to completion under the given policy, pool, and
/// swap threshold. Returns the finished engine and the per-step traces.
fn run<P: SchedulerPolicy>(
    policy: P,
    pool: usize,
    swap_threshold: usize,
    // (prompt_len, gen_len, priority) per request; id = index + 1.
    specs: &[(usize, usize, u8)],
) -> Engine<MockRuntime, P> {
    let mut rt = MockRuntime::new();
    let seqs: Vec<Sequence> = specs
        .iter()
        .enumerate()
        .map(|(i, &(p, g, pr))| scripted(&mut rt, (i + 1) as u64, p, g, pr))
        .collect();

    let mut engine = Engine::new(
        BlockAllocator::new(pool),
        rt,
        policy,
        EngineConfig {
            eos_token: Some(EOS),
            swap_threshold_tokens: swap_threshold,
        },
    );
    for seq in seqs {
        engine.submit(seq);
    }
    engine.run_to_idle().unwrap();
    engine
}

/// Map of sequence id -> output tokens, for cross-run comparison.
fn outputs<P: SchedulerPolicy>(engine: &Engine<MockRuntime, P>) -> Vec<(u64, Vec<TokenId>)> {
    let mut out: Vec<(u64, Vec<TokenId>)> = engine
        .finished()
        .iter()
        .map(|s| (s.id().0, s.output_tokens().to_vec()))
        .collect();
    out.sort_by_key(|(id, _)| *id);
    out
}

#[test]
fn sustained_overload_never_ooms_and_completes() {
    // Each request can grow to two blocks; the pool holds only two, so at most
    // one full-size sequence fits and the rest must be preempted and resumed.
    let pool = 2;
    let specs = [(1, 20, 0), (1, 25, 0), (1, 30, 0), (1, 18, 0)];

    let engine = run(Fcfs, pool, usize::MAX, &specs);

    assert_eq!(engine.finished().len(), specs.len());
    assert_eq!(
        engine.free_blocks(),
        pool,
        "pool fully reclaimed at the end"
    );
    assert!(
        engine.metrics().preemptions > 0,
        "a tight pool must force preemptions"
    );

    // Every request produced exactly its scripted output despite the churn.
    for (id, out) in outputs(&engine) {
        let (_, gen_len, _) = specs[(id - 1) as usize];
        assert_eq!(out, generated(gen_len), "sequence {id} output mismatch");
    }
}

#[test]
fn preempted_output_matches_unpreempted() {
    let specs = [(2, 22, 0), (3, 17, 0), (1, 31, 0), (4, 12, 0)];

    // A roomy pool never preempts; a tight pool preempts heavily.
    let roomy = run(Fcfs, 64, usize::MAX, &specs);
    let tight = run(Fcfs, 3, usize::MAX, &specs);

    assert_eq!(roomy.metrics().preemptions, 0);
    assert!(tight.metrics().preemptions > 0);

    // Output is identical token-for-token regardless of preemption.
    assert_eq!(outputs(&roomy), outputs(&tight));
}

#[test]
fn priority_preempts_low_priority_running_sequence() {
    // A low-priority long request starts first and fills the two-block pool; a
    // high-priority arrival then preempts it, finishes first, and the
    // low-priority request resumes to completion afterward.
    let mut rt = MockRuntime::new();
    let low = scripted(&mut rt, 1, 16, 8, 0); // priority 0, long-ish
    let high = scripted(&mut rt, 2, 16, 3, 9); // priority 9, short

    let mut engine = Engine::new(
        BlockAllocator::new(2),
        rt,
        Priority,
        EngineConfig {
            eos_token: Some(EOS),
            swap_threshold_tokens: usize::MAX,
        },
    );

    engine.submit(low);
    engine.step().unwrap(); // low admitted, holds the pool
    engine.submit(high);
    engine.run_to_idle().unwrap();

    assert!(
        engine.metrics().preemptions > 0,
        "high priority should preempt low"
    );
    let finish_order: Vec<u64> = engine.finished().iter().map(|s| s.id().0).collect();
    assert_eq!(
        finish_order.first().copied(),
        Some(2),
        "high-priority request finishes first"
    );
    assert_eq!(engine.finished().len(), 2);
}

#[test]
fn recovery_mode_follows_swap_threshold() {
    let specs = [(1, 20, 0), (1, 25, 0), (1, 30, 0)];

    // Threshold above every sequence length: all recompute, never swap.
    let recompute = run(Fcfs, 2, usize::MAX, &specs);
    let m = recompute.metrics();
    assert!(m.preemptions > 0);
    assert_eq!(m.preemptions_swap, 0);
    assert_eq!(m.preemptions_recompute, m.preemptions);

    // Threshold of zero: every victim is long enough to swap.
    let swap = run(Fcfs, 2, 0, &specs);
    let m = swap.metrics();
    assert!(m.preemptions > 0);
    assert_eq!(m.preemptions_recompute, 0);
    assert_eq!(m.preemptions_swap, m.preemptions);
}
