//! Simulation harness for Layer 4: copy-on-write prefix sharing.
//!
//! Drives the engine end-to-end and asserts that two requests with the same
//! system prompt share physical blocks (so the pool holds fewer blocks than the
//! naive sum), while still producing each request's own correct output.

use tessera_core::{
    BlockAllocator, Engine, EngineConfig, Fcfs, SamplingParams, SeqId, Sequence, TokenId,
    BLOCK_SIZE,
};
use tessera_runtime_mock::MockRuntime;

const EOS: TokenId = TokenId(0);

fn generated(gen_len: usize) -> Vec<TokenId> {
    (0..gen_len).map(|i| TokenId(100 + i as u32)).collect()
}

#[test]
fn shared_system_prompt_reuses_blocks_across_requests() {
    // A two-block system prompt is identical across both requests.
    let system_prompt: Vec<TokenId> = (1..=2 * BLOCK_SIZE as u32).map(TokenId).collect();

    let mut rt = MockRuntime::new();
    let mut submit = Vec::new();
    for (id, gen_len) in [(1u64, 4usize), (2, 7)] {
        let mut script = generated(gen_len);
        script.push(EOS);
        rt.script(SeqId(id), script);
        submit.push(Sequence::new(
            SeqId(id),
            system_prompt.clone(),
            SamplingParams::default(),
            0,
        ));
    }

    let mut engine = Engine::new(
        BlockAllocator::new(64),
        rt,
        Fcfs,
        EngineConfig {
            eos_token: Some(EOS),
            ..Default::default()
        },
    );
    for seq in submit {
        engine.submit(seq);
    }
    engine.run_to_idle().unwrap();

    // The second request shared both complete system-prompt blocks.
    assert_eq!(
        engine.metrics().blocks_shared,
        2,
        "the shared two-block prompt is reused by the second request"
    );

    // Sharing does not corrupt output: each request emits its own tokens.
    assert_eq!(engine.finished().len(), 2);
    for fin in engine.finished() {
        let gen_len = if fin.id() == SeqId(1) { 4 } else { 7 };
        assert_eq!(fin.output_tokens(), generated(gen_len).as_slice());
    }
}

#[test]
fn distinct_prompts_do_not_share() {
    let mut rt = MockRuntime::new();
    let mut submit = Vec::new();
    for id in [1u64, 2] {
        // Distinct two-block prompts (offset by the id).
        let prompt: Vec<TokenId> = (0..2 * BLOCK_SIZE as u32)
            .map(|t| TokenId(t + id as u32 * 1000))
            .collect();
        let mut script = generated(3);
        script.push(EOS);
        rt.script(SeqId(id), script);
        submit.push(Sequence::new(
            SeqId(id),
            prompt,
            SamplingParams::default(),
            0,
        ));
    }

    let mut engine = Engine::new(
        BlockAllocator::new(64),
        rt,
        Fcfs,
        EngineConfig {
            eos_token: Some(EOS),
            ..Default::default()
        },
    );
    for seq in submit {
        engine.submit(seq);
    }
    engine.run_to_idle().unwrap();

    assert_eq!(
        engine.metrics().blocks_shared,
        0,
        "no prefix overlap, no sharing"
    );
    assert_eq!(engine.finished().len(), 2);
}
