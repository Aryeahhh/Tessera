# Tessera

Tessera is a from-scratch LLM **serving engine** written in Rust. It maximizes
accelerator throughput via **continuous (iteration-level) batching** and manages
attention memory with a **paged KV-cache** — fixed-size blocks, per-sequence
block tables, and copy-on-write prefix sharing.

The engineering focus is the **scheduler and memory allocator**. The model
runtime sits behind a trait boundary (`ModelRuntime`) and is initially driven by
a deterministic mock, so the entire scheduling and allocation stack can be built
and tested with zero ML dependency.

## Workspace layout

| Crate | Kind | Responsibility |
|-------|------|----------------|
| `tessera-core` | lib | Block allocator, sequences, scheduler, engine step loop, runtime trait |
| `tessera-runtime-mock` | lib | Deterministic, scripted `ModelRuntime` for testing |
| `tessera-runtime-real` | lib | Tensor-backed `ModelRuntime` (later) |
| `tessera-api` | bin | HTTP/SSE serving front end (later) |
| `tessera-bench` | bin | Load generator and baseline comparison (later) |
| `xtask` | bin | Repository automation |

`tessera-core` depends on no runtime crate — runtimes are injected through the
trait it defines, keeping the dependency graph acyclic.

## Development

A pinned stable toolchain is declared in `rust-toolchain.toml`. The same checks
CI runs can be run locally:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all --all-features
cargo build --release
```

## Status

Early development. The workspace scaffold, tooling, CI, and the mock runtime are
in place; the allocator, scheduler, and engine loop are being built up layer by
layer.
