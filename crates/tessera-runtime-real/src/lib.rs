//! Real model runtime backed by a tensor library (candle) — arrives in Layer 5.
//!
//! This crate is a placeholder skeleton in Layer 0; it exists so the workspace
//! topology and dependency direction are fixed from day one. The concrete
//! [`ModelRuntime`](tessera_core::ModelRuntime) implementation, tokenizer
//! integration, and paged-attention reads land later (see `CLAUDE.md` §3,
//! Layer 5). This is the one crate permitted `unsafe` (behind the FFI
//! boundary, every block justified with `// SAFETY:`), so it intentionally
//! does **not** forbid it.
