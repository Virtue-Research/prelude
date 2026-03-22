# Models Layout

- `architectures/<model>/mod.rs`: architecture-specific model implementation.
- `architectures/qwen3/mod.rs`: current working implementation used by `CandleEngine`.
- `architectures/llama/mod.rs`, `architectures/phi/mod.rs`, `architectures/gemma/mod.rs`: placeholders for next implementations.

When adding a new model:
1. Implement the architecture in its own folder.
2. Export it from `architectures/mod.rs`.
3. Wire loading/routing in the engine layer.
