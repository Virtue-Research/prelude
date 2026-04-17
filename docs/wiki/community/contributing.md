# Contributing

Prelude is an early-stage open source project and contributions are very welcome. This page covers how to get started.

## Getting Help / Reporting Issues

- **GitHub Issues** — open an issue at `Virtue-Research/prelude` with steps to reproduce, expected behavior, and relevant logs
- **Discord** — join the community chat at [discord.gg/zbKe5ue8xc](https://discord.gg/zbKe5ue8xc)

## Ways to Contribute

- Bug reports and bug fixes
- New model architecture support
- Performance improvements and benchmarks
- Documentation improvements
- Test coverage


## Commit Messages

Follow the Conventional Commits format:

```
feat(component): short description
fix(component): short description
refactor(component): short description
docs(component): short description
```

Examples:
```
feat(scheduler): add adaptive EWMA batch size tuning
fix(flashinfer): correct paged block size for FA3 backend
docs(wiki): add configuration reference
```

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes with tests where applicable
3. Ensure `cargo test` passes
4. Update relevant documentation
5. Open a PR with a clear description of what changes and why

## Code Style

- Rust code: `cargo fmt` before committing
- Keep dependencies minimal — Prelude aims for small binary size and few runtime deps
- Prefer editing existing files over adding new ones
- No `unwrap()` on user-facing paths — return proper errors

## Adding a Model

See the [Adding a Model](../developer_guide/adding-a-model.md) guide in the developer docs.
