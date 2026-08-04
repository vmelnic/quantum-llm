# Contributing

Contributions are welcome, especially those that improve correctness,
observability, long-context memory use, and reproducible performance.

## Development workflow

1. Open an issue for architectural or ABI changes before implementation.
2. Keep Expert Pack and worker protocol changes versioned and fail-closed.
3. Add deterministic tests for every correctness fix.
4. Run the portable suite locally:

   ```bash
   cmake --preset portable-release
   cmake --build --preset portable-release
   ctest --preset portable-release
   python -m unittest -v tests.compiler.test_expert_pack tests.server.test_expert_server
   ```

5. CUDA changes must additionally pass `Invoke-BuildExpertRuntime.ps1`, the
   Qwen CUDA smoke, and the relevant P6 gate on an SM86 host.

Do not commit model weights, generated packs, logs, benchmark artifacts,
credentials, or machine-specific paths. Performance claims must state the
hardware, cache state, context, concurrency, exact commit, and metric semantics.

## Compatibility rules

- Unknown format/ABI/protocol versions are rejected, not guessed.
- Top-k experts are never dropped to keep a request alive.
- A faster kernel needs a numerical oracle and declared tolerance.
- API fields that are not implemented must return a precise error; silently
  ignoring capability parameters is not acceptable.

By contributing, you agree that your contribution is licensed under Apache-2.0.
