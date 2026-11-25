# .github/workflows

- `ci.yml`: continuous integration (Python validator and hygiene check on Linux,
  Windows x64 and ARM64 builds with the `unit` and `ort` tests, the `npu` tests on
  ARM64, which must all skip there, x64 coverage).
  Described in [docs/setup.md](../../docs/setup.md#continuous-integration).
