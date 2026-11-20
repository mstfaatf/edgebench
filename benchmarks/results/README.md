# benchmarks/results

Benchmark result files written by `edgebench`, one JSON file per run (plus an
optional per-iteration latency CSV). The format is specified in
[docs/results-schema.md](../../docs/results-schema.md).

| Directory | Committed | Contents |
|---|---|---|
| `published/` | yes | Curated runs that are deliberately published (e.g. shown on the dashboard) |
| `local/` | no (gitignored) | Scratch and experimental runs; write here by default |

To publish a run, copy it (and its latency CSV, if any) from `local/` to
`published/` and validate it:

```powershell
.\.venv\Scripts\python.exe scripts\validate_results.py benchmarks\results\published
```
