# benchmarks/results

Benchmark result files written by `edgebench`, one JSON file per run (plus an
optional per-iteration latency CSV). The format is specified in
[docs/results-schema.md](../../docs/results-schema.md).

| Directory | Committed | Contents |
|---|---|---|
| `published/` | yes | Curated runs that are deliberately published (e.g. shown on the dashboard) |
| `local/` | no (gitignored) | Scratch and experimental runs |

`local/` is not part of a fresh clone; create it when you first need it.
To publish a run, copy it (and its latency CSV, if any) from `local/` to
`published/` and validate it from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\validate_results.py benchmarks\results\published
```
