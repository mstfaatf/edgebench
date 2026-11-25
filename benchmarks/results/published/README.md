# benchmarks/results/published

Published benchmark runs. Only add runs here deliberately: real measurements
that pass `scripts/validate_results.py`, together with any latency CSV they
reference. Never add illustrative examples or partial/experimental runs; those
belong in `../local/`.

`accuracy/` holds the accuracy results of every model variant on ImageNetV2
(`scripts/eval_accuracy.py`; schema `schemas/accuracy-result.v1.0.0.schema.json`,
checked by `scripts/validate_accuracy.py`; see [docs/accuracy.md](../../../docs/accuracy.md)).
`validate_results.py` reports them as skipped.
