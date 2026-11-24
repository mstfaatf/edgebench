"""Regenerate or check the recipe sidecars and the model manifest.

Every model file listed in ``models/manifest.json`` gets a recipe sidecar
``<name>.recipe.json`` next to it, built from the file alone (its ``edgebench.*``
metadata, graph inputs/outputs, SHA256 and size; see ``common/model_recipe.py``).
Export, fold and quantize write the sidecars themselves; this command
regenerates them for files that already exist.

``--check`` writes nothing and fails (exit 1) unless:

- the manifest matches its schema and its lineage is sound: every parent is in
  the manifest, and following parents always ends at an FP32 export with
  external normalization (no cycles, no missing parents);
- for every model file on disk: its sidecar exists, matches the recipe schema
  and equals the recipe rebuilt from the file, and the file's manifest entry
  (hash, size, model, variant, precision, normalization, batch size, pruning,
  parent) equals what the file's metadata says.

Files listed in the manifest but not on disk are reported and skipped.

Usage:
    python scripts/model_recipes.py [--check] [--model NAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model_registry import (  # noqa: E402
    MANIFEST_PATH,
    MANIFEST_SCHEMA,
    MODELS,
    MODELS_DIR,
    MetadataError,
    lineage_problems,
    read_manifest,
)


def manifest_schema_problems() -> list[str]:
    import jsonschema

    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    doc = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [f"manifest {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in jsonschema.Draft202012Validator(schema).iter_errors(doc)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="check sidecars, manifest and lineage; write nothing")
    parser.add_argument("--model", nargs="+", choices=sorted(MODELS),
                        help="only these models (default: all)")
    args = parser.parse_args(argv)

    from common import model_recipe as mr

    files, _ = read_manifest()
    if not files:
        print(f"{MANIFEST_PATH} is missing or empty", file=sys.stderr)
        return 1
    problems: list[str] = []
    lineage: list[str] = []
    if args.check:
        problems += manifest_schema_problems()
        lineage = lineage_problems(files)
        problems += lineage
    selected = [f for f in files if args.model is None or f.model in args.model]
    present = missing = 0
    for entry in selected:
        path = MODELS_DIR / entry.file
        if not path.is_file():
            missing += 1
            print(f"{entry.file}: not on disk, skipped")
            continue
        present += 1
        if not args.check:
            try:
                written = mr.record_model_file(path, entry.file)
            except (MetadataError, ValueError) as exc:
                problems.append(str(exc))
                continue
            if written != entry:
                problems.append(f"{entry.file}: manifest entry differs from the file's metadata; "
                                "re-run the script that made it")
            print(f"{mr.recipe_path(path).relative_to(MODELS_DIR).as_posix()}: written")
            continue
        sidecar = mr.recipe_path(path)
        if not sidecar.is_file():
            problems.append(f"{entry.file}: no recipe sidecar ({sidecar.name}); "
                            "run scripts/model_recipes.py")
            continue
        recipe = json.loads(sidecar.read_text(encoding="utf-8"))
        problems += [f"{sidecar.name}: {p}" for p in mr.schema_problems(recipe)]
        problems += mr.consistency_problems(recipe, entry, mr.read_model_file(path, entry.file))

    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    action = "checked" if args.check else "regenerated"
    summary = f"{action} {present} file(s), {missing} not on disk"
    if args.check:
        summary += (f"; lineage of {len(files)} manifest entries "
                    + ("OK" if not lineage else f"has {len(lineage)} problem(s)"))
    print(summary + (f"; {len(problems)} problem(s)" if problems else "; OK"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
