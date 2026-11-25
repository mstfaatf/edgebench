"""Print how a model file was made: its recipe and lineage.

Reads the recipe sidecar next to the file (``<name>.recipe.json``) if it exists
and matches the file's SHA256; otherwise builds the recipe from the file's own
``edgebench.*`` metadata (needs onnx). The lineage beyond the direct parent is
followed through ``models/manifest.json`` when the file is listed there.

Usage:
    python scripts/model_info.py <file.onnx> [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model_registry import MODELS_DIR, MetadataError, lineage, read_manifest  # noqa: E402


def relative_to_models(path: Path) -> str:
    try:
        return path.resolve().relative_to(MODELS_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def load_recipe(path: Path) -> tuple[dict, str]:
    """The file's recipe and where it came from ("sidecar" or "metadata")."""
    from common import model_recipe as mr

    data = path.read_bytes()
    sidecar = mr.recipe_path(path)
    if sidecar.is_file():
        recipe = json.loads(sidecar.read_text(encoding="utf-8"))
        if recipe.get("identity", {}).get("sha256") == hashlib.sha256(data).hexdigest():
            return recipe, "sidecar"
        print(f"note: {sidecar.name} is for another version of the file; using its metadata",
              file=sys.stderr)
    return mr.build_recipe(mr.model_file_from_bytes(relative_to_models(path), data)), "metadata"


def shape(t: dict) -> str:
    return f"{t['name']} {t['dtype']} [{', '.join(str(d) for d in t['shape'])}]"


def describe(recipe: dict, source: str) -> list[str]:
    i, w, e, p, q = (recipe[k] for k in ("identity", "weights", "export", "preprocessing",
                                         "quantization"))
    lines = [
        f"{i['file']}  ({source})",
        f"  model          {i['model']}, variant {i['variant']}, {i['precision']}, "
        f"batch {i['batch_size']}, normalization {i['normalization']}",
        f"  file           {i['size_bytes']:,} bytes, sha256 {i['sha256']}",
        "weights",
        f"  {w['library']:<14} {w['id']}",
        f"  url            {w['url']}" if "url" in w else f"  source file    {w['path']}",
        f"  sha256         {w['sha256']}",
    ]
    if "notebook" in w:
        lines += [f"  notebook       {w['notebook']} at {w['notebook_commit']}",
                  f"  code commit    {w['code_commit']}",
                  f"  training rec.  sha256 {w['training_record_sha256']}"]
    if "hf_revision" in w:
        lines.append(f"  hugging face   {w['hf_repo']}@{w['hf_revision']} ({w['file']})")
    versions = ", ".join(f"{k} {v}" for k, v in e["tool_versions"].items())
    lines += [
        "export",
        f"  exporter       {e['exporter']}, opset {e['opset']}, IR {e['ir_version']}",
        f"  inputs         {'; '.join(shape(t) for t in e['inputs'])}",
        f"  outputs        {'; '.join(shape(t) for t in e['outputs'])}",
        f"  tools          {versions}",
    ]
    if e["local_tool_versions"]:
        lines.append("  local tools    " + ", ".join(f"{k} {v}" for k, v in
                                                    e["local_tool_versions"].items()))
    lines.append("preprocessing")
    if p["kind"] == "time_series":
        lines += [f"  windows        {p['dataset']}, length {p['window_length']}, stride "
                  f"{p['window_stride']}, {p['layout']}, {p['dtype']}",
                  f"                 split manifest {p['split_manifest']} "
                  f"(sha256 {p['split_manifest_sha256'][:12]})",
                  f"  fold           {p['fold_not_applicable']}"]
    else:
        lines.append(f"  image          resize shorter side {p['resize_shorter_side']} "
                     f"({p['interpolation']}), center crop {p['center_crop']}, scale "
                     f"{p['scale']}, {p['layout']} {p['dtype']}")
    lines += [
        f"  normalization  mean {p['mean']}, std {p['std']}; applied by "
        f"{p['normalization_applied_by']}",
        f"  input          {p['input_range']}",
    ]
    g = recipe["graph_cleanup"]
    if g:
        lines += ["graph clean-up",
                  f"  {g['nodes_before']} -> {g['nodes_after']} nodes: {g['bias_adds_folded']} bias "
                  f"Adds folded, {g['squeeze_unsqueeze_pairs_removed']} Squeeze/Unsqueeze pairs "
                  f"removed, {g['reshape_chains_collapsed']} reshape chains collapsed"]
    if p["fold"]:
        lines.append(f"  fold           padding {p['fold']['padding']}, "
                     f"first conv {p['fold']['first_conv']}")
    if q:
        c = q["calibration"]
        lines += [
            "quantization",
            f"  recipe         {q['method']} {q['format']}; weights {q['weight_type']} "
            f"{'symmetric' if q['weight_symmetric'] else 'asymmetric'}"
            f"{', per channel' if q['per_channel'] else ', per tensor'}; activations "
            f"{q['activation_type']} {'symmetric' if q['activation_symmetric'] else 'asymmetric'}; "
            f"bias {q['bias_type']}",
            f"  calibration    {c['method']} on {c['sample_count']} samples, {c['dataset']}",
            f"                 manifest sha256 {c['manifest_sha256']}; batch {c['batch']}",
            f"  pruning        " + (f"pruned: {q['pruned_chains']} dead Q/DQ chains, "
                                    f"{q['pruned_nodes']} nodes removed" if q["pruned"]
                                    else "not pruned (reference file)"),
            f"  quantizer      {q['quantizer']}",
        ]
    return lines


def lineage_lines(recipe: dict) -> list[str]:
    parent = recipe["lineage"]["parent"]
    if parent is None:
        if recipe["weights"]["library"] == "colab_bundle":
            return ["lineage", "  imported from the source file above (root)"]
        return ["lineage", "  exported from the weights above (root)"]
    files, _ = read_manifest()
    entry = next((f for f in files if f.sha256 == recipe["identity"]["sha256"]), None)
    if entry is None:
        return ["lineage", f"  made from {parent['file']} (sha256 {parent['sha256'][:12]}); "
                           "this file is not in models/manifest.json"]
    try:
        chain = lineage(files, entry)
    except ValueError as exc:
        return ["lineage", f"  made from {parent['file']}; manifest lineage broken: {exc}"]
    return ["lineage"] + [f"  {'<- ' if n else '   '}{f.file} ({f.variant}, sha256 {f.sha256[:12]})"
                          for n, f in enumerate(chain)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path, help="ONNX model file")
    parser.add_argument("--json", action="store_true", help="print the recipe as JSON")
    args = parser.parse_args(argv)
    if not args.model.is_file():
        print(f"error: {args.model} not found", file=sys.stderr)
        return 2
    try:
        recipe, source = load_recipe(args.model)
    except MetadataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(recipe, indent=2))
        return 0
    print("\n".join(describe(recipe, source) + lineage_lines(recipe)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
