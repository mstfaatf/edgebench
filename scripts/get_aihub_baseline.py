"""Download Qualcomm AI Hub's pre-quantized (w8a8) MobileNetV2 ONNX model and its float twin.

Qualcomm publishes AI Hub models on Hugging Face under the ``qualcomm`` org.
The ``qualcomm/MobileNet-v2`` repository holds the model card and a
``release_assets.json`` that points at the actual assets on Qualcomm's public
S3 bucket. No account or token is needed. This script:

1. fetches ``release_assets.json``, ``README.md`` (model card) and ``LICENSE``
   from the pinned Hugging Face revision and checks that the ONNX asset URLs in
   it are the ones pinned here,
2. downloads the ``w8a8`` and ``float`` ONNX assets (zips, pinned by S3 object
   version and SHA256) into ``models/aihub/downloads/``,
3. extracts them to ``models/aihub/mobilenet_v2_<precision>/``,
4. writes ``models/aihub/mobilenet_v2_<precision>.json`` describing the source,
   license, archive contents, and the model's inputs and outputs.

The w8a8 archive contains an ONNX QDQ model (``mobilenet_v2.onnx``) whose
float weights live in an external data file (``mobilenet_v2.data``) and are
quantized by QuantizeLinear nodes in the graph, plus ``labels.txt`` and
``metadata.json``. It contains no pre-compiled QNN context binary. The float
model is the same network before quantization; it is used to separate the
effect of AI Hub's different trained weights from its quantization.
``models/`` is not committed.

Usage:
    python scripts/get_aihub_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.preprocess import REPO_ROOT  # noqa: E402

HF_REPO = "qualcomm/MobileNet-v2"
HF_REVISION = "31a0cdd09d996f7d71847ce14a83d92ccb747caa"  # 2026-11-18, AI Hub Models v0.67.0
HF_FILES = ["release_assets.json", "README.md", "LICENSE"]
RELEASE = "0.67.0"
_BUCKET = ("https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/"
           "mobilenet_v2/releases/v0.67.0/")
# precision -> (asset URL, S3 object version id, SHA256 of the zip)
ASSETS = {
    "w8a8": (_BUCKET + "mobilenet_v2-onnx-w8a8.zip", "sZ4TesoRKcf.fYEFLxRCWMdgKSihVzMh",
             "4e0ab0c7ec6136a7cccbbe96e6f2213e6cc259e7a08bcc2b699c44e6d7c8341e"),
    "float": (_BUCKET + "mobilenet_v2-onnx-float.zip", "LtCxRMlzwHos5auC84iuUfZr8QBNTYbP",
              "4a479de6925b49e60317ab2e52827e5e8dd94e997d8994afba62855907eb8837"),
}
LICENSE = {
    "model_card": "apache-2.0",
    "original_implementation": "https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/LICENSE",
    "qai_hub_models_source": "BSD-3-Clause",
}

AIHUB_DIR = REPO_ROOT / "models" / "aihub"
DOWNLOAD_DIR = AIHUB_DIR / "downloads"
MODEL_RELPATH = "mobilenet_v2_onnx/mobilenet_v2.onnx"


def model_path(precision: str) -> Path:
    """Path of the extracted AI Hub ONNX model for ``precision`` ("w8a8" or "float")."""
    return AIHUB_DIR / f"mobilenet_v2_{precision}" / MODEL_RELPATH


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)


def fetch_hf_metadata() -> dict:
    meta_dir = DOWNLOAD_DIR / f"hf_{HF_REVISION[:12]}"
    for name in HF_FILES:
        dest = meta_dir / name
        if not dest.is_file():
            download(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{name}", dest)
    assets = json.loads((meta_dir / "release_assets.json").read_text(encoding="utf-8"))
    if assets.get("version") != RELEASE:
        raise RuntimeError(f"release_assets.json is for {assets.get('version')}, expected {RELEASE}")
    tool_versions = {}
    for precision, (url, _, _) in ASSETS.items():
        onnx_asset = assets["precisions"][precision]["universal_assets"]["onnx"]
        if onnx_asset["download_url"] != url:
            raise RuntimeError(f"{precision} asset URL changed: {onnx_asset['download_url']}")
        tool_versions[precision] = onnx_asset["tool_versions"]
    return {"dir": meta_dir, "tool_versions": tool_versions,
            "files": {n: sha256_of(meta_dir / n) for n in HF_FILES}}


def safe_extract(archive: Path, dest: Path, archive_root: str) -> list[dict]:
    """Extract a zip, refusing absolute paths and parent-directory components."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    contents = []
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts = PurePosixPath(info.filename).parts
            if not parts or parts[0] != archive_root or ".." in parts or info.filename.startswith("/"):
                raise RuntimeError(f"unexpected path in archive: {info.filename!r}")
            rel = PurePosixPath(*parts[1:])
            if info.is_dir() or not rel.parts:
                continue
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            contents.append({"path": str(rel), "size_bytes": info.file_size,
                             "sha256": sha256_of(target)})
    return contents


def describe_io(model_path: Path) -> dict:
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)

    def spec(v) -> dict:
        t = v.type.tensor_type
        return {"name": v.name, "dtype": onnx.TensorProto.DataType.Name(t.elem_type),
                "shape": [d.dim_value or d.dim_param for d in t.shape.dim]}

    external = sorted({e.value for i in model.graph.initializer for e in i.external_data
                       if e.key == "location"})
    return {"ir_version": model.ir_version,
            "opset": {o.domain or "ai.onnx": o.version for o in model.opset_import},
            "producer": f"{model.producer_name} {model.producer_version}".strip(),
            "inputs": [spec(v) for v in model.graph.input],
            "outputs": [spec(v) for v in model.graph.output],
            "external_data_files": external}


def fetch_asset(precision: str, meta: dict) -> Path:
    url, version_id, sha256 = ASSETS[precision]
    archive = DOWNLOAD_DIR / PurePosixPath(url).name
    if not archive.is_file():
        download(f"{url}?versionId={version_id}", archive)
    digest = sha256_of(archive)
    if digest != sha256:
        raise RuntimeError(f"SHA256 mismatch for {archive}\n  expected {sha256}\n  got      {digest}\n"
                           "Delete the file and re-run to download it again.")
    print(f"\n[{precision}] archive OK: {archive.name} (sha256 {digest})")

    extract_dir = model_path(precision).parents[1]
    contents = safe_extract(archive, extract_dir, PurePosixPath(url).stem)
    for c in contents:
        print(f"  {c['path']:<40} {c['size_bytes']:>10} bytes")
    path = model_path(precision)
    io = describe_io(path)
    sidecar = {
        "model": path.relative_to(REPO_ROOT).as_posix(),
        "source": {
            "hugging_face_repo": HF_REPO,
            "hugging_face_revision": HF_REVISION,
            "hugging_face_files_sha256": meta["files"],
            "ai_hub_models_release": RELEASE,
            "precision": precision,
            "asset_url": url,
            "asset_s3_version_id": version_id,
            "asset_sha256": sha256,
            "tool_versions": meta["tool_versions"][precision],
        },
        "license": LICENSE,
        "archive_contents": contents,
        "contains_qnn_context_binary": any(c["path"].endswith(".bin") for c in contents),
        "graph_io": io,
        "ai_hub_metadata": json.loads((extract_dir / "metadata.json").read_text(encoding="utf-8")),
    }
    sidecar_path = AIHUB_DIR / f"mobilenet_v2_{precision}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"  inputs:  {io['inputs']}\n  outputs: {io['outputs']}\n"
          f"  external data: {io['external_data_files']}\n  wrote {sidecar_path.name}")
    return path


def main() -> int:
    meta = fetch_hf_metadata()
    print(f"Hugging Face {HF_REPO}@{HF_REVISION[:12]}: release {RELEASE}, "
          f"tool versions {meta['tool_versions']}")
    for precision in ASSETS:
        fetch_asset(precision, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
