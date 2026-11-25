# Third-party notices

EdgeBench itself is licensed under the [MIT License](LICENSE). It depends on
the third-party components below. **None of them is stored in this
repository**: the native libraries are downloaded by `scripts/fetch_deps.ps1`,
the C++ libraries by CMake at configure time, the Python packages by pip, and
models and datasets by the scripts that use them. Every download is pinned by
version or revision; the native libraries, C++ libraries, dataset and AI Hub
model archives are also checked against a SHA-256. Each component's own license
file is included in what is downloaded; where it is linked below, the upstream
copy is authoritative.

## Native runtime (downloaded by `scripts/fetch_deps.ps1`)

| Component | Version | License | Source |
|---|---|---|---|
| ONNX Runtime (`Microsoft.ML.OnnxRuntime` NuGet package, `win-arm64` and `win-x64` native libraries) | 1.32.0 | MIT | <https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/1.32.0>, <https://github.com/microsoft/onnxruntime> |
| ONNX Runtime QNN execution provider plugin (`onnxruntime_providers_qnn.dll`), ARM64 builds only | 2.8.0 | MIT | <https://github.com/onnxruntime/onnxruntime-qnn/releases/tag/v2.8.0> |
| Qualcomm QNN backend libraries shipped in the same release (`QnnHtp*.dll`, `QnnSystem.dll`, `QnnGpu.dll`, `libQnnHtp*Skel.so`, …), ARM64 builds only | QNN SDK 2.52.0 | Qualcomm "AI Stack License" (*Terms and Conditions of Use*, `Qualcomm_LICENSE.pdf` in the release archive); proprietary | <https://github.com/onnxruntime/onnxruntime-qnn/releases/tag/v2.8.0> |

ONNX Runtime and the QNN plugin each ship a `ThirdPartyNotices.txt` for the
components they contain. The Qualcomm license permits use for developing
applications and for benchmarking, and redistribution only in object code as
part of an application, not on its own. EdgeBench does not redistribute these
libraries; `fetch_deps.ps1` downloads them from the release above, and by doing
so you accept Qualcomm's terms.

## C++ libraries (fetched by CMake)

| Component | Version | License | Source |
|---|---|---|---|
| CLI11 | 2.7.2 | BSD-3-Clause | <https://github.com/CLIUtils/CLI11> |
| nlohmann/json | 3.12.0 | MIT | <https://github.com/nlohmann/json> |
| GoogleTest (tests only) | 1.18.0 | BSD-3-Clause | <https://github.com/google/googletest> |

## Python packages (installed from `requirements.txt` / `requirements-ci.txt`)

Main direct dependencies; the full pinned list, including transitive
dependencies, is in [requirements.txt](requirements.txt).

| Package | Version | License | Source |
|---|---|---|---|
| onnxruntime | 1.32.0 | MIT | <https://pypi.org/project/onnxruntime/> |
| onnxruntime-qnn (bundles the QNN backend libraries above) | 2.8.0 | MIT; QNN libraries under the Qualcomm license above | <https://pypi.org/project/onnxruntime-qnn/> |
| onnx | 1.23.2 | Apache-2.0 | <https://github.com/onnx/onnx> |
| torch | 2.13.0 (CPU) | BSD-3-Clause (with bundled components under other licenses) | <https://github.com/pytorch/pytorch> |
| torchvision | 0.28.0 | BSD-3-Clause | <https://github.com/pytorch/vision> |
| timm | 1.0.31 | Apache-2.0 | <https://github.com/huggingface/pytorch-image-models> |
| numpy | 2.5.5 | BSD-3-Clause (with bundled components under other licenses) | <https://numpy.org> |
| pandas | 3.1.1 | BSD-3-Clause | <https://pandas.pydata.org> |
| Pillow | 13.0.0 | MIT-CMU | <https://python-pillow.org> |
| huggingface_hub | 2.4.0 | Apache-2.0 | <https://github.com/huggingface/huggingface_hub> |
| jsonschema | 4.26.0 | MIT | <https://github.com/python-jsonschema/jsonschema> |
| pytest | 9.2.0 | MIT | <https://github.com/pytest-dev/pytest> |

## Training notebook (Colab only)

`notebooks/train_autoencoder.ipynb` trains the Conv1D autoencoder in Google
Colab and exports it to ONNX; these packages are installed there by the
notebook's first cell, pinned by version, and are not used on the target
machine. The notebook also installs onnx and onnxruntime (versions as above).

| Package | Version | License | Source |
|---|---|---|---|
| TensorFlow | 2.21.0 | Apache-2.0 | <https://github.com/tensorflow/tensorflow> |
| Keras | 3.15.1 | Apache-2.0 | <https://github.com/keras-team/keras> |
| tf2onnx | 1.17.0 | Apache-2.0 | <https://github.com/onnx/tensorflow-onnx> |

## Continuous integration only

Used by `.github/workflows/ci.yml` on GitHub-hosted runners, not part of any
build or result.

| Component | Version | License | Source |
|---|---|---|---|
| OpenCppCoverage (coverage job; installer pinned by SHA-256) | 0.9.9.0 | GPL-3.0 | <https://github.com/OpenCppCoverage/OpenCppCoverage> |
| GitHub Actions `actions/checkout`, `actions/setup-python`, `actions/cache`, `actions/upload-artifact` (pinned by commit) | see `ci.yml` | MIT | <https://github.com/actions> |

## Models (downloaded by the scripts, not redistributed)

| Model | Used by | License | Source |
|---|---|---|---|
| MobileNetV2, torchvision weights `MobileNet_V2_Weights.IMAGENET1K_V2` | `scripts/export_models.py` | torchvision code: BSD-3-Clause; weights trained on ImageNet: see upstream | <https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html> |
| ResNet-18, torchvision weights `ResNet18_Weights.IMAGENET1K_V1` | `scripts/export_models.py` | torchvision code: BSD-3-Clause; weights trained on ImageNet: see upstream | <https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html> |
| EfficientNet-Lite0, timm weights `efficientnet_lite0.ra_in1k` (Hugging Face revision `29046ba6`) | `scripts/export_models.py` | Model card: Apache-2.0; timm: Apache-2.0; weights trained on ImageNet: see upstream | <https://huggingface.co/timm/efficientnet_lite0.ra_in1k>, <https://github.com/huggingface/pytorch-image-models> |
| Qualcomm AI Hub MobileNet-v2 (float and w8a8 ONNX assets, AI Hub Models 0.67.0) | `scripts/get_aihub_baseline.py` | Model card: Apache-2.0; `qai_hub_models`: BSD-3-Clause | <https://huggingface.co/qualcomm/MobileNet-v2>, <https://github.com/qualcomm/ai-hub-models> |

Exported and quantized models are written to `models/`, which is not committed.
The one model whose weights are committed, the Conv1D autoencoder in
`models/sources/conv1d_autoencoder/`, was trained for EdgeBench by the notebook
above and is covered by EdgeBench's MIT License.

## Datasets (downloaded by the scripts, not redistributed)

| Dataset | Used by | License | Source |
|---|---|---|---|
| ImageNetV2 matched-frequency (`imagenetv2-matched-frequency.tar.gz`, Hugging Face revision `d626240b`) | `scripts/get_data.py`, accuracy evaluation | Code repository and Hugging Face dataset card: MIT; the images were collected from Flickr for use under the same terms as ImageNet (non-commercial research and educational use): see upstream | <https://github.com/modestyachts/ImageNetV2>, <https://huggingface.co/datasets/vaishaal/ImageNetV2>, <https://www.image-net.org/download.php> |
| Imagenette (`imagenette2-320`) | `scripts/get_data.py`, calibration and the feasibility accuracy scripts | Repository: Apache-2.0; the images are a subset of ImageNet: see upstream for their terms | <https://github.com/fastai/imagenette> |
| Numenta Anomaly Benchmark: `data/realKnownCause/machine_temperature_system_failure.csv` and `labels/combined_windows.json` (commit `ea702d75`) | `scripts/get_data.py`, the time-series autoencoder (its committed weights in `models/sources/conv1d_autoencoder/` were trained on this series; the data itself is not committed) | MIT (Copyright 2014-2024 Numenta Inc.; repository license covering data and labels) | <https://github.com/numenta/NAB> |

The archives, the extracted images, the NAB files and the split lists derived
from them are kept under `data/`, which is not committed. The committed
manifests in `datasets/` hold only paths, class indices and SHA-256 hashes of
the images, and for NAB the source hashes, index ranges of the splits and the
normalization constants, not the data.

`scripts/common/imagenet_classes.json` (ImageNet-1k class index, WordNet ID and
name) combines the WordNet IDs from ImageNetV2's `data/metadata/class_info.json`
(MIT) with the category names from torchvision's weights metadata
(BSD-3-Clause).
