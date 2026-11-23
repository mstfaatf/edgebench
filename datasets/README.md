# datasets

Manifests of the images EdgeBench uses for accuracy evaluation and quantization
calibration. The images themselves are not part of this repository:
`scripts/get_data.py` downloads them into `data/` (gitignored) and checks them
against these files. See [Preparing data and models](../docs/setup.md#preparing-data-and-models).

| Manifest | Images | Dataset root after download |
|---|---|---|
| `imagenetv2-matched-frequency/eval-full.tsv` | 10,000 (10 per class): published accuracy | `data/imagenetv2/imagenetv2-matched-frequency-format-val/` |
| `imagenetv2-matched-frequency/eval-quick.tsv` | 1,000 (1 per class, seed 20261123): fast iteration | same |
| `imagenette/calibration.tsv` | 200 (20 per class, Imagenette train): calibration | `data/imagenette/imagenette2-320/` |

## Format

UTF-8, LF line endings, tab-separated, with a header row:

```
path	label	sha256
0/58fbc3e79ef15162b7726de04e98c90bb91a3055.jpeg	0	3a9fea9a...
```

- `path`: relative to the dataset root, forward slashes.
- `label`: ImageNet-1k class index (0-999), in torchvision's order; see
  `scripts/common/imagenet_classes.json`. For ImageNetV2 it equals the class
  directory name read as an integer.
- `sha256`: SHA256 of the image file's bytes.

Rows are sorted by label (as an integer), then path. `scripts/common/manifests.py`
reads and writes this format; `scripts/tests/test_manifests.py` checks the
files, including that eval-quick is the seeded sample of eval-full and that no
calibration image has the same content as an evaluation image.

To confirm a local copy: `python scripts/get_data.py --dataset all --verify`.

## Terms

The images are not redistributed here. ImageNetV2 and Imagenette images are
subject to the same terms as ImageNet (non-commercial research and educational
use); see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
