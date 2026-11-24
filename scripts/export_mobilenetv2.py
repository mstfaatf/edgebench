"""Export torchvision MobileNetV2 to ``models/mobilenetv2/mobilenetv2_fp32_b1.onnx``.

Equivalent to ``python scripts/export_models.py --model mobilenetv2 --batch 1``;
kept so existing instructions keep working. See ``scripts/export_models.py``
for the checks and docs/models.md for the export settings.

Usage:
    python scripts/export_mobilenetv2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import export_models  # noqa: E402

if __name__ == "__main__":
    sys.exit(export_models.main(["--model", "mobilenetv2", "--batch", "1"] + sys.argv[1:]))
