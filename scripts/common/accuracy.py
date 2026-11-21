"""Top-1 accuracy and agreement metrics on the Imagenette eval split.

Shared by the CPU and NPU evaluation scripts so that both compute the numbers
the same way. All functions take logits as an [N, 1000] array in the order of
the image list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.preprocess import load_imagenette_classes, wnid_of

METRIC_DEFINITIONS = {
    "top1_1000": "argmax over all 1000 ImageNet logits equals the label",
    "top1_imagenette10": "argmax over the 10 Imagenette class logits equals the label",
    "agreement": "fraction of images whose predicted class (same definition) matches the reference",
    "max_abs_logit_diff": "largest absolute difference over all logits of all images",
}

DATASET_NOTE = (
    "Most Imagenette val images come from the ImageNet-1k train set, so absolute accuracy is "
    "optimistic for ImageNet-pretrained weights; comparisons between variants are unaffected."
)


@dataclass(frozen=True)
class Labels:
    imagenet_index: np.ndarray  # [N] ImageNet-1k label of each image
    class_indices: np.ndarray  # [10] ImageNet indices of the Imagenette classes
    from_imagenet_val: int  # images whose file name marks them as ImageNet val images


def labels_for(images: list[str]) -> Labels:
    classes = load_imagenette_classes()
    wnid_to_index = {c.wnid: c.imagenet_index for c in classes}
    return Labels(
        imagenet_index=np.array([wnid_to_index[wnid_of(p)] for p in images]),
        class_indices=np.array([c.imagenet_index for c in classes]),
        from_imagenet_val=sum(Path(p).name.startswith("ILSVRC2012_val_") for p in images),
    )


def predictions(logits: np.ndarray, labels: Labels) -> tuple[np.ndarray, np.ndarray]:
    """(1000-class predictions, 10-class predictions as ImageNet indices)."""
    pred_1000 = logits.argmax(axis=1)
    pred_10 = labels.class_indices[logits[:, labels.class_indices].argmax(axis=1)]
    return pred_1000, pred_10


def accuracy(logits: np.ndarray, labels: Labels) -> dict:
    p1000, p10 = predictions(logits, labels)
    return {
        "top1_1000": float(np.mean(p1000 == labels.imagenet_index)),
        "top1_imagenette10": float(np.mean(p10 == labels.imagenet_index)),
        "correct_1000": int(np.sum(p1000 == labels.imagenet_index)),
        "correct_imagenette10": int(np.sum(p10 == labels.imagenet_index)),
    }


def agreement(logits: np.ndarray, reference: np.ndarray, labels: Labels) -> dict:
    a1000, a10 = predictions(logits, labels)
    r1000, r10 = predictions(reference, labels)
    return {
        "top1_1000": float(np.mean(a1000 == r1000)),
        "top1_imagenette10": float(np.mean(a10 == r10)),
    }


def max_abs_diff(logits: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(logits.astype(np.float64) - reference.astype(np.float64))))
