"""Tests for scripts/check_doc_numbers.py (standard library only; run in CI)."""

from __future__ import annotations

import copy

import pytest

import check_doc_numbers as cdn


@pytest.fixture(scope="module")
def committed():
    return cdn.read_docs(), cdn.load_sources()


def problems(docs, src):
    return cdn.check(docs, src).problems


@pytest.mark.parametrize("text, value, scale, expected", [
    ("59.56", 0.5956, 100, True),
    ("59.57", 0.5956, 100, False),
    ("+0.01", 0.0001, 100, True),
    ("-0.38", -0.0038, 100, True),
    ("1", 1.0, 1, True),
    ("0.0421", 0.0421, 1, True),
    ("3.45e-4", 0.00034549495594041885, 1, True),
    ("3.46e-4", 0.00034549495594041885, 1, False),
    ("2e-117", 2.1e-117, 1, True),
    ("1,766", 1766, 1, True),
    ("99.92%", 0.9991985752448798, 100, True),
    ("0.125", 0.125, 1, True),       # exact
    ("0.13", 0.125, 1, True),        # half-up of the shortest repr
    ("0.12", 0.125, 1, True),        # half-even
    ("−4.67", -4.67, 1, True),  # typographic minus
])
def test_agrees(text, value, scale, expected):
    assert cdn.agrees(text, value, scale) is expected


def test_not_a_number():
    with pytest.raises(ValueError):
        cdn.agrees("n/a", 1.0)


def test_committed_docs_match_the_results(committed):
    docs, src = committed
    c = cdn.check(docs, src)
    assert c.problems == []
    assert c.checked > 400


def test_main_exit_code(capsys):
    assert cdn.main() == 0
    assert "OK" in capsys.readouterr().out


def test_table_reader():
    doc = "text\n| a | b |\n|---|---|\n| 1 | **2** |\n| 3 | x \\| y |\n\nafter\n"
    assert cdn.table(doc, "| a |") == [["1", "2"], ["3", "x \\| y"]]
    assert cdn.table(doc, "| z |") is None


def _tamper_doc(docs, name, old, new):
    assert old in docs[name], old
    changed = dict(docs)
    changed[name] = docs[name].replace(old, new, 1)
    return changed


@pytest.mark.parametrize("name, old, new", [
    # accuracy table cell
    ("docs/accuracy.md", "| 59.56 (58.59-60.52) |", "| 59.57 (58.59-60.52) |"),
    # comparison table count
    ("docs/accuracy.md", "| -0.05 | 95.22% | 97 | 102 |", "| -0.05 | 95.22% | 98 | 102 |"),
    # summarising sentence
    ("docs/accuracy.md", "within 0.15 points of CPU INT8", "within 0.16 points of CPU INT8"),
    ("README.md", "within 0.15 top-1 points", "within 0.25 top-1 points"),
    # batch 8 on the NPU: table, summary sentence, eval-quick count
    ("docs/accuracy.md", "| 57.21 | 57.14 | -0.07 | 98.84% |", "| 57.21 | 57.15 | -0.07 | 98.84% |"),
    ("docs/accuracy.md", "within 0.07 top-1 points of batch 4", "within 0.05 top-1 points of batch 4"),
    ("docs/accuracy.md", "| 152 of 1,000 |", "| 151 of 1,000 |"),
    # batch equivalence with finalization mode 1
    ("docs/models.md", "With finalization mode 1, all 36 image INT8",
     "With finalization mode 1, all 35 image INT8"),
    ("docs/models.md", "| NPU, INT8, finalization mode 1 | same | same | 0 |\n| `resnet18` | INT8 (all four) | CPU",
     "| NPU, INT8, finalization mode 1 | same | differs | 0 |\n| `resnet18` | INT8 (all four) | CPU"),
    # anomaly tables
    ("docs/autoencoder.md", "| 0.813 | 0.714 | 0.796 |", "| 0.813 | 0.715 | 0.796 |"),
    ("docs/autoencoder.md", "| 155 (12 h 55 min) | 155 (12 h 55 min) | 153",
     "| 155 (12 h 55 min) | 156 (13 h) | 153"),
    ("docs/accuracy.md", "| CPU INT8 | 0.813 | 0.714 | 0.635 | 4 of 4 |",
     "| CPU INT8 | 0.813 | 0.714 | 0.635 | 3 of 4 |"),
    # split manifest, training record, manifest sizes, model_info example
    ("docs/autoencoder.md", "| 8,408 | 8,281 |", "| 8,408 | 8,282 |"),
    ("docs/autoencoder.md", "best epoch 133", "best epoch 132"),
    ("docs/autoencoder.md", "| 44,765 |", "| 44,766 |"),
    ("docs/models.md", "sha256 dd64f81b21ea", "sha256 dd64f81b21eb"),
    ("docs/models.md", "| `cpu` | all 66 |", "| `cpu` | all 65 |"),
    # abbreviated hash
    ("docs/autoencoder.md", "`a73173e6…f37c`", "`a73173e6…f37d`"),
    # NPU batch-size findings
    ("docs/npu.md", "| 7 | 0 of 8 | 0 of 8 |", "| 7 | 1 of 8 | 0 of 8 |"),
    ("docs/npu.md", "| 6.89 MiB | 4.59 MiB |", "| 6.89 MiB | 4.60 MiB |"),
    ("docs/npu.md", "| 2.83 | +29% |", "| 2.83 | +30% |"),
    ("docs/npu.md", "differs is 7 for all three", "differs is 8 for all three"),
])
def test_a_changed_document_number_is_reported(committed, name, old, new):
    docs, src = committed
    assert problems(_tamper_doc(docs, name, old, new), src)


def test_a_removed_table_row_is_reported(committed):
    docs, src = committed
    row = "| NPU FP16 | 8 | 3.47e-4 |"
    lines = [ln for ln in docs["docs/autoencoder.md"].splitlines() if not ln.startswith(row)]
    changed = dict(docs, **{"docs/autoencoder.md": "\n".join(lines)})
    assert any("rows, expected" in p for p in problems(changed, src))


def test_a_missing_sentence_is_reported(committed):
    docs, src = committed
    changed = _tamper_doc(docs, "docs/accuracy.md", "INT8 and FP32 agree on", "INT8 and FP32 match on")
    assert any("sentence not found" in p for p in problems(changed, src))


def test_a_changed_result_is_reported(committed):
    docs, src = committed
    changed = copy.deepcopy(src)
    changed.runs["resnet18_int8_npu"]["metrics"]["top1"]["accuracy"] = 0.5731
    assert problems(docs, changed)
    changed = copy.deepcopy(src)
    changed.anomaly_comparisons["NPU INT8 vs CPU INT8"]["flagged_only_a"] = 10
    assert problems(docs, changed)
    changed = copy.deepcopy(src)
    changed.manifest["conv1d_autoencoder/conv1d_autoencoder_int8_b1.onnx"]["size_bytes"] += 1
    assert problems(docs, changed)
    changed = copy.deepcopy(src)
    changed.npu_batch["models"]["resnet18"]["threshold"]["batches"]["6"][
        "items_equal_b1_reference"] = 7
    assert problems(docs, changed)
    changed = copy.deepcopy(src)
    changed.npu_batch["layer_scan"]["rows"][3]["differing"] = 1
    assert problems(docs, changed)


def test_an_unknown_full_hash_is_reported(committed):
    docs, src = committed
    changed = dict(docs)
    changed["docs/models.md"] = docs["docs/models.md"] + "\n`" + "ab" * 32 + "`\n"
    assert any("not in any committed file" in p for p in problems(changed, src))
