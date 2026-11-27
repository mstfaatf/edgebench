# NPU runs

How EdgeBench runs models on the Snapdragon NPU (Qualcomm QNN execution
provider, HTP backend) and what to know when reading NPU numbers. Options and
exit codes are in [cli.md](cli.md#npu-runs).

## Batch size and INT8 outputs

### Observation

On the NPU, the INT8 image models (MobileNetV2, ResNet-18, EfficientNet-Lite0)
give logits at batch 4 that are bit-identical to batch 1, but at batch 8 every
image's logits differ from batch 1, and the top-1 class changes on a few of the
1,000 eval-quick images ([models.md](models.md#batch-equivalence),
[accuracy.md](accuracy.md)). The same files on the CPU, the FP32 files on the NPU
(FP16 compute) and the INT8 autoencoder on the NPU are batch-consistent, so the
quantized graph itself computes the same thing at every batch size; the
difference comes from how the HTP executes it. The C++ engine and Python ONNX
Runtime give the same b8 outputs bit for bit.

`scripts/investigate_batch_equivalence.py` regenerates everything below and
writes it to [`docs/findings/npu_int8_batch_size.json`](findings/npu_int8_batch_size.json)
(about 20 minutes on the NPU; nothing under `models/` changes). All sessions are
strict QNN HTP sessions with the reference options (burst performance mode).
Latencies are a quick check (interleaved, 10 rounds of 50 runs, medians), not a
benchmark.

### Where the threshold is

The INT8 files were built at batch sizes 2, 3, 5, 6, 7 and 16 with the shipped
recipe, reusing the batch-1 calibration ranges: every batch size has exactly the
shipped scales and zero points, and the rebuilt b1 and b4 graphs are
byte-identical to the shipped files. Each batch size ran the 8 reference images
(repeated to fill a batch); the table counts the images whose logits equal the
batch-1 reference bit for bit, next to the largest uint8 activation in the graph
(the first expanded block's output for MobileNetV2 and EfficientNet-Lite0, the
first convolution's output for ResNet-18). The device's VTCM is 8 MiB.

| Batch | MobileNetV2 | ResNet-18 | EfficientNet-Lite0 | Largest activation, MobileNetV2 | Largest activation, ResNet-18 | Largest activation, EfficientNet-Lite0 |
|---|---|---|---|---|---|---|
| 1 | 8 of 8 | 8 of 8 | 8 of 8 | 1.15 MiB | 0.77 MiB | 1.15 MiB |
| 2 | 8 of 8 | 8 of 8 | 8 of 8 | 2.30 MiB | 1.53 MiB | 2.30 MiB |
| 3 | 8 of 8 | 8 of 8 | 8 of 8 | 3.45 MiB | 2.30 MiB | 3.45 MiB |
| 4 | 8 of 8 | 8 of 8 | 8 of 8 | 4.59 MiB | 3.06 MiB | 4.59 MiB |
| 5 | 8 of 8 | 8 of 8 | 8 of 8 | 5.74 MiB | 3.83 MiB | 5.74 MiB |
| 6 | 8 of 8 | 8 of 8 | 8 of 8 | 6.89 MiB | 4.59 MiB | 6.89 MiB |
| 7 | 0 of 8 | 0 of 8 | 0 of 8 | 8.04 MiB | 5.36 MiB | 8.04 MiB |
| 8 | 0 of 8 | 0 of 8 | 0 of 8 | 9.19 MiB | 6.12 MiB | 9.19 MiB |
| 16 | 0 of 8 | 0 of 8 | 0 of 8 | 18.38 MiB | 12.25 MiB | 18.38 MiB |

The smallest batch size that differs is 7 for all three models. For MobileNetV2
and EfficientNet-Lite0 the largest activation crosses 8 MiB between b6 and b7,
but for ResNet-18 it is only 5.36 MiB at b7, so the threshold does not follow
the activation size. From b7 on the outputs are the same at every batch size
(b7, b8 and b16 give identical logits per image) and do not depend on an
image's position in the batch; repeated runs are identical.

### Where it starts

Of the 33 activations of ResNet-18 (the output of every activation
DequantizeLinear), compared at b1 and b7, only the last one, `logits`, differs:
every intermediate tensor, including the pooled features that enter the
classifier, is bit-identical. Each comparison is a truncated model ending at
that tensor. Truncation can change how the HTP compiles the end of a graph, so
the classifier tail was also cut out as sub-models fed with uint8 activations:
the real activations entering GlobalAveragePool for the 8 reference images,
and, for the Gemm alone, 64 random uint8 vectors. The table counts the outputs
that differ from the same sub-model at b1:

| Model | Sub-model | b2 | b4 | b6 | b7 | b8 |
|---|---|---|---|---|---|---|
| MobileNetV2 | GlobalAveragePool | 0 | 0 | 0 | 0 | 0 |
| MobileNetV2 | GlobalAveragePool + Flatten | 0 | 0 | 129 of 10,240 | 129 of 10,240 | 965 of 10,240 |
| MobileNetV2 | GlobalAveragePool to logits | 0 | 0 | 0 | 1,671 of 8,000 | 1,671 of 8,000 |
| MobileNetV2 | Gemm only | 0 | 0 | 0 | 12,180 of 64,000 | 12,180 of 64,000 |
| ResNet-18 | GlobalAveragePool | 0 | 0 | 0 | 0 | 0 |
| ResNet-18 | GlobalAveragePool + Flatten | 0 | 0 | 0 | 0 | 0 |
| ResNet-18 | GlobalAveragePool to logits | 0 | 0 | 0 | 1,713 of 8,000 | 1,713 of 8,000 |
| ResNet-18 | Gemm only | 0 | 0 | 0 | 13,683 of 64,000 | 13,683 of 64,000 |
| EfficientNet-Lite0 | GlobalAveragePool | 0 | 0 | 0 | 0 | 0 |
| EfficientNet-Lite0 | GlobalAveragePool + Flatten | 0 | 0 | 142 of 10,240 | 142 of 10,240 | 1,153 of 10,240 |
| EfficientNet-Lite0 | GlobalAveragePool to logits | 0 | 0 | 0 | 1,866 of 8,000 | 1,866 of 8,000 |
| EfficientNet-Lite0 | Gemm only | 0 | 0 | 0 | 6,615 of 64,000 | 6,615 of 64,000 |

The final fully connected layer (ONNX Gemm, QNN FullyConnected) reproduces the
threshold on its own: its sub-model is bit-identical up to b6 and differs from
b7, and the tail from GlobalAveragePool to the logits differs exactly as much as
the full model does (1,671 / 1,713 / 1,866 of 8,000, see below). The
"GlobalAveragePool + Flatten" sub-models of MobileNetV2 and EfficientNet-Lite0
also differ, already at b6, but only when the Flatten output is the end of the
graph: in the full model and in the tail sub-model the b6 logits are
bit-identical, so this is an effect of cutting the graph there, not part of the
divergence. The QNN graphs the execution provider builds for the Gemm
sub-model at b6 and b7 (`dump_json_qnn_graph`) differ only in the batch
dimension of 3 tensors: the choice that changes the results is made inside the
HTP backend when it finalizes the graph.

How much to trust this: the full-model layer scan covers ResNet-18 only; for
the other two models only the tail was isolated. Their pooled features (the
GlobalAveragePool output) are bit-identical at b1 and b7, and the tail
sub-model reproduces the full model's logits difference exactly, so an earlier
divergence that disappears before the classifier cannot be excluded for them,
but nothing points to one.

### How large the differences are

In units of the logits' quantization scale (LSB), every differing element is +1
LSB: at b7 and above the HTP never produces a smaller value than at b1, and
never one more than 1 LSB away. Full models, b8 against b1, 8 reference images:

| Model | Output scale | Outputs | +1 LSB | Top-1 changes |
|---|---|---|---|---|
| MobileNetV2 | 0.0570 | 8,000 | 1,671 (20.9) | 0 |
| ResNet-18 | 0.1686 | 8,000 | 1,713 (21.4) | 0 |
| EfficientNet-Lite0 | 0.0823 | 8,000 | 1,866 (23.3) | 0 |

(+1 LSB: count and percentage of the outputs.) A one-sided shift is not what a
different accumulation order gives (that moves values both ways); it is what a
different requantization of the Gemm's int32 accumulator gives. To see which
batch size is closer to the QDQ definition, the Gemm was computed in exact
integer arithmetic for the 64 random vectors (int32 accumulator with the int32
bias, scaled by input scale x per-channel weight scale / output scale, plus the
output zero point; the bias scale is input scale x weight scale, as the quantizer
writes it) and the NPU outputs compared with that exact value, in output LSB,
over the outputs not clipped to the uint8 range:

| Model | Batch | Mean error (LSB) | Standard deviation (LSB) | Range (LSB) | Exactly rounded |
|---|---|---|---|---|---|
| MobileNetV2 | 1 | -0.001 | 0.29 | -0.56 to 0.56 | 97.8 |
| MobileNetV2 | 8 | 0.238 | 0.51 | -0.56 to 1.55 | 74.7 |
| ResNet-18 | 1 | -0.001 | 0.29 | -0.55 to 0.55 | 97.7 |
| ResNet-18 | 8 | 0.239 | 0.51 | -0.55 to 1.55 | 74.5 |
| EfficientNet-Lite0 | 1 | 0.005 | 0.29 | -0.56 to 0.56 | 97.7 |
| EfficientNet-Lite0 | 8 | 0.260 | 0.52 | -0.56 to 1.55 | 73.0 |

("Exactly rounded": percentage of outputs equal to the exact value rounded to
the nearest integer.) At b1 the HTP rounds to nearest: the error is centred on
zero with the spread of plain rounding (0.29 = 1/sqrt(12)), and the outputs that
are not exactly rounded lie within 0.06 LSB of a rounding boundary. From b7 on,
the Gemm output is on average about a quarter LSB higher, up to 1.55 LSB above
the exact value (b1: at most 0.56). The b7+ kernel requantizes less precisely;
nothing indicates an error (no output is off by more than 1 LSB from b1, and
the shift is the same for every model and input). On the
8 reference images no top-1 class changes; on the 1,000 eval-quick images 6, 10
and 5 do (below and [models.md](models.md#batch-equivalence)).

### Options

Full models at b1 (control), b6, b7 and b8, each option added to the reference
options ("same": all 8 images of all three models equal the batch-1 reference
bit for bit; "differs": none does):

| Option | b1 | b6 | b7 | b8 |
|---|---|---|---|---|
| `default` | same | same | differs | differs |
| `vtcm_mb=2` | same | same | differs | differs |
| `vtcm_mb=4` | same | same | differs | differs |
| `vtcm_mb=8` | same | same | differs | differs |
| `htp_graph_finalization_optimization_mode=0` | same | same | differs | differs |
| `htp_graph_finalization_optimization_mode=1` | same | same | same | same |
| `htp_graph_finalization_optimization_mode=2` | same | same | differs | differs |
| `htp_graph_finalization_optimization_mode=3` | same | same | differs | differs |
| `htp_performance_mode=balanced` | same | same | differs | differs |
| `htp_performance_mode=power_saver` | same | same | differs | differs |
| `htp_performance_mode=high_performance` | same | same | differs | differs |

A smaller VTCM does not move the threshold, and the performance mode changes
no output (as expected). Only graph finalization mode 1 (the quickest
finalization, fewer graph optimizations) makes b7 and b8 bit-identical to b1;
it does the same for the Gemm sub-model on its own. On all 1,000 eval-quick
images, and its latency cost (median ms per inference of the whole batch;
mode 1 at b1 is also bit-identical to the default on all 1,000 images):

| Model | Default b8: top-1 changes | Mode 1 b8: bitwise equal to b1 | b1 default | b1 mode 1 | b8 default | b8 mode 1 | Mode 1 cost at b8 |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | 6 of 1,000 | 1,000 of 1,000 | 0.32 | 0.32 | 1.98 | 2.29 | +15% |
| ResNet-18 | 10 of 1,000 | 1,000 of 1,000 | 0.37 | 0.44 | 2.18 | 2.83 | +29% |
| EfficientNet-Lite0 | 5 of 1,000 | 1,000 of 1,000 | 0.34 | 0.35 | 2.22 | 2.45 | +10% |

Mode 1 is not used: EdgeBench keeps the backend's default finalization, and NPU
INT8 runs at b8 carry the `batch_not_equivalent` warning.

### Conclusion

The NPU INT8 image models differ at batch 7 and above because of one layer, the
final fully connected layer: from 7 rows on, the HTP backend's default graph
finalization compiles QNN FullyConnected into a kernel whose requantization of
the output rounds up to 1 LSB higher than at 1 to 6 rows (+0.24 LSB on
average). The quantized graph and the QNN graph are the same at every batch
size; nothing before the classifier changes (checked layer by layer for
ResNet-18). This is a different computation, not an error. It is not a memory
effect: the threshold is the same for all three models although their
activations differ in size, and the VTCM size does not move it. The
autoencoder is not affected because it has no fully connected layer, and FP16
runs use different kernels.

Confidence: high that the final Gemm is where b7+ differs and that
finalization mode 1 removes it (reproduced in the full models, in the isolated
Gemm, and on 1,000 images). What the backend's kernel does internally (which
kernel, why from 7 rows) is inferred from the outputs; the QNN SDK does not
report it, and settling it would need the backend's per-op profiling or
Qualcomm's documentation of the FullyConnected kernels.
