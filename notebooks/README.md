# notebooks

Notebooks that run outside this machine.

- `train_autoencoder.ipynb`: trains the Keras time-series autoencoder in Google
  Colab (TensorFlow has no native Windows on Arm build) and exports it to ONNX
  at batch sizes 1, 4 and 8. It clones this repository at a pinned commit and
  uses its data code, so the windows and splits are exactly those of
  `datasets/nab-machine-temperature/splits.json`. The bundle of the run the
  model pipeline uses is committed in
  [`models/sources/conv1d_autoencoder/`](../models/sources/conv1d_autoencoder/).
  See [docs/autoencoder.md](../docs/autoencoder.md).

Notebooks are committed with outputs cleared; `scripts/tests/test_autoencoder.py`
checks that, the pinned package versions, and that the pinned commit holds the
current split manifest.
