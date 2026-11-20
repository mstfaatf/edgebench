# ml/tensorflow

The Keras model is trained in Google Colab, because TensorFlow has no native
Windows ARM64 build. This directory holds the notebook/training code and its own
Colab requirements. Only the exported ONNX model and its reference outputs come
back to this repository's pipeline; TensorFlow is never installed locally.
