"""Grad-CAM, implemented directly against the model's last conv feature map.

Shared by ``src/evaluate.py`` and the Streamlit app so the explanation shown
in the demo is produced by exactly the same code as the one in the report.

Why it is written this way: the transfer model inlines its backbone into a
single functional graph (see ``src/model.py``), so the last conv activation
(``out_relu``, 7x7x1280 for MobileNetV2) is addressable from the model's own
graph and a two-output sub-model can be built without unpacking a nested
sub-model - which Keras 3 does not allow.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf

from src.model import find_last_conv_layer


def make_gradcam_model(model: tf.keras.Model, last_conv_layer=None):
    """Build a model returning (conv feature map, predictions)."""
    layer = last_conv_layer or find_last_conv_layer(model)
    return tf.keras.Model(model.inputs, [layer.output, model.outputs[0]]), layer.name


def compute_heatmap(
    grad_model: tf.keras.Model,
    image_batch,
    class_index: Optional[int] = None,
) -> np.ndarray:
    """Grad-CAM heatmap for one preprocessed image, normalised to [0, 1].

    ``image_batch`` must already be preprocessed and have a leading batch
    dimension of 1 - i.e. exactly what the training pipeline feeds the model.
    """
    image_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(image_batch, training=False)
        tape.watch(conv_out)
        idx = int(tf.argmax(predictions[0])) if class_index is None else int(class_index)
        score = predictions[:, idx]

    grads = tape.gradient(score, conv_out)
    if grads is None:  # pragma: no cover - would mean a broken graph
        raise RuntimeError("Grad-CAM: no gradient reached the conv layer.")

    # Channel importance = spatially averaged gradient, then a ReLU'd
    # weighted sum over channels (Selvaraju et al., 2017).
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    peak = tf.reduce_max(heatmap)
    if float(peak) <= 0:
        # No positive evidence anywhere (happens on out-of-distribution input);
        # return a flat map rather than dividing by zero.
        return np.zeros(heatmap.shape, dtype=np.float32)
    return (heatmap / peak).numpy().astype(np.float32)


def overlay_heatmap(
    display_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend a heatmap over a uint8 RGB image; returns uint8 RGB.

    ``display_image`` is the *human-viewable* image (0-255), not the
    preprocessed tensor.
    """
    import matplotlib

    display_image = np.asarray(display_image)
    if display_image.dtype != np.uint8:
        display_image = np.clip(display_image, 0, 255).astype(np.uint8)

    height, width = display_image.shape[:2]
    resized = tf.image.resize(heatmap[..., None], (height, width)).numpy()[..., 0]
    coloured = matplotlib.colormaps[colormap](resized)[..., :3] * 255.0
    blended = (1 - alpha) * display_image + alpha * coloured
    return np.clip(blended, 0, 255).astype(np.uint8)


def deprocess(preprocessed_image: np.ndarray) -> np.ndarray:
    """Undo mobilenet_v2 preprocessing ([-1, 1] -> uint8) for display."""
    image = (np.asarray(preprocessed_image) + 1.0) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)
