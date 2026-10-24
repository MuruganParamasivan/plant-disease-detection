"""Model builders: a baseline CNN and a transfer-learning classifier.

Both models expect inputs that are **already preprocessed** by the data
pipeline (``mobilenet_v2.preprocess_input`` -> float32 in [-1, 1]); there is
no Rescaling/Normalization layer inside the model. Keeping preprocessing in
one place (``src/data.py``) is what lets the Streamlit app reproduce training
conditions exactly.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from src import config

# Backbones exposed through --backbone. MobileNetV2 is the locked default:
# small enough to fine-tune on a free Colab GPU, strong ImageNet features.
BACKBONES = {
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "ResNet50": tf.keras.applications.ResNet50,
}


def build_baseline_cnn(
    num_classes: int,
    img_size: int = config.IMG_SIZE,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """A small Conv/Pool stack trained from scratch - the honest baseline.

    Deliberately modest (~0.5M params): its job is to show how much the
    ImageNet features in the transfer model are actually worth.
    """
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


HEAD_LAYER_NAMES = ("gap", "head_dropout", "predictions")


def build_transfer_model(
    num_classes: int,
    img_size: int = config.IMG_SIZE,
    backbone: str = "MobileNetV2",
    fine_tune_at: Optional[int] = None,
    dropout: float = 0.2,
) -> tf.keras.Model:
    """ImageNet backbone (frozen) + GlobalAveragePooling + dropout + softmax.

    The backbone is **inlined** into a single flat functional graph
    (``input_tensor=inputs``) rather than nested as a sub-model. That matters
    for Grad-CAM: in Keras 3 a nested model's ``.output`` belongs to its own
    inner graph, so ``Model(model.inputs, [nested.output, model.output])``
    raises "not connected to inputs". Inlining keeps every backbone
    activation addressable from the outer graph, before and after save/load.

    Parameters
    ----------
    fine_tune_at : if given, unfreeze the backbone from this layer index
        onwards (BatchNorm layers stay frozen - unfreezing them at batch 32
        destabilises the pretrained statistics).
    """
    if backbone not in BACKBONES:
        raise ValueError(
            "Unknown backbone '{}'. Available: {}".format(
                backbone, ", ".join(BACKBONES)
            )
        )
    if backbone != "MobileNetV2":
        print(
            "[model] NOTE: the data pipeline applies mobilenet_v2 preprocessing "
            "([-1, 1]). {} was pretrained with a different scheme; expect a "
            "small accuracy hit unless you adjust src/data.py.".format(backbone)
        )

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    base = BACKBONES[backbone](
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        input_shape=(img_size, img_size, 3),
    )
    # Freeze layer-by-layer: a BatchNormalization layer with trainable=False
    # also runs in inference mode, which is what a frozen backbone needs.
    for layer in base.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="{}_transfer".format(backbone.lower()))

    if fine_tune_at is not None:
        unfreeze_top_layers(model, len(backbone_layers(model)) - fine_tune_at)
    return model


def backbone_layers(model: tf.keras.Model) -> list:
    """The backbone layers of a (possibly reloaded) transfer model.

    Everything except the input layer and the classification head; works on a
    flat inlined graph, so it still works after ``keras.models.load_model``.
    """
    return [
        layer
        for layer in model.layers
        if layer.name not in HEAD_LAYER_NAMES
        and not isinstance(layer, tf.keras.layers.InputLayer)
    ]


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int) -> int:
    """Unfreeze the top ``num_layers`` backbone layers for fine-tuning.

    BatchNormalization layers are left frozen on purpose.
    """
    base = backbone_layers(model)
    if not base:
        raise ValueError("This model has no backbone to fine-tune.")
    fine_tune_at = max(0, len(base) - num_layers)
    for layer in base[:fine_tune_at]:
        layer.trainable = False
    for layer in base[fine_tune_at:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)
    trainable = sum(1 for layer in base if layer.trainable)
    print(
        "[model] fine-tuning: unfroze {} of {} backbone layers "
        "(from index {}; BatchNorm kept frozen)".format(
            trainable, len(base), fine_tune_at
        )
    )
    return fine_tune_at


def _output_rank(layer) -> Optional[int]:
    try:
        return len(layer.output.shape)
    except Exception:
        return None


def find_last_conv_layer(model: tf.keras.Model):
    """Locate the feature map Grad-CAM should attribute to.

    Walks the graph backwards for the last rank-4 activation: ``out_relu``
    (7x7x1280) for MobileNetV2, the final conv block for the baseline CNN.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D) or _output_rank(layer) == 4:
            return layer
    raise ValueError("No 4-D feature map found - cannot build Grad-CAM.")


def get_model(
    name: str,
    num_classes: int,
    img_size: int = config.IMG_SIZE,
    backbone: str = "MobileNetV2",
    fine_tune_at: Optional[int] = None,
) -> tf.keras.Model:
    """Factory used by train.py: name is 'baseline' or 'transfer'."""
    if name == "baseline":
        return build_baseline_cnn(num_classes, img_size)
    if name == "transfer":
        return build_transfer_model(num_classes, img_size, backbone, fine_tune_at)
    raise ValueError("Unknown model '{}' (expected 'baseline' or 'transfer')".format(name))
