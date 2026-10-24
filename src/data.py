"""
Design notes
------------
* PlantVillage has a single TFDS ``train`` split, so train/val/test are carved
  out with subsplit syntax (80/10/10) - see ``src/config.py``.
* We deliberately do **not** call ``.cache()`` after resizing: 54k images at
  224x224x3 float32 is ~32 GB, which OOMs immediately. The pipeline relies on
  ``.prefetch(AUTOTUNE)`` instead.
* Augmentation is applied to the **training split only**, and *before*
  ``preprocess_input`` so the augmentation layers operate on a plain [0, 255]
  image and any padding they introduce stays in the same numeric space.
* Only the training split is shuffled; val/test keep a fixed order so that
  evaluation artifacts (confusion matrix, Grad-CAM samples) are reproducible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional, Sequence

import tensorflow as tf

from src import config

AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 1000


# --------------------------------------------------------------------------
# Class-name persistence (so evaluate.py / the app never need TFDS)
# --------------------------------------------------------------------------
def save_class_names(
    class_names: Sequence[str],
    classes_filter: Optional[str] = None,
    original_indices: Optional[Sequence[int]] = None,
    path=None,
) -> None:
    """Write class names (and the filtered -> contiguous label map) to JSON."""
    path = path or config.CLASS_NAMES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": config.DATASET_NAME,
        "classes_filter": classes_filter,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        # original TFDS label id -> new contiguous id (only set when filtering)
        "label_mapping": (
            {str(orig): new for new, orig in enumerate(original_indices)}
            if original_indices is not None
            else None
        ),
        "img_size": config.IMG_SIZE,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[data] wrote {len(class_names)} class names -> {path}")


def load_class_names(path=None) -> list:
    """Read the persisted class names. Used by evaluate.py and the app."""
    path = path or config.CLASS_NAMES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run src/train.py first (it persists the "
            "class names alongside the model)."
        )
    return json.loads(path.read_text(encoding="utf-8"))["class_names"]


# --------------------------------------------------------------------------
# Preprocessing / augmentation
# --------------------------------------------------------------------------
def build_augmenter(seed: int = config.SEED):
    """Train-only augmentation stack (locked: flip / rotate 0.1 / zoom 0.1)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.1, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
        ],
        name="augmentation",
    )


def preprocess_image(image, img_size: int = config.IMG_SIZE):
    """Resize + apply MobileNetV2 preprocessing to a single image tensor.

    Shared by the training pipeline and the Streamlit app so that inference
    sees exactly the same numeric range as training ([-1, 1]).
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (img_size, img_size))
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def prep(
    ds,
    training: bool,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    augment: bool = True,
):
    """Resize -> (shuffle) -> batch -> (augment) -> preprocess -> prefetch."""
    ds = ds.map(
        lambda x, y: (tf.image.resize(tf.cast(x, tf.float32), (img_size, img_size)), y),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    if training and augment:
        augmenter = build_augmenter(seed)
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.map(
        lambda x, y: (tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
        num_parallel_calls=AUTOTUNE,
    )
    # No .cache() here on purpose - see the module docstring.
    return ds.prefetch(AUTOTUNE)


# --------------------------------------------------------------------------
# Class filtering (e.g. --classes-filter tomato)
# --------------------------------------------------------------------------
def resolve_classes_filter(all_class_names: Sequence[str], classes_filter: str):
    """Return (kept_names, kept_original_indices) for a class-name prefix."""
    prefix = classes_filter.lower()
    kept = [(i, n) for i, n in enumerate(all_class_names) if n.lower().startswith(prefix)]
    if not kept:
        raise ValueError(
            "--classes-filter '{}' matched none of the {} classes. Examples: {}".format(
                classes_filter, len(all_class_names), ", ".join(all_class_names[:3])
            )
        )
    return [n for _, n in kept], [i for i, _ in kept]


def _filter_and_remap(ds, keep_indices: Sequence[int], num_original: int):
    """Keep only ``keep_indices`` labels and remap them to 0..k-1."""
    lookup = [-1] * num_original
    for new_id, orig_id in enumerate(keep_indices):
        lookup[orig_id] = new_id
    lookup_tensor = tf.constant(lookup, dtype=tf.int64)

    def _keep(_, label):
        return tf.gather(lookup_tensor, tf.cast(label, tf.int64)) >= 0

    def _remap(image, label):
        return image, tf.gather(lookup_tensor, tf.cast(label, tf.int64))

    return ds.filter(_keep).map(_remap, num_parallel_calls=AUTOTUNE)


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------
def _tfds_class_names() -> list:
    """Class names straight from TFDS metadata, falling back to config."""
    try:
        import tensorflow_datasets as tfds

        builder = tfds.builder(config.DATASET_NAME, data_dir=config.TFDS_DATA_DIR)
        return list(builder.info.features["label"].names)
    except Exception as exc:  # offline, or TFDS metadata not available
        print(
            f"[data] TFDS metadata unavailable ({type(exc).__name__}); "
            "using the class list baked into src/config.py"
        )
        return list(config.PLANT_VILLAGE_CLASSES)


def _synthetic_split(num_examples: int, num_classes: int, img_size: int, seed: int):
    """Random-noise stand-in used when the real dataset is not downloaded.

    This exists so ``--smoke-test`` can prove the *plumbing* (shapes, dtypes,
    label range, training loop, checkpointing, Grad-CAM) on a laptop without
    pulling the ~1 GB PlantVillage archive. It teaches the model nothing:
    metrics from a synthetic smoke run are noise by construction.
    """
    rng = tf.random.Generator.from_seed(seed)
    images = tf.cast(
        rng.uniform((num_examples, img_size, img_size, 3), 0, 256, dtype=tf.float32),
        tf.uint8,
    )
    labels = tf.cast(
        rng.uniform((num_examples,), 0, num_classes, dtype=tf.int32), tf.int64
    )
    return tf.data.Dataset.from_tensor_slices((images, labels))


def load_datasets(
    classes_filter: Optional[str] = None,
    smoke: bool = False,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
    data_dir: Optional[str] = None,
    download: Optional[bool] = None,
    persist: bool = True,
    augment: bool = True,
):
    """Build ``(train_ds, val_ds, test_ds, class_names)``.

    Parameters
    ----------
    classes_filter : keep only classes whose name starts with this prefix
        (case-insensitive, e.g. "tomato"); labels are remapped to 0..k-1 and
        the mapping is persisted to models/class_names.json.
    smoke : use the tiny 2% subset - and fall back to synthetic data if the
        dataset has not been downloaded - so the pipeline can be validated
        on a CPU in minutes.
    download : force/forbid a TFDS download. Defaults to False in smoke mode
        (never pull ~1 GB during development) and True otherwise.
    """
    import tensorflow_datasets as tfds

    if download is None:
        download = not smoke

    splits = (
        [config.SMOKE_TRAIN_SPLIT, config.SMOKE_VAL_SPLIT, config.SMOKE_TEST_SPLIT]
        if smoke
        else [config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT]
    )

    all_class_names = _tfds_class_names()

    try:
        raw_splits = tfds.load(
            config.DATASET_NAME,
            split=splits,
            as_supervised=True,
            shuffle_files=False,
            data_dir=data_dir or config.TFDS_DATA_DIR,
            download=download,
        )
        source = "TFDS {} splits={}".format(config.DATASET_NAME, splits)
    except Exception as exc:
        if not smoke:
            raise
        print(
            "\n[data] !! Real dataset unavailable ({}: {}).\n"
            "[data] !! Falling back to SYNTHETIC random images for this smoke test.\n"
            "[data] !! It validates the pipeline only - the metrics are noise.\n"
            "[data] !! Use --download (or train on Colab) to work with real data.\n".format(
                type(exc).__name__, str(exc).splitlines()[0][:200]
            )
        )
        n_classes = len(all_class_names)
        raw_splits = [
            _synthetic_split(128, n_classes, img_size, seed),
            _synthetic_split(32, n_classes, img_size, seed + 1),
            _synthetic_split(32, n_classes, img_size, seed + 2),
        ]
        source = "SYNTHETIC (nothing downloaded)"

    train_raw, val_raw, test_raw = raw_splits

    original_indices = None
    class_names = list(all_class_names)
    if classes_filter:
        class_names, original_indices = resolve_classes_filter(
            all_class_names, classes_filter
        )
        n_original = len(all_class_names)
        train_raw = _filter_and_remap(train_raw, original_indices, n_original)
        val_raw = _filter_and_remap(val_raw, original_indices, n_original)
        test_raw = _filter_and_remap(test_raw, original_indices, n_original)

    if persist:
        save_class_names(class_names, classes_filter, original_indices)

    print("[data] source: {}".format(source))
    print(
        "[data] classes: {}{}".format(
            len(class_names),
            " (filtered by '{}')".format(classes_filter) if classes_filter else "",
        )
    )
    print("[data] {}".format(config.AUTOTUNE_MSG))

    train_ds = prep(train_raw, True, img_size, batch_size, seed, augment=augment)
    val_ds = prep(val_raw, False, img_size, batch_size, seed)
    test_ds = prep(test_raw, False, img_size, batch_size, seed)
    return train_ds, val_ds, test_ds, class_names


def train_labels(
    classes_filter: Optional[str] = None,
    smoke: bool = False,
    data_dir: Optional[str] = None,
    download: Optional[bool] = None,
):
    """Return the training-split labels as a numpy array (for class weights).

    Images are skipped at decode time, so this is a cheap metadata-only pass
    rather than a full read of ~43k JPEGs.
    """
    import numpy as np
    import tensorflow_datasets as tfds

    if download is None:
        download = not smoke

    split = config.SMOKE_TRAIN_SPLIT if smoke else config.TRAIN_SPLIT
    all_class_names = _tfds_class_names()
    try:
        ds = tfds.load(
            config.DATASET_NAME,
            split=split,
            data_dir=data_dir or config.TFDS_DATA_DIR,
            download=download,
            decoders={"image": tfds.decode.SkipDecoding()},
        )
        labels = np.array([int(ex["label"]) for ex in ds.as_numpy_iterator()])
    except Exception as exc:
        if not smoke:
            raise
        print(
            "[data] label-only pass unavailable ({}); synthesising labels "
            "for the smoke test".format(type(exc).__name__)
        )
        rng = np.random.default_rng(config.SEED)
        labels = rng.integers(0, len(all_class_names), size=128)

    if classes_filter:
        _, keep = resolve_classes_filter(all_class_names, classes_filter)
        remap = {orig: new for new, orig in enumerate(keep)}
        labels = np.array([remap[int(v)] for v in labels if int(v) in remap])
    return labels
