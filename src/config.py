"""Project-wide constants and reproducibility helpers.

Every module imports its constants from here so that the data pipeline, the
training script, the evaluation script and the Streamlit app agree on image
size, preprocessing and where artifacts live.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# --- Core hyper-parameters (locked) -----------------------------------------
IMG_SIZE = 224          # MobileNetV2's native input resolution
BATCH_SIZE = 32
SEED = 42

# --- Dataset ----------------------------------------------------------------
# PlantVillage ships as a single TFDS 'train' split (~54k images, 38 classes),
# so train/val/test are carved out here with TFDS subsplit syntax.
DATASET_NAME = "plant_village"
TRAIN_SPLIT = "train[:80%]"
VAL_SPLIT = "train[80%:90%]"
TEST_SPLIT = "train[90%:]"

# Smoke-test splits: 2% of the data, then split 80/10/10 *within* that 2%.
# Used by --smoke-test so the whole pipeline can be exercised on a CPU in
# a couple of minutes.
SMOKE_FRACTION = 2  # percent
SMOKE_TRAIN_SPLIT = "train[:1.6%]"
SMOKE_VAL_SPLIT = "train[1.6%:1.8%]"
SMOKE_TEST_SPLIT = "train[1.8%:2%]"

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"

# Where TFDS keeps its prepared data. Overridable so Colab can point at Drive.
TFDS_DATA_DIR = os.environ.get("TFDS_DATA_DIR") or None

AUTOTUNE_MSG = (
    "Note: resized float32 tensors are NOT cached - 54k images at 224x224x3 "
    "would not fit in RAM. The pipeline uses .prefetch(AUTOTUNE) only."
)


def ensure_dirs() -> None:
    """Create the artifact directories if they do not exist yet."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Seed python, numpy and tensorflow for reproducible runs.

    Note: full bit-for-bit determinism on GPU also needs
    TF_DETERMINISTIC_OPS=1, which costs throughput; we seed everything that
    is cheap to seed and record the seed in run_config.json instead.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# The 38 PlantVillage class names, in TFDS label order. Only used as a
# fallback when TFDS metadata is unavailable (e.g. offline smoke tests);
# the authoritative list is always written to models/class_names.json.
PLANT_VILLAGE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___healthy',
    'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___healthy',
    'Corn___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___healthy',
    'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
]
