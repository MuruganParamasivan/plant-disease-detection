"""Streamlit demo: upload a leaf photo and get a prediction and a Grad-CAM view.

Run from the repository root::

    streamlit run app/streamlit_app.py

The app reuses ``src/data.preprocess_image`` and ``src/gradcam`` so that
inference matches training exactly and the explanation is produced by the
same code as the evaluation report.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from the repo root, and also the
# Docker image where the code sits at /app.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402
from src.gradcam import compute_heatmap, make_gradcam_model, overlay_heatmap  # noqa: E402

MAX_UPLOAD_MB = 10
DISCLAIMER = (
    "**Educational / advisory only - this is not a diagnosis.** The model was "
    "trained on PlantVillage, whose images are single leaves on uniform lab "
    "backgrounds. On real field photos (soil, sky, multiple leaves, shadows) it "
    "is far less reliable than the test accuracy suggests. Confirm anything that "
    "matters with an agronomist or plant-health service before treating a crop."
)

st.set_page_config(page_title="Plant Disease Detection", page_icon=":seedling:", layout="wide")


def _find_model(models_dir: Path):
    """Newest .keras file in models/ (an explicit MODEL_PATH env var wins)."""
    override = os.environ.get("MODEL_PATH")
    if override:
        path = Path(override)
        return path if path.exists() else None
    candidates = sorted(models_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_classes(model_path_str: str):
    """Load the Keras model, class names and Grad-CAM sub-model once."""
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path_str)
    class_names = json.loads(
        config.CLASS_NAMES_FILE.read_text(encoding="utf-8")
    )["class_names"]
    grad_model, conv_layer_name = make_gradcam_model(model)
    return model, class_names, grad_model, conv_layer_name


def pretty(name: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato - Late blight'."""
    crop, _, condition = name.partition("___")
    return "{} - {}".format(crop.replace("_", " "), condition.replace("_", " ") or "unknown")


def interpretation(class_name: str, confidence: float) -> str:
    """One plain-language line the user can act on (or not)."""
    crop, _, condition = class_name.partition("___")
    crop = crop.replace("_", " ")
    healthy = condition.lower() == "healthy"

    if confidence >= 0.80:
        strength = "confident"
    elif confidence >= 0.50:
        strength = "moderately confident"
    else:
        strength = "not confident"

    if strength == "not confident":
        return (
            "The model is **not confident** about this image ({:.0%}). That usually means "
            "the photo is unlike its training data - a field photo, a non-leaf subject, or "
            "a condition it was never taught. Treat this output as unreliable.".format(confidence)
        )
    if healthy:
        return (
            "The model is **{}** ({:.0%}) that this {} leaf shows no disease it was trained "
            "to recognise. It cannot rule out conditions outside its 38 classes.".format(
                strength, confidence, crop.lower())
        )
    return (
        "The model is **{}** ({:.0%}) that this {} leaf shows signs consistent with "
        "**{}**. Check the Grad-CAM view: if the highlighted region is not on the lesion, "
        "the prediction is probably driven by something irrelevant.".format(
            strength, confidence, crop.lower(), condition.replace("_", " ").lower())
    )


def main() -> None:
    st.title("Plant Leaf Disease Detection")
    st.caption(
        "MobileNetV2 transfer learning on PlantVillage (38 classes) with a Grad-CAM "
        "explanation of where the model looked."
    )
    st.warning(DISCLAIMER)

    models_dir = Path(os.environ.get("MODELS_DIR", config.MODELS_DIR))
    model_path = _find_model(models_dir)
    if model_path is None:
        st.error(
            "No trained model found in `{}`.\n\nTrain one first:\n\n"
            "```\npython -m src.train --model transfer --smoke-test\n```".format(models_dir)
        )
        st.stop()
    if not config.CLASS_NAMES_FILE.exists():
        st.error("`models/class_names.json` is missing - re-run training, which writes it.")
        st.stop()

    try:
        model, class_names, grad_model, conv_layer_name = load_model_and_classes(str(model_path))
    except Exception as exc:
        st.error("Could not load the model: {}: {}".format(type(exc).__name__, exc))
        st.stop()

    with st.sidebar:
        st.subheader("Model")
        st.write("**File:** `{}`".format(model_path.name))
        st.write("**Classes:** {}".format(len(class_names)))
        st.write("**Grad-CAM layer:** `{}`".format(conv_layer_name))
        st.write("**Input:** {}x{}, mobilenet_v2 preprocessing".format(
            config.IMG_SIZE, config.IMG_SIZE))
        show_all = st.checkbox("Show all class probabilities", value=False)
        st.divider()
        st.caption(
            "Known limitation: uniform lab backgrounds in training mean field "
            "photos are out of distribution. Try one deliberately - the "
            "confidence usually collapses, which is the honest behaviour."
        )

    uploaded = st.file_uploader(
        "Upload a leaf image (jpg / jpeg / png)", type=["jpg", "jpeg", "png"]
    )
    if uploaded is None:
        st.info("Upload a single leaf photo to run a prediction.")
        return

    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error("That file is {:.1f} MB; the limit is {} MB.".format(
            uploaded.size / 1024 / 1024, MAX_UPLOAD_MB))
        return

    # Anything can be uploaded, including files that are not really images -
    # decode defensively rather than letting the app crash.
    try:
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = False
        pil_image = Image.open(uploaded)
        pil_image.verify()          # header check
        uploaded.seek(0)
        pil_image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(
            "That file could not be read as an image ({}). Try a standard "
            "JPG or PNG.".format(type(exc).__name__)
        )
        return

    width, height = pil_image.size
    if min(width, height) < 32:
        st.error("Image is only {}x{} px - too small to classify.".format(width, height))
        return
    if min(width, height) < 100:
        st.warning(
            "Low resolution ({}x{} px). It will be upscaled to {}px, so the "
            "prediction is less trustworthy.".format(width, height, config.IMG_SIZE)
        )

    import tensorflow as tf

    from src.data import preprocess_image

    raw = np.array(pil_image)
    preprocessed = preprocess_image(raw, config.IMG_SIZE).numpy()[None, ...]
    probabilities = model.predict(preprocessed, verbose=0)[0]

    if probabilities.shape[0] != len(class_names):
        st.error(
            "Model outputs {} classes but class_names.json lists {}. The model "
            "and the class list are out of sync - re-run training.".format(
                probabilities.shape[0], len(class_names))
        )
        return

    order = np.argsort(probabilities)[::-1]
    top_idx = int(order[0])
    top_conf = float(probabilities[top_idx])

    display = np.array(
        tf.image.resize(tf.cast(raw, tf.float32), (config.IMG_SIZE, config.IMG_SIZE))
    ).astype(np.uint8)

    try:
        heatmap = compute_heatmap(grad_model, preprocessed, top_idx)
        overlay = overlay_heatmap(display, heatmap)
        gradcam_error = None
    except Exception as exc:  # never let the explanation break the prediction
        overlay, gradcam_error = None, "{}: {}".format(type(exc).__name__, exc)

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded image")
        st.image(pil_image, use_container_width=True)
    with right:
        st.subheader("Grad-CAM")
        if overlay is not None:
            st.image(overlay, use_container_width=True)
            st.caption(
                "Red = regions that most increased the predicted class score "
                "(layer `{}`).".format(conv_layer_name)
            )
        else:
            st.info("Grad-CAM unavailable for this model ({}).".format(gradcam_error))

    st.divider()
    st.subheader("Prediction")
    metric_col, text_col = st.columns([1, 2])
    with metric_col:
        st.metric("Predicted class", pretty(class_names[top_idx]))
        st.metric("Confidence", "{:.1%}".format(top_conf))
    with text_col:
        st.markdown(interpretation(class_names[top_idx], top_conf))
        if top_conf < 0.50:
            st.warning(
                "Low confidence - the softmax output is spread across several "
                "classes. Do not act on this."
            )

    st.subheader("Top 3 classes")
    import pandas as pd

    top3 = pd.DataFrame(
        {
            "class": [pretty(class_names[int(i)]) for i in order[:3]],
            "probability": [float(probabilities[int(i)]) for i in order[:3]],
        }
    ).set_index("class")
    st.bar_chart(top3, horizontal=True)

    if show_all:
        st.subheader("All class probabilities")
        st.dataframe(
            pd.DataFrame(
                {
                    "class": [pretty(c) for c in class_names],
                    "probability": [float(p) for p in probabilities],
                }
            ).sort_values("probability", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
