# Build Plan — Plant Disease Detection

Phase 1 is done (the EDA + data notebook). Phases 2–4 are the heavier multi-file work —
each has a ready-to-paste prompt for **Claude Code** so you can delegate the
implementation and keep your review time for architecture and results.

---

## Phase 1 — Data + EDA  ✅ (notebooks/01_data_access_and_eda.ipynb)

Loads PlantVillage, inspects the 38 classes, plots the class distribution, shows sample
images, and builds resized/augmented `train/val/test` tf.data pipelines. Run this end to
end (on Colab GPU) before Phase 2. Confirm the class count and that the pipelines yield
correct `(image, label)` batches.

---

## Phase 2 — Modeling

Goal: a baseline CNN and a MobileNetV2 transfer-learning model, both trained and saved.

**Claude Code prompt:**
> Create `src/model.py` and `src/train.py` for a PlantVillage classifier.
> `model.py`: two builders — (1) `build_baseline_cnn(num_classes, img_size=224)` a small
> Conv/Pool stack with dropout; (2) `build_transfer_model(num_classes, img_size=224,
> backbone="MobileNetV2")` using `tf.keras.applications` with `include_top=False`,
> frozen base, GlobalAveragePooling, dropout, and a softmax head. Add a `fine_tune`
> option that unfreezes the top N layers at a low learning rate.
> `train.py`: reuse the tf.data pipeline from the notebook (factor it into `src/data.py`),
> train with `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`; support optional
> class weights; save the best model to `models/` and the training history to `reports/`.
> Make batch size, epochs, backbone, learning rate, and image size CLI args.

After this: run baseline, then transfer model, then a short fine-tune pass. Log
val-accuracy and val-loss curves for each into `reports/`.

---

## Phase 3 — Evaluation + explainability

Goal: honest, class-aware evaluation and Grad-CAM overlays.

**Claude Code prompt:**
> Create `src/evaluate.py`. Load a saved model and the held-out test set, then produce:
> overall accuracy and macro F1; a scikit-learn `classification_report` (per-class
> precision/recall/F1); a confusion matrix saved as a figure; the worst-confused class
> pairs; a grid of sample correct and incorrect predictions; and Grad-CAM overlays for a
> handful of test images (implement Grad-CAM directly against the last conv layer of the
> backbone). Save everything to `reports/`. Print a short summary table.

Interpretation notes for your report:
- Don't headline accuracy alone — show per-class recall; some diseases look alike and
  will be confused. Use the confusion matrix + Grad-CAM to explain *why*.
- High val/test accuracy is expected on PlantVillage because backgrounds are uniform.
  Note explicitly that this does **not** imply field performance.

---

## Phase 4 — Deployment

Goal: a demo that loads the trained model and predicts on an uploaded leaf image.

**Claude Code prompt:**
> Create `app/streamlit_app.py` and `app/Dockerfile`. The Streamlit app loads the saved
> model once (cached), accepts a leaf-image upload, runs the same preprocessing as
> training, and displays the predicted class, a confidence score, the top-3 classes, and a
> Grad-CAM overlay. Include a short plain-language interpretation and a visible disclaimer
> that this is educational/advisory, not a diagnosis. The Dockerfile should package the app
> + model so it runs with `docker run` without retraining. Add run instructions to README.

Demo must include at least 3 sample predictions and 1 known failure case (e.g. a real
field photo with a cluttered background — a good way to demonstrate the generalisation
limit on purpose).

---

## Cross-cutting

- **Experiment tracking:** a simple CSV/JSON per run (config + final metrics) in
  `reports/` is enough for a solo capstone; no need for MLflow unless you want it.
- **Reproducibility:** set seeds, pin versions (requirements.txt), commit notebooks with
  outputs cleared.
- **Presentation:** build the 8-slide deck (see the problem statement) as you go — pull
  figures straight from `reports/`.
