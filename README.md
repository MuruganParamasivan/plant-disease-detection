# Plant Disease Detection using CNN and Transfer Learning

AI/ML Specialisation Capstone — solo build, 3-week timeline.

An image classifier that takes a plant-leaf photo and predicts which of the 38
PlantVillage classes it belongs to, with a confidence score and a Grad-CAM
explanation, deployed as a Streamlit demo (Dockerised).
---
## Scope

- **Primary path:** full **38-class** PlantVillage classification using
  **MobileNetV2 transfer learning**, with a small **baseline CNN** for contrast.
- **Develop on a subset, train on the full set.** The pipeline is validated
  locally on CPU with `--smoke-test`; full training runs on **Google Colab
  (free GPU)**. Training 38 classes from scratch on a CPU is not practical.
- **Fallback (if time/compute is tight):** tomato-only (~10 classes) via
  `--classes-filter tomato`, which is a one-flag change on every script.

## Dataset

- **PlantVillage** via TensorFlow Datasets (`plant_village`): ~54k leaf images,
  38 classes across 14 crop species. A single `train` split, so train/val/test
  are carved out with TFDS subsplit syntax:

  | split | subsplit | share |
  |---|---|---|
  | train | `train[:80%]` | 80% |
  | validation | `train[80%:90%]` | 10% |
  | test | `train[90%:]` | 10% |

- Source: https://www.tensorflow.org/datasets/catalog/plant_village
- Integer labels throughout → `SparseCategoricalCrossentropy`.

## Preprocessing (identical in training, evaluation and the app)

1. Resize to **224×224**, batch **32**.
2. **Train split only:** `RandomFlip("horizontal")`, `RandomRotation(0.1)`,
   `RandomZoom(0.1)` — applied after resize, before preprocessing.
3. `tf.keras.applications.mobilenet_v2.preprocess_input` → float32 in [-1, 1].
4. `.prefetch(AUTOTUNE)`; **no `.cache()`** — 54k images at 224×224×3 float32 is
   ~32 GB and OOMs. Only the train split is shuffled, so evaluation artifacts
   stay reproducible.

## Repository structure

```
plant-disease-detection/
├── README.md                  # this file
├── BUILD_PLAN.md              # original phased plan
├── requirements.txt
├── src/
│   ├── config.py              # constants, paths, set_seeds()
│   ├── data.py                # tf.data pipeline, subsplits, class filtering
│   ├── model.py               # baseline CNN + transfer model builders
│   ├── gradcam.py             # Grad-CAM (shared by evaluate.py and the app)
│   ├── train.py               # argparse training + fine-tuning
│   └── evaluate.py            # metrics, confusion matrix, Grad-CAM report
├── app/
│   ├── streamlit_app.py       # upload → prediction + Grad-CAM demo
│   └── Dockerfile             # runs the demo without retraining
├── models/                    # saved .keras models + class_names.json (gitignored)
└── reports/                   # metrics, curves, confusion matrix, Grad-CAM (gitignored)
```

---

## 1. Local setup and subset development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Prove the whole pipeline on CPU in a couple of minutes — no GPU, no dataset
download:

```bash
python -m src.train --model transfer --smoke-test
python -m src.evaluate --model-path models/<run_name>.keras --smoke-test
```

`--smoke-test` uses `train[:2%]` split three ways. **If PlantVillage has not been
prepared locally, it falls back to synthetic random images and says so loudly.**
That is deliberate: TFDS must download and prepare the full ~1 GB archive before
it can serve *any* subsplit, so a 2% split does not by itself avoid the download.
A smoke run proves shapes, the training loop, checkpointing, evaluation and
Grad-CAM all execute — **its metrics are noise by construction.**

To develop against real images locally, accept the one-off download:

```bash
python -m src.train --model transfer --smoke-test --download --epochs 2
```

### Useful flags

| flag | effect |
|---|---|
| `--model {baseline,transfer}` | which architecture to train |
| `--backbone MobileNetV2` | also `EfficientNetB0`, `ResNet50` |
| `--epochs N` / `--fine-tune-epochs N` | head training, then unfreezing |
| `--fine-tune-layers N` | how many top backbone layers to unfreeze (default 30) |
| `--lr 1e-3` | Adam learning rate (fine-tuning uses `lr/10`) |
| `--classes-filter tomato` | reduce to one crop; labels remapped to 0..k-1 |
| `--class-weights` | sklearn `balanced` weights for the imbalanced classes |
| `--img-size` / `--batch-size` / `--seed` | as named |
| `--data-dir` | where TFDS keeps prepared data (e.g. a Drive path) |

---

## 2. Full training on Google Colab (GPU)

Runtime → Change runtime type → **T4 GPU**, then in a notebook cell:

```python
!git clone https://github.com/<your-user>/plant-disease-detection.git
%cd plant-disease-detection
!pip install -q "tensorflow-datasets>=4.9" scikit-learn seaborn streamlit
# Colab already ships TensorFlow with GPU support - do not reinstall it.

import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))
```

Train the two models (the first run downloads ~1 GB of PlantVillage):

```python
# Baseline CNN - the contrast case
!python -m src.train --model baseline --epochs 15 --lr 1e-3

# MobileNetV2: frozen head, then fine-tune the top 30 layers at lr/10
!python -m src.train --model transfer --backbone MobileNetV2 \
    --epochs 15 --fine-tune-epochs 5 --fine-tune-layers 30 --lr 1e-3
```

Optional variants:

```python
# class-imbalance-aware run
!python -m src.train --model transfer --epochs 15 --fine-tune-epochs 5 --class-weights

# tomato-only fallback (~10 classes)
!python -m src.train --model transfer --epochs 15 --classes-filter tomato
```

Keep the artifacts (Colab VMs are wiped):

```python
from google.colab import drive; drive.mount('/content/drive')
!cp -r models reports /content/drive/MyDrive/plant-disease-detection/
```

To avoid re-downloading the dataset on every session, point TFDS at Drive:
`--data-dir /content/drive/MyDrive/tfds`.

---

## 3. Evaluation

```bash
python -m src.evaluate --model-path models/<run_name>.keras --num-gradcam 12
```

Writes to `reports/`:

- `<run>_classification_report.txt` / `.csv` — per-class precision/recall/F1
- `<run>_confusion_matrix.png` — row-normalised by true class
- `<run>_confused_pairs.csv` — worst confusions, ranked by share of true class
- `<run>_samples_correct.png` / `<run>_samples_incorrect.png`
- `<run>_gradcam.png` — overlays, mistakes shown first
- `<run>_metrics.json` — accuracy, macro-F1, per-class stats, worst recalls

The console summary deliberately leads with the **lowest-recall classes** and the
most-confused pairs rather than headline accuracy.

## 4. Running the demo app

```bash
streamlit run app/streamlit_app.py
```

It loads the newest `.keras` file in `models/` (override with `MODEL_PATH`), plus
`models/class_names.json`, and shows the predicted class, confidence, a top-3 bar
chart, a Grad-CAM overlay, a plain-language interpretation and the disclaimer.

## 5. Docker

Build from the **repository root** (the context needs `src/` and `models/`):

```bash
docker build -f app/Dockerfile -t plant-disease-app .
docker run --rm -p 8501:8501 plant-disease-app
# open http://localhost:8501
```

The image bundles a trained model, so it serves predictions without retraining.
Train (or copy in) a model first — `models/` is gitignored. The image is large
(~2 GB) because it installs full TensorFlow; swapping `tensorflow` for
`tensorflow-cpu` in a demo-only requirements file roughly halves it.

> Not yet built end-to-end on this machine: Docker is installed but its daemon
> was not running during development, so the Dockerfile is unverified by a real
> build. Everything else in this README was executed.

---

## Reproducibility

- `set_seeds()` seeds python, numpy and TensorFlow from `--seed` (default 42).
- `models/class_names.json` records the class list and any filtered-label
  mapping, so evaluation and the app never depend on TFDS metadata and label
  order cannot silently drift.
- Every run writes `reports/<run>_run_config.json`: all CLI args, effective
  epochs, class names, final metrics, wall-clock time, and the Python /
  TensorFlow / platform / GPU it ran on.
- Metrics reported at the end of training come from the **reloaded checkpoint**,
  so they describe the file `evaluate.py` will read.

## Limitations

1. **PlantVillage is not the field.** Every image is a single detached leaf on a
   uniform lab background. A model trained on it can reach very high test
   accuracy by learning leaf-and-background statistics that do not exist in a
   real photograph of a plant in soil, under variable light, with several
   overlapping leaves. **High test accuracy here does not imply field
   performance** — treat the two as unrelated until tested on field data.
2. **Report per-class recall, not accuracy alone.** The classes are imbalanced;
   an aggregate number hides classes the model rarely gets right. Use
   `<run>_classification_report.csv` and the lowest-recall list in the console
   summary. `--class-weights` is available when imbalance hurts the rare classes.
3. **Some diseases look alike.** Early blight vs late blight, and several tomato
   leaf-spot conditions, share visual signatures; expect them to be confused.
   The confusion matrix and `confused_pairs.csv` show *which* pairs, and
   Grad-CAM shows *what* the model keyed on when it got them wrong.
4. **Grad-CAM is a sanity check, not proof.** It shows which regions increased
   the predicted class score — not that the model's reasoning is correct. A
   heatmap centred on background rather than lesion is a red flag worth
   reporting.
5. **38 fixed classes, no "unknown".** Softmax always produces an answer. On a
   non-leaf image, or a disease outside the 38 classes, the model still returns
   a class; the app surfaces low confidence as a warning, but there is no
   genuine out-of-distribution detector.
6. **Educational / advisory output only.** This is not a clinical or agronomic
   diagnosis, and should not be the sole basis for treating a crop.

## Deliverables checklist

- [x] Problem statement & scope document (approved PDF + this README)
- [x] Data access + preprocessing (`src/data.py`)
- [x] Baseline + advanced models (`src/model.py`, `src/train.py`)
- [x] Evaluation report with Grad-CAM (`src/evaluate.py` → `reports/`)
- [x] Deployment prototype (`app/streamlit_app.py` + `app/Dockerfile`)
- [ ] Full 38-class training run on Colab
- [ ] Final presentation deck (pull figures from `reports/`)
- [x] GitHub repo with README + reproducible instructions

See **BUILD_PLAN.md** for the original phase-by-phase plan.
