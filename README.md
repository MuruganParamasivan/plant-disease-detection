# Plant Disease Detection using CNN and Transfer Learning

AI/ML Specialisation Capstone — solo build, 3-week timeline.

An image classifier that takes a plant leaf photo and predicts whether it is healthy
or affected by disease, with a confidence score and a Grad-CAM explanation, deployed
as a Streamlit / FastAPI demo.

---

## Scope decision (read first)

- **Primary path:** full **38-class** PlantVillage classification using **MobileNetV2
  transfer learning**, with a small **baseline CNN** for contrast.
- **Develop on a subset, train on the full set.** Debug the pipeline locally in VS Code
  on a few classes; run the full training on **Google Colab (free GPU)**. Training 38
  classes from scratch on a CPU is not practical — transfer learning + GPU is the path.
- **Fallback (if time/compute is tight):** switch to **tomato-only (~10 classes)**. The
  code is written so this is a one-line change (see `notebooks/01`). Document the exact
  classes retained if you do this.

## Dataset

- **PlantVillage** via TensorFlow Datasets (`plant_village`): ~54k leaf images, 38
  classes across 14 crop species (healthy + diseased). Single `train` split — you create
  train/val/test yourself (the notebook does this).
- Source: https://www.tensorflow.org/datasets/catalog/plant_village
- **Known limitation:** images are on uniform/controlled lab backgrounds, so a model
  trained on this data will *not* generalise to real field photos. State this plainly in
  your report — it is the main honest caveat and evaluators look for it.

## Repository structure

```
plant-disease-detection/
├── README.md
├── BUILD_PLAN.md              # phased plan + Claude Code prompts for the rest
├── requirements.txt
├── notebooks/
│   └── 01_data_access_and_eda.ipynb   # START HERE (runnable)
├── src/                       # built in Phase 2–3 via Claude Code
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── app/                       # built in Phase 4
│   ├── streamlit_app.py
│   └── Dockerfile
├── models/                    # saved model artifacts (gitignored)
└── reports/                   # metrics, plots, confusion matrix, Grad-CAM
```

## Setup

**Local (VS Code) — for development on a subset:**
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Colab — for full training (recommended):**
Upload `notebooks/01_data_access_and_eda.ipynb`, set Runtime → Change runtime type → GPU,
run the install cell at the top. TFDS downloads PlantVillage (~1 GB) automatically.

## 3-week plan (solo)

| Week | Focus | Deliverables |
|---|---|---|
| 1 | Repo, data access, EDA, preprocessing, train/val/test split | This notebook, data pipeline |
| 2 | Baseline CNN, MobileNetV2 transfer learning, augmentation, tuning, experiment logs | Baseline + advanced model notebooks, saved models |
| 3 | Evaluation (confusion matrix, per-class P/R, Grad-CAM), Streamlit + Docker, report, deck | Eval report, deployed prototype, README, presentation |

## Deliverables checklist (from the problem statement)

- [x] Problem statement & scope document (your approved PDF + this README)
- [ ] Data access + preprocessing notebook  ← Phase 1 (this notebook)
- [ ] EDA notebook  ← Phase 1 (this notebook)
- [ ] Baseline + advanced model notebooks  ← Phase 2
- [ ] Evaluation report (metrics, plots, Grad-CAM, sample errors)  ← Phase 3
- [ ] Deployment prototype (Streamlit/FastAPI + Docker)  ← Phase 4
- [ ] Final presentation deck  ← Phase 3–4
- [ ] GitHub repo with README + reproducible instructions  ← ongoing

See **BUILD_PLAN.md** for the phase-by-phase build, including ready-to-paste prompts for
Claude Code to generate the model, training, evaluation, and deployment files.
