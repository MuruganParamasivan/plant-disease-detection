"""Class-aware evaluation of a saved model, with Grad-CAM explanations.

Examples
--------
    python -m src.evaluate --model-path models/transfer_mobilenetv2_...keras --smoke-test
    python -m src.evaluate --model-path models/best.keras --num-gradcam 12

Produces, under ``reports/``:
  * ``<run>_classification_report.txt`` / ``.csv``  - per-class P/R/F1
  * ``<run>_confusion_matrix.png``                  - normalised by true class
  * ``<run>_confused_pairs.csv``                    - the worst class confusions
  * ``<run>_samples_correct.png`` / ``_samples_incorrect.png``
  * ``<run>_gradcam.png``                           - overlay grid
  * ``<run>_metrics.json``                          - headline numbers

Read the per-class recall, not the headline accuracy: PlantVillage is
recorded on uniform lab backgrounds, so overall accuracy flatters the model
and says nothing about field performance.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone

import numpy as np

from src import config


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a saved PlantVillage classifier on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="path to a .keras model")
    p.add_argument("--classes-filter", default=None,
                   help="must match the filter used for training")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--download", action="store_true",
                   help="allow TFDS to download the dataset in smoke mode")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--num-gradcam", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=9,
                   help="how many correct / incorrect examples to show")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    return p.parse_args(argv)


def _short(name: str, width: int = 28) -> str:
    """Compact a PlantVillage class name for axis labels."""
    name = name.replace("___", " / ").replace("_", " ")
    return name if len(name) <= width else name[: width - 3] + "..."


def plot_confusion_matrix(cm, class_names, out_path, normalise=True):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float)
    if normalise:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

    n = len(class_names)
    size = max(6.0, min(0.42 * n + 3, 22))
    fig, ax = plt.subplots(figsize=(size, size * 0.92))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    ax.set_xticks(range(n), [_short(c) for c in class_names], rotation=90, fontsize=7)
    ax.set_yticks(range(n), [_short(c) for c in class_names], fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        "Confusion matrix ({} by true class)".format("row-normalised" if normalise else "counts")
    )
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8)
    if n <= 12:  # only readable on the reduced (e.g. tomato-only) problem
        for i in range(n):
            for j in range(n):
                if data[i, j] > 0.005:
                    ax.text(j, i, "{:.2f}".format(data[i, j]), ha="center", va="center",
                            fontsize=6, color="white" if data[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[eval] confusion matrix -> {}".format(out_path))


def most_confused_pairs(cm, class_names, top_k=15):
    """Off-diagonal confusions, ranked by share of the true class."""
    rows = []
    totals = cm.sum(axis=1)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                rows.append(
                    {
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                        "share_of_true_class": round(float(cm[i, j]) / max(int(totals[i]), 1), 4),
                    }
                )
    rows.sort(key=lambda r: (-r["share_of_true_class"], -r["count"]))
    return rows[:top_k]


def plot_sample_grid(images, titles, colours, out_path, title, cols=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not len(images):
        print("[eval] no samples for '{}' - skipping".format(title))
        return
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, image, text, colour in zip(axes, images, titles, colours):
        ax.imshow(image)
        ax.set_title(text, fontsize=7, color=colour, wrap=True)
    fig.suptitle(title)
    # Two-line titles need explicit room, otherwise they land on the image above.
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("[eval] {} -> {}".format(title.lower(), out_path))


def main(argv=None) -> int:
    args = parse_args(argv)
    config.ensure_dirs()
    config.set_seeds(args.seed)

    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    from src import data as data_mod
    from src import gradcam as gradcam_mod

    reports_dir = pathlib.Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = pathlib.Path(args.model_path)
    if not model_path.exists():
        raise SystemExit("Model not found: {}".format(model_path))
    run_name = model_path.stem

    print("[eval] loading {}".format(model_path))
    model = tf.keras.models.load_model(model_path)

    # class_names.json is written by training, so evaluation never needs TFDS
    # metadata and the label order can never silently drift.
    class_names = data_mod.load_class_names()
    num_model_outputs = int(model.outputs[0].shape[-1])
    if num_model_outputs != len(class_names):
        raise SystemExit(
            "Model has {} outputs but models/class_names.json lists {} classes. "
            "Re-run training, or pass the --classes-filter used for this model.".format(
                num_model_outputs, len(class_names)
            )
        )

    _, _, test_ds, _ = data_mod.load_datasets(
        classes_filter=args.classes_filter,
        smoke=args.smoke_test,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        data_dir=args.data_dir,
        download=True if args.download else None,
        persist=False,  # never overwrite the training-time class list
    )

    # ---- predictions -----------------------------------------------------
    y_true, y_pred, y_conf = [], [], []
    kept_images, kept_meta = [], []
    keep_budget = max(args.num_gradcam, args.num_samples * 2) + 16

    for batch_images, batch_labels in test_ds:
        probs = model.predict(batch_images, verbose=0)
        preds = probs.argmax(axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_conf.extend(probs.max(axis=1).tolist())
        if len(kept_images) < keep_budget:
            room = keep_budget - len(kept_images)
            kept_images.extend(batch_images.numpy()[:room])
            kept_meta.extend(
                list(zip(batch_labels.numpy()[:room].tolist(), preds[:room].tolist(),
                         probs.max(axis=1)[:room].tolist()))
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)
    if y_true.size == 0:
        raise SystemExit("Test split is empty - nothing to evaluate.")

    labels_index = list(range(len(class_names)))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels_index,
                              zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels_index,
                                 zero_division=0))

    # ---- classification report (txt + csv) -------------------------------
    report_txt = classification_report(
        y_true, y_pred, labels=labels_index, target_names=class_names, zero_division=0, digits=4
    )
    txt_path = reports_dir / "{}_classification_report.txt".format(run_name)
    txt_path.write_text(
        "Model: {}\nTest images: {}\nAccuracy: {:.4f}   Macro-F1: {:.4f}   "
        "Weighted-F1: {:.4f}\n\n{}\n".format(
            model_path.name, len(y_true), accuracy, macro_f1, weighted_f1, report_txt
        ),
        encoding="utf-8",
    )
    print("[eval] classification report -> {}".format(txt_path))

    import pandas as pd

    report_dict = classification_report(
        y_true, y_pred, labels=labels_index, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    csv_path = reports_dir / "{}_classification_report.csv".format(run_name)
    report_df.to_csv(csv_path, index_label="class")
    print("[eval] per-class csv -> {}".format(csv_path))

    # ---- confusion matrix + worst confusions -----------------------------
    cm = confusion_matrix(y_true, y_pred, labels=labels_index)
    plot_confusion_matrix(cm, class_names,
                          reports_dir / "{}_confusion_matrix.png".format(run_name))
    pairs = most_confused_pairs(cm, class_names)
    pairs_path = reports_dir / "{}_confused_pairs.csv".format(run_name)
    pd.DataFrame(pairs).to_csv(pairs_path, index=False)
    print("[eval] most-confused pairs -> {}".format(pairs_path))

    # ---- sample grids ----------------------------------------------------
    correct_idx = [i for i, (t, p, _) in enumerate(kept_meta) if t == p][: args.num_samples]
    wrong_idx = [i for i, (t, p, _) in enumerate(kept_meta) if t != p][: args.num_samples]
    for idx_list, label, colour in (
        (correct_idx, "Correct predictions", "green"),
        (wrong_idx, "Incorrect predictions", "firebrick"),
    ):
        images = [gradcam_mod.deprocess(kept_images[i]) for i in idx_list]
        titles = [
            "true: {}\npred: {} ({:.0%})".format(
                _short(class_names[kept_meta[i][0]], 24),
                _short(class_names[kept_meta[i][1]], 24),
                kept_meta[i][2],
            )
            for i in idx_list
        ]
        plot_sample_grid(
            images, titles, [colour] * len(images),
            reports_dir / "{}_samples_{}.png".format(
                run_name, "correct" if colour == "green" else "incorrect"),
            label,
        )

    # ---- Grad-CAM --------------------------------------------------------
    grad_model, conv_layer_name = gradcam_mod.make_gradcam_model(model)
    print("[eval] Grad-CAM target layer: {}".format(conv_layer_name))

    n_cam = min(args.num_gradcam, len(kept_images))
    # Prefer a mix: show mistakes first (that is where an explanation earns
    # its keep), then correct ones.
    cam_order = (wrong_idx + correct_idx + list(range(len(kept_images))))[:n_cam]
    seen, cam_idx = set(), []
    for i in cam_order:
        if i not in seen:
            seen.add(i)
            cam_idx.append(i)

    cam_images, cam_titles, cam_colours = [], [], []
    for i in cam_idx:
        preprocessed = kept_images[i][None, ...]
        true_label, pred_label, conf = kept_meta[i]
        heatmap = gradcam_mod.compute_heatmap(grad_model, preprocessed, pred_label)
        display = gradcam_mod.deprocess(kept_images[i])
        cam_images.append(gradcam_mod.overlay_heatmap(display, heatmap))
        cam_titles.append(
            "true: {}\npred: {} ({:.0%})".format(
                _short(class_names[true_label], 24), _short(class_names[pred_label], 24), conf)
        )
        cam_colours.append("green" if true_label == pred_label else "firebrick")

    plot_sample_grid(
        cam_images, cam_titles, cam_colours,
        reports_dir / "{}_gradcam.png".format(run_name),
        "Grad-CAM ({}): red = most influential region".format(conv_layer_name),
    )

    # ---- metrics summary -------------------------------------------------
    per_class = {
        name: {
            "precision": float(report_dict[name]["precision"]),
            "recall": float(report_dict[name]["recall"]),
            "f1": float(report_dict[name]["f1-score"]),
            "support": int(report_dict[name]["support"]),
        }
        for name in class_names
        if name in report_dict
    }
    supported = {k: v for k, v in per_class.items() if v["support"] > 0}
    worst = sorted(supported.items(), key=lambda kv: kv[1]["recall"])[:5]

    metrics = {
        "model_path": str(model_path),
        "smoke_test": args.smoke_test,
        "test_images": int(len(y_true)),
        "num_classes": len(class_names),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_confidence": float(y_conf.mean()),
        "mean_confidence_when_wrong": (
            float(y_conf[y_true != y_pred].mean()) if (y_true != y_pred).any() else None
        ),
        "per_class": per_class,
        "worst_recall_classes": [
            {"class": k, "recall": v["recall"], "support": v["support"]} for k, v in worst
        ],
        "most_confused_pairs": pairs[:10],
        "gradcam_layer": conv_layer_name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveat": (
            "PlantVillage images sit on uniform lab backgrounds. High test "
            "accuracy here does not imply field performance; read per-class "
            "recall and the confusion matrix, not the headline number."
        ),
    }
    metrics_path = reports_dir / "{}_metrics.json".format(run_name)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[eval] metrics -> {}".format(metrics_path))

    # ---- console summary -------------------------------------------------
    print("\n" + "=" * 66)
    print("SUMMARY  {}".format(run_name))
    print("=" * 66)
    print("{:<26}{}".format("test images", len(y_true)))
    print("{:<26}{}".format("classes", len(class_names)))
    print("{:<26}{:.4f}".format("accuracy", accuracy))
    print("{:<26}{:.4f}".format("macro F1", macro_f1))
    print("{:<26}{:.4f}".format("weighted F1", weighted_f1))
    print("{:<26}{:.4f}".format("mean confidence", float(y_conf.mean())))
    print("-" * 66)
    print("Lowest-recall classes (the honest headline):")
    for name, stats in worst:
        print("  {:<40} recall={:.3f}  n={}".format(
            _short(name, 38), stats["recall"], stats["support"]))
    print("-" * 66)
    print("Most-confused pairs (true -> predicted):")
    for row in pairs[:5]:
        print("  {:<28} -> {:<28} {:.1%} ({})".format(
            _short(row["true_class"], 26), _short(row["predicted_class"], 26),
            row["share_of_true_class"], row["count"]))
    print("=" * 66)
    if args.smoke_test:
        print("SMOKE TEST: these numbers are noise by design - the run only "
              "proves that evaluation and Grad-CAM execute end to end.")
    else:
        print(metrics["caveat"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
