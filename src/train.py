"""Train the baseline CNN or the MobileNetV2 transfer model.

Examples
--------
Smoke test (CPU, minutes, no download)::

    python -m src.train --model transfer --smoke-test

Full 38-class run (Colab GPU)::

    python -m src.train --model transfer --epochs 15 --fine-tune-epochs 5

Every run writes three artifacts: the best checkpoint to ``models/``, and
``<run>_history.json`` + ``<run>_curves.png`` + ``<run>_run_config.json`` to
``reports/`` so a result can always be traced back to the arguments that
produced it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time
from datetime import datetime, timezone

from src import config


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a PlantVillage leaf-disease classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    p.add_argument("--backbone", default="MobileNetV2",
                   help="transfer backbone (MobileNetV2 | EfficientNetB0 | ResNet50)")
    p.add_argument("--epochs", type=int, default=15, help="head-training epochs")
    p.add_argument("--fine-tune-epochs", type=int, default=0,
                   help="extra epochs with the top of the backbone unfrozen (transfer only)")
    p.add_argument("--fine-tune-layers", type=int, default=30,
                   help="how many top backbone layers to unfreeze")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--classes-filter", default=None,
                   help="keep only classes starting with this prefix, e.g. 'tomato'")
    p.add_argument("--class-weights", action="store_true",
                   help="weight the loss by inverse class frequency (sklearn 'balanced')")
    p.add_argument("--smoke-test", action="store_true",
                   help="tiny subset + 1 epoch: proves the pipeline runs on a CPU")
    p.add_argument("--download", action="store_true",
                   help="allow TFDS to download the ~1 GB dataset in smoke mode")
    p.add_argument("--data-dir", default=None, help="TFDS data dir (e.g. a Drive path)")
    p.add_argument("--output-dir", default=str(config.MODELS_DIR))
    p.add_argument("--run-name", default=None, help="defaults to <model>_<backbone>_<timestamp>")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args(argv)


def _plot_history(history_dict, out_path, title):
    """Save accuracy/loss curves; marks where fine-tuning started, if it did."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (key, label) in zip(axes, [("accuracy", "Accuracy"), ("loss", "Loss")]):
        if key in history_dict:
            ax.plot(history_dict[key], label="train", marker="o", ms=3)
        if "val_" + key in history_dict:
            ax.plot(history_dict["val_" + key], label="validation", marker="o", ms=3)
        split_at = history_dict.get("fine_tune_start_epoch")
        if split_at:
            ax.axvline(split_at - 0.5, color="grey", ls="--", lw=1)
            ax.text(split_at - 0.4, ax.get_ylim()[0], " fine-tune", fontsize=8, color="grey")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("[train] curves -> {}".format(out_path))


def _merge_history(first, second):
    """Concatenate the head-training and fine-tuning histories."""
    merged = {k: list(v) for k, v in first.items()}
    if second:
        for k, v in second.items():
            merged.setdefault(k, [])
            merged[k].extend(list(v))
        merged["fine_tune_start_epoch"] = len(next(iter(first.values())))
    return merged


def main(argv=None) -> int:
    args = parse_args(argv)

    config.ensure_dirs()
    config.set_seeds(args.seed)

    import tensorflow as tf

    from src import data as data_mod
    from src import model as model_mod

    # Smoke mode: one short epoch on a tiny subset, just to prove the wiring.
    epochs = 1 if args.smoke_test else args.epochs
    fine_tune_epochs = (
        1 if (args.smoke_test and args.model == "transfer") else args.fine_tune_epochs
    )
    img_size = args.img_size
    batch_size = args.batch_size

    run_name = args.run_name or "{}_{}_{}".format(
        args.model,
        args.backbone.lower() if args.model == "transfer" else "cnn",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else config.MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "{}.keras".format(run_name)

    print("=" * 72)
    print("[train] run: {}".format(run_name))
    print("[train] {}".format("SMOKE TEST (tiny subset, 1 epoch)" if args.smoke_test else "full run"))
    print("=" * 72)

    train_ds, val_ds, test_ds, class_names = data_mod.load_datasets(
        classes_filter=args.classes_filter,
        smoke=args.smoke_test,
        img_size=img_size,
        batch_size=batch_size,
        seed=args.seed,
        data_dir=args.data_dir,
        download=True if args.download else None,
    )
    num_classes = len(class_names)

    model = model_mod.get_model(
        args.model, num_classes, img_size=img_size, backbone=args.backbone
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda s: print("[model] " + s), line_length=100)

    class_weight = None
    if args.class_weights:
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight

        labels = data_mod.train_labels(
            classes_filter=args.classes_filter,
            smoke=args.smoke_test,
            data_dir=args.data_dir,
            download=True if args.download else None,
        )
        present = np.unique(labels)
        weights = compute_class_weight("balanced", classes=present, y=labels)
        class_weight = {int(c): float(w) for c, w in zip(present, weights)}
        # Classes absent from the training split get weight 1.0 rather than
        # a KeyError during fit.
        for c in range(num_classes):
            class_weight.setdefault(c, 1.0)
        print("[train] class weights: min={:.3f} max={:.3f} over {} classes".format(
            min(class_weight.values()), max(class_weight.values()), len(class_weight)))

    # One ModelCheckpoint instance is shared across both training stages so
    # that its "best so far" spans them: if fine-tuning turns out worse than
    # the frozen-head model, the saved file stays the better of the two.
    # EarlyStopping / ReduceLROnPlateau are rebuilt per stage so their
    # patience counters do not carry over and cut fine-tuning short.
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    def stage_callbacks():
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
            ),
            checkpoint,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
        ]

    started = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=stage_callbacks(),
        class_weight=class_weight,
    )
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    ft_hist = None

    # ---- optional fine-tuning pass -------------------------------------
    if args.model == "transfer" and fine_tune_epochs > 0:
        print("\n[train] --- fine-tuning stage: unfreezing the top of the backbone ---")
        model_mod.unfreeze_top_layers(model, args.fine_tune_layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr / 10),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=len(hist["loss"]),
            callbacks=stage_callbacks(),
            class_weight=class_weight,
        )
        ft_hist = {k: [float(x) for x in v] for k, v in ft_history.history.items()}

    elapsed = time.time() - started
    merged = _merge_history(hist, ft_hist)

    # ModelCheckpoint(save_best_only) may not have fired if val_accuracy never
    # improved (possible in a 1-epoch smoke run); make sure a model exists.
    if not model_path.exists():
        model.save(model_path)
        print("[train] checkpoint never fired; saved the final weights instead")
    else:
        # Re-load the checkpoint so the metrics reported below (and recorded in
        # run_config.json) describe exactly the file evaluate.py will load.
        model = tf.keras.models.load_model(model_path)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    print("[train] model -> {}".format(model_path))

    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    print("[train] saved-checkpoint val: " + ", ".join(
        "{}={:.4f}".format(k, v) for k, v in val_metrics.items()))

    reports = config.REPORTS_DIR
    (reports / "{}_history.json".format(run_name)).write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    _plot_history(
        merged,
        reports / "{}_curves.png".format(run_name),
        "{} - {} classes{}".format(run_name, num_classes,
                                   " [SMOKE]" if args.smoke_test else ""),
    )

    run_config = {
        "run_name": run_name,
        "args": vars(args),
        "effective": {
            "epochs": epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "num_classes": num_classes,
            "class_names": class_names,
        },
        "results": {
            "final_val": {k: float(v) for k, v in val_metrics.items()},
            "best_val_accuracy": max(merged.get("val_accuracy", [float("nan")])),
            "train_seconds": round(elapsed, 1),
        },
        "artifacts": {
            "model": str(model_path),
            "history": str(reports / "{}_history.json".format(run_name)),
            "curves": str(reports / "{}_curves.png".format(run_name)),
            "class_names": str(config.CLASS_NAMES_FILE),
        },
        "environment": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (reports / "{}_run_config.json".format(run_name)).write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8"
    )
    print("[train] run config -> {}".format(reports / "{}_run_config.json".format(run_name)))

    if args.smoke_test:
        print("\n[train] SMOKE TEST PASSED. Metrics from a smoke run are meaningless "
              "by design - it only proves the pipeline runs end to end.")
    print("[train] next: python -m src.evaluate --model-path {}{}".format(
        model_path, " --smoke-test" if args.smoke_test else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
