"""YOLO Object Detection.

Main file for training, tuning, and inference.
"""

import argparse

from pathlib import Path
from typing import Dict

from bur_detection.training import YOLOTrainer
from bur_detection.tuning import YOLOTuner
from bur_detection.inference import YOLOInference
from bur_detection.utils import (
    load_config,
    plot_dataset_samples,
    get_output_dir,
)


def run_training(args, config: Dict, run_dir, override_params=None):
    """Train using config's training_params[0], or override_params.

    override_params (e.g. a preceding tune step's winner in a
    chained run) takes priority when given.
    """
    print("\n" + "=" * 80)
    print("Training with Best Known Hyperparameters")
    print("=" * 80)

    params = override_params or config["training_params"][0]
    if override_params:
        print(
            f"Using tuned hyperparameters handed off from the tuning step "
            f"(model_size={params.get('model_size')})."
        )

    trainer = YOLOTrainer(
        model_size=params["model_size"],
        prints_per_epoch=5,
        training_steps=config["training_steps"],
        score_weights=config.get("score_weights"),
        warmstart=config.get("warmstart", False),
        tal_topk=config.get("tal_topk"),
        step_transition_warmup_epochs=config.get(
            "step_transition_warmup_epochs", 10.0
        ),
    )

    trainer.train(
        yolo_data_dir=str(Path(config["data"]["training_dir"])),
        config=params,
        plot_mode=args.plot_mode,
        conf_threshold=config["inference"]["conf_threshold"],
        iou_threshold=config["inference"]["iou_threshold"],
        output_dir=Path(run_dir) / "train",
    )

    print(f"\nTraining complete! Results saved to: {trainer.output_dir}")


def run_tuning(args, config: Dict, run_dir):
    """Run hyperparameter tuning and return the winning config.

    Builds a YOLOTuner from config and runs it. If a best trial
    exists, returns its hyperparameters (for a chained train step);
    otherwise returns None.
    """
    print("\n" + "=" * 80)
    print("Starting Hyperparameter Tuning")
    print("=" * 80)

    tuner = YOLOTuner(
        num_samples=config["ray_tune"]["num_samples"],
        max_concurrent_trials=config["ray_tune"]["max_concurrent_trials"],
        yolo_data_dir=str(Path(config["data"]["training_dir"])),
        training_steps=config["training_steps"],
        # Seed trial 0 with training_params when warm_start is on; otherwise
        # Optuna explores from scratch (after search space changes greatly).
        points_to_evaluate=(
            config["training_params"]
            if config["ray_tune"].get("warm_start", False)
            else None
        ),
        tuning_space=config["tuning_space"],
        conf_threshold=config["inference"]["conf_threshold"],
        iou_threshold=config["inference"]["iou_threshold"],
        plot_mode=args.plot_mode,
        score_weights=config.get("score_weights"),
        analysis_enabled=config.get("analysis", {}).get("enabled", True),
        analysis_top_n=config.get("analysis", {}).get("top_n", 10),
        outputs_dir=str(Path(run_dir) / "tune"),
        warmstart=config.get("warmstart", False),
        tal_topk=config.get("tal_topk"),
        step_transition_warmup_epochs=config.get(
            "step_transition_warmup_epochs", 10.0
        ),
        keep_top_n=config["ray_tune"].get("keep_top_n", 5),
        # Cross-run winner index at the dataset's outputs/ root
        # (accumulates over runs).
        registry_path=str(
            Path(
                config["data"].get(
                    "outputs_dir", "bur_detection/sample_data/training/outputs"
                )
            )
            / "model_registry.csv"
        ),
    )

    tuner.run()

    best_output_dir = tuner.best_output_dir
    if not tuner.best_trial or best_output_dir is None:
        print("Tuning completed but no best trial found.")
        return None

    print(f"\nTuning complete! Results saved to: {best_output_dir}")

    # Return the best hyperparameters so a chained `train` step can use them.
    best_cfg_path = Path(best_output_dir) / "best_trial_config.json"
    if best_cfg_path.exists():
        import json

        return json.loads(best_cfg_path.read_text())
    return None


def run_inference(args, config: Dict, run_dir):
    """Run bur detection on unlabeled canopy images from config."""
    print("\n" + "=" * 80)
    print("Bur detection on unlabeled canopy images")
    print("=" * 80)

    tiling = config["data"].get("tiling", {})
    inference = YOLOInference(
        model_path=config["inference"]["model_path"],
        image_selections_path=config["data"]["image_selections"],
        conf_threshold=config["inference"]["conf_threshold"],
        iou_threshold=config["inference"]["iou_threshold"],
        plot_mode=args.plot_mode,
        global_nms_iou=config["inference"].get("global_nms_iou", 0.3),
        tile_batch_size=config["inference"].get("tile_batch_size", 96),
        tile_size=tiling.get("tile_size", 224),
        overlap=tiling.get("overlap", 0.2),
        output_dir=Path(run_dir) / "inference",
        outputs_dir=config["data"].get(
            "outputs_dir", "bur_detection/sample_data/training/outputs"
        ),
    )
    inference.run()

    print(f"Inference complete! Results saved to: {inference.output_dir}")


def run_preprocess(args, config: Dict, run_dir):
    """Build/refresh the tiled training set and save QA overlays.

    With data.full_res_images_dir + polygon_labels_dir set (e.g. via
    --data-root), runs the polygon tiler (canopy mask -> tile ->
    clip polygons to bboxes -> quality filters -> group-aware
    split); otherwise splits the pre-made tiles already in
    training_dir.
    """
    from bur_detection.dataset import (
        create_tiled_dataset,
        prepare_dataset_splits,
        bur_tile_group_key,
    )

    print("\n" + "=" * 80)
    print("Preprocessing: tile (optional) + group-aware split + QA")
    print("=" * 80)

    data = config["data"]
    training_dir = Path(data["training_dir"])
    split_cfg = config.get("split", {})
    fracs = tuple(split_cfg.get("fracs", [0.7, 0.2, 0.1]))
    seed = split_cfg.get("seed", 666)

    src_images = data.get("full_res_images_dir")
    src_labels = data.get("polygon_labels_dir")
    if src_images and src_labels:
        tcfg = data.get("tiling", {}) or {}
        print(f"Tiling full-resolution images from: {src_images}")
        create_tiled_dataset(
            images_dir=src_images,
            labels_dir=src_labels,
            output_dir=training_dir,
            canopy_dir=data.get("canopy_labels_dir"),
            tile_size=tcfg.get("tile_size", 224),
            overlap=tcfg.get("overlap", 0.2),
            min_canopy_frac=tcfg.get("min_canopy_frac", 0.15),
            bg_keep_ratio=tcfg.get("bg_keep_ratio", 0.3),
            dedup_iou=tcfg.get("dedup_iou", 0.8),
            seed=seed,
        )
    else:
        # No tiling source: split the pre-made tiles in training_dir.
        print(f"Splitting pre-made tiles in: {training_dir}")
        prepare_dataset_splits(
            images_dir=training_dir / "images",
            labels_dir=training_dir / "labels",
            output_dir=training_dir,
            splits=fracs,
            seed=seed,
            group_key_fn=bur_tile_group_key,
        )
    print(f"\nDataset ready at: {training_dir}")

    out_dir = Path(run_dir) / "preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_mode != "none":
        num_samples = 24 if args.plot_mode == "all" else 8
        print(f"\nSaving {num_samples} QA sample plots (train split)...")
        plot_dataset_samples(
            split_txt=training_dir / "train.txt",
            labels_dir=training_dir / "labels",
            save_dir=out_dir / "qa_samples",
            num_samples=num_samples,
            seed=seed,
            class_names={0: "Chestnut-bur"},
        )

    print(f"\nPreprocess reports -> {out_dir}")


def _apply_data_root(config, data_root):
    """Point the pipeline at a user dataset, without editing config.

    Derives the layout produced by the prep script and overrides
    config['data']:
      <root>/tiled                              -> training_dir
      <root>/outputs                            -> outputs_dir
      <root>/full_canopy/{images,labels,canopy} -> tiling source
    """
    root = Path(data_root)
    d = config.setdefault("data", {})
    d["training_dir"] = str(root / "tiled")
    d["outputs_dir"] = str(root / "outputs")
    d["full_res_images_dir"] = str(root / "full_canopy" / "images")
    d["polygon_labels_dir"] = str(root / "full_canopy" / "labels")
    d["canopy_labels_dir"] = str(root / "full_canopy" / "canopy")
    print(f"Using --data-root {root} (training_dir={d['training_dir']})")
    return config


def run_detection():
    """Parse CLI args and run the requested pipeline mode(s) in order."""
    parser = argparse.ArgumentParser(
        description="YOLO Bur Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build the tiled training set from full_canopy/ images + polygon labels
  python -m bur_detection.detection --mode preprocess --plot-mode subset

  # Hyperparameter tuning
  python -m bur_detection.detection --mode tune --plot-mode none

  # Full pipeline in one command (tune hands its best hparams to train)
  python -m bur_detection.detection --mode preprocess,tune,train,inference `
      --plot-mode subset

  # Point at your own dataset (overrides the sample paths)
  python -m bur_detection.detection --mode preprocess,tune,train,inference `
      --data-root "C:/path/to/your/dataset" --plot-mode subset
        """,
    )

    # Mode selection (one mode, or a comma-separated sequence run in order)
    parser.add_argument(
        "--mode",
        default="inference",
        help="Operation mode, or a comma-separated sequence run in "
        "order, from: "
        "preprocess, tune, train, inference. "
        "Example: --mode preprocess,tune,train,inference (tune hands its best "
        "hyperparameters to train).",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="bur_detection/config.yml",
        help="Path to configuration YAML file",
    )

    # Plotting arguments
    parser.add_argument(
        "--plot-mode",
        choices=["all", "subset", "none"],
        default="subset",
        help="How many prediction plots to save",
    )

    # Point at your own dataset without editing the committed (sample) config.
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root of your dataset; derives tiled/, full_canopy/ and "
        "outputs/, overriding the "
        "sample paths. Unset = use sample data.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        _apply_data_root(config, args.data_root)

    valid = ["preprocess", "tune", "train", "inference"]
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    bad = [m for m in modes if m not in valid]
    if bad:
        parser.error(
            f"invalid --mode value(s) {bad}; choose from {valid} "
            f"(single or comma-separated)"
        )

    # One run folder per invocation:
    # outputs/run_<ts>/{preprocess,tune,train,inference}/
    outputs_base = config["data"].get(
        "outputs_dir", "bur_detection/sample_data/training/outputs"
    )
    run_dir = get_output_dir(outputs_base, "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun outputs -> {run_dir}")

    tuned_params = None
    for mode in modes:
        if mode == "preprocess":
            run_preprocess(args, config, run_dir)
        elif mode == "tune":
            tuned_params = run_tuning(args, config, run_dir)
        elif mode == "train":
            run_training(args, config, run_dir, override_params=tuned_params)
        else:  # inference
            run_inference(args, config, run_dir)


if __name__ == "__main__":
    run_detection()
