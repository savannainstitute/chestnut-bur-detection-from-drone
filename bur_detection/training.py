"""Train YOLO models with multi-step progressive unfreezing.

Handles early stopping, learning-rate warmup, and optimizer-state
handoff between training steps.
"""

import time
import math
import sys
import os
import shutil
import pandas as pd
import random
from copy import deepcopy
from pathlib import Path
from datetime import timedelta
import logging
from typing import cast
import torch
from ultralytics import YOLO

from bur_detection.utils import (
    SmoothedValue,
    MetricLogger,
    set_seed,
    evaluate_test_set,
    plot_ground_truth_vs_predictions,
    get_output_dir,
    compute_composite_objective,
    pick_device,
)


def set_trainable_layers(model, num_layers, runtime_model=None):
    """Unfreeze model layers progressively based on the stage number.

    Stage-based unfreezing follows the model YAML backbone/head split:
    1 = predictor only, 2 = full head, 3 = head + last third of
    backbone, 4+ = full model. Pass runtime_model (the live
    trainer.model) to also freeze there -- required because
    Ultralytics re-enables requires_grad on user-frozen params during
    train setup, so a freeze on the wrapper before .train() doesn't
    stick. Returns a stats dict.
    """
    stage = max(1, int(num_layers))

    def _head_start_idx(core_model, n_blocks):
        cfg = getattr(core_model, "yaml", {}) or {}
        bb = cfg.get("backbone", []) if isinstance(cfg, dict) else []
        hd = cfg.get("head", []) if isinstance(cfg, dict) else []
        if isinstance(bb, list) and isinstance(hd, list):
            if len(bb) > 0 and len(hd) > 0 and len(bb) + len(hd) == n_blocks:
                return int(len(bb))
            if len(hd) > 0 and len(hd) <= n_blocks:
                return int(max(0, n_blocks - len(hd)))
        # Fallback when the yaml partition is unavailable: treat the
        # last 3 blocks as the head.
        return int(max(0, n_blocks - min(3, n_blocks)))

    def _apply_to_core(core_model, stage_idx):
        blocks = list(core_model.model)
        n_blocks = len(blocks)
        if n_blocks == 0:
            return {
                "trainable_params": 0,
                "trainable_tensors": 0,
                "n_blocks": 0,
                "head_start": 0,
                "predictor_idx": -1,
            }

        predictor_idx = n_blocks - 1
        head_start = min(
            max(0, _head_start_idx(core_model, n_blocks)), predictor_idx
        )
        backbone_last = head_start - 1
        backbone_count = max(0, backbone_last + 1)

        for p in core_model.parameters():
            p.requires_grad = False

        def _unfreeze(start_idx, end_idx):
            for bi in range(max(0, start_idx), min(n_blocks, end_idx + 1)):
                for p in blocks[bi].parameters():
                    p.requires_grad = True

        _unfreeze(predictor_idx, predictor_idx)  # stage 1: predictor only
        if stage_idx >= 2:
            _unfreeze(head_start, predictor_idx)  # stage 2: full head
        if stage_idx >= 3 and backbone_count > 0:
            k = max(1, int(round(backbone_count / 3.0)))
            _unfreeze(
                backbone_count - k, backbone_last
            )  # stage 3: + last third of backbone
        if stage_idx >= 4:
            _unfreeze(0, predictor_idx)  # stage 4+: full model

        return {
            "trainable_params": sum(
                p.numel() for p in core_model.parameters() if p.requires_grad
            ),
            "trainable_tensors": sum(
                1 for p in core_model.parameters() if p.requires_grad
            ),
            "n_blocks": n_blocks,
            "head_start": head_start,
            "predictor_idx": predictor_idx,
        }

    _apply_to_core(model.model, stage)
    target_core = runtime_model if runtime_model is not None else model.model
    return _apply_to_core(target_core, stage)


class YOLOTrainer:
    """Run multi-step progressive YOLO training with early stopping.

    Wraps an Ultralytics YOLO model, unfreezing layers in stages
    across configured training steps and carrying optimizer state
    and best-epoch weights between them.
    """

    def __init__(
        self,
        model_size="yolo11n.pt",
        prints_per_epoch=5,
        ray_tune_callback=None,
        training_steps=None,
        score_weights=None,
        warmstart=False,
        tal_topk=None,
        step_transition_warmup_epochs=10.0,
    ):
        """Build the model and initialize per-run training state."""
        self.model = self._create_model(model_size, warmstart)
        self.model_size = model_size
        if tal_topk is not None:
            self._override_tal_topk(int(tal_topk))
        self.prints_per_epoch = prints_per_epoch
        self.ray_tune_callback = ray_tune_callback
        self.training_steps = training_steps
        self.score_weights = score_weights or {
            "loss": 0.45,
            "f1": 0.35,
            "map50": 0.20,
        }
        self._prev_stage_trainable_tensors = None
        self.current_step_patience = 0
        # LR warmup epochs at each progressive-unfreeze step transition.
        self.step_transition_warmup_epochs = float(
            step_transition_warmup_epochs
        )
        self._pending_handoff_snapshot = None
        self._pending_model_handoff_state = None
        self._current_step_best_snapshot = None
        self._current_step_best_model_state = None
        self._current_step_best_objective = float("inf")
        self._step_start_lr = None
        self._step_target_lr = None
        self._step_warmup_iters = 0
        self._step_warmup_iter_idx = 0
        self._manual_step_warmup_active = False
        self.batch_idx = 0
        self.num_batches = 0
        self.current_epoch = 0
        self.epochs = 0
        self.end_time = time.time()
        self.iter_time = SmoothedValue(fmt="{avg:.4f}")
        self.train_metric_logger = MetricLogger(delimiter="  ")
        self.train_metric_logger.add_meter(
            "lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        self.validation_metrics = {}
        self.metrics_history = []
        self.yaml_path = None
        self.output_dir = None
        self.final_weights_path = None
        self.training_metrics_path = None
        self.test_preds = None
        self.best_model_path = None
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

    def _create_model(self, model_size, warmstart):
        """Build the YOLO model, warm-starting from pretrained weights.

        For a .yaml arch with warmstart, load the matching pretrained
        .pt to transfer the backbone (the new P2 head stays random and
        is learned during fine-tuning).
        """
        model = YOLO(model_size)
        if warmstart and str(model_size).endswith(".yaml"):
            base = Path(model_size).stem.replace("-p2", "").replace("-p6", "")
            try:
                model.load(f"{base}.pt")
                print(f"warm-started {Path(model_size).name} from {base}.pt")
            except Exception as e:
                print(f"warm-start from {base}.pt skipped: {e}")
        return model

    def _override_tal_topk(self, tal_topk):
        """Fix the TAL assigner top-k via a contained instance override.

        Uses an instance-level init_criterion override (not a global
        monkeypatch), pinned to ultralytics
        v8DetectionLoss(model, tal_topk=...).
        """
        from ultralytics.utils.loss import v8DetectionLoss

        core = self.model.model
        core.init_criterion = lambda: v8DetectionLoss(
            core, tal_topk=int(tal_topk)
        )
        print(f"tal_topk override = {tal_topk}")

    def train(
        self,
        yolo_data_dir,
        config=None,
        conf_threshold=0.5,
        iou_threshold=0.45,
        plot_mode="subset",
        outputs_dir=None,
        output_dir=None,
    ):
        """Run the full multi-step progressive training pipeline.

        Validates the YOLO dataset directory, coerces numeric
        hyperparameters from config, then trains each configured step
        with progressive layer unfreezing and LR warmup. Outside of
        Ray Tune, also selects the best checkpoint across steps,
        copies its weights, and evaluates it against the test set.
        """
        if config is None:
            config = {}
        # PyYAML loads bare sci-notation (`1e-5`) as str; coerce
        # numeric hparams to float.
        _numeric_hparams = (
            "lr0",
            "lrf",
            "max_lr0",
            "max_scaled_lr",
            "lr_scale_power",
            "momentum",
            "weight_decay",
            "warmup_momentum",
            "warmup_bias_lr",
            "box_gain",
            "cls_gain",
            "dfl_gain",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "scale",
            "shear",
            "perspective",
            "mosaic",
            "mixup",
            "copy_paste",
            "flipud",
            "dropout",
        )
        for _k in _numeric_hparams:
            if config.get(_k) is not None and not isinstance(config[_k], bool):
                try:
                    config[_k] = float(config[_k])
                except (TypeError, ValueError):
                    pass
        set_seed(666)
        os.environ["TQDM_DISABLE"] = "1"
        try:
            from tqdm import tqdm

            tqdm.disable = True
        except Exception:
            pass
        yaml_path = Path(yolo_data_dir) / "dataset.yml"
        train_txt_path = Path(yolo_data_dir) / "train.txt"
        val_txt_path = Path(yolo_data_dir) / "val.txt"
        test_txt_path = Path(yolo_data_dir) / "test.txt"

        for f in [yaml_path, train_txt_path, val_txt_path, test_txt_path]:
            if not f.exists():
                raise FileNotFoundError(f"Required file not found: {f}")

        self.yaml_path = yaml_path

        # imgsz = the tile size, so the model trains at native
        # resolution (no resize).
        from PIL import Image as _PILImage

        _tiles = sorted(
            (Path(yolo_data_dir) / "images").glob("*.jpg")
        ) or sorted((Path(yolo_data_dir) / "images").glob("*.png"))
        self.imgsz = _PILImage.open(_tiles[0]).width if _tiles else 224
        print(f"imgsz = {self.imgsz} (native tile size, no resize)")

        device = pick_device()
        if device == "cpu":
            print("No GPU (CUDA/MPS) available, using CPU for training")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if output_dir is not None:
            output_dir = Path(
                output_dir
            )  # exact dir (e.g. run_<ts>/train or a tuning trial dir)
        else:
            base = (
                str(outputs_dir)
                if outputs_dir
                else str(Path(yolo_data_dir).parent / "outputs")
            )
            output_dir = get_output_dir(base, "training", timestamp)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        start_step = config.get("_resume_from_step", 0)
        step_epochs_completed = config.get("_resume_step_epochs", 0)
        total_epochs = config.get("_resume_total_epochs", 0)

        training_steps = (
            self.training_steps
            if self.training_steps is not None
            else [
                {
                    "batch": 8,
                    "accumulate": 1,
                    "max_epochs": 50,
                    "patience": 25,
                },
                {
                    "batch": 8,
                    "accumulate": 4,
                    "max_epochs": 50,
                    "patience": 20,
                },
                {
                    "batch": 8,
                    "accumulate": 16,
                    "max_epochs": 50,
                    "patience": 15,
                },
                {
                    "batch": 8,
                    "accumulate": 64,
                    "max_epochs": 50,
                    "patience": 10,
                },
            ]
        )

        requested_lr0 = config.get("lr0", 0.001)
        max_lr0 = config.get("max_lr0", 0.01)
        lr0 = min(requested_lr0, max_lr0)
        lr_ref_eb = config.get("lr_reference_effective_batch", 64)
        lr_scale_power = config.get("lr_scale_power", 0.5)
        max_scaled_lr = config.get("max_scaled_lr", 0.03)
        self.score_weights = config.get("score_weights", self.score_weights)
        self._prev_stage_trainable_tensors = None
        self._pending_handoff_snapshot = None
        self._pending_model_handoff_state = None
        self.metrics_history = []
        # save=False under tuning: per-epoch last.pt/best.pt overwrites raced
        # Windows Defender and killed trials; Ray checkpoints carry the model.
        save = self.ray_tune_callback is None
        save_period = 1 if save else -1
        print("\n" + "-" * 80)
        print(f"Starting YOLO training: {self.model_size}")
        print("-" * 80)
        with open(train_txt_path, "r") as f:
            num_train_images = len(f.readlines())
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        for step_idx in range(start_step, len(training_steps)):
            step = training_steps[step_idx]
            print(f"\nStep {step_idx + 1}/{len(training_steps)}")
            print("-" * 60)
            effective_batch = step["batch"] * step["accumulate"]
            lr_multiplier = (effective_batch / lr_ref_eb) ** lr_scale_power
            scaled_lr = min(lr0 * lr_multiplier, max_scaled_lr)
            if step_idx == start_step and step_epochs_completed > 0:
                epochs_to_run = step["max_epochs"] - step_epochs_completed
                if epochs_to_run <= 0:
                    continue
                print(
                    f"Resuming step {step_idx + 1} from epoch "
                    f"{step_epochs_completed + 1}"
                )
            else:
                epochs_to_run = step["max_epochs"]
            self.epochs = epochs_to_run
            # Resume from the previous step's best-objective epoch, not its
            # last: lenient patience runs well past the best.
            if self._pending_model_handoff_state is not None:
                cast("torch.nn.Module", self.model.model).load_state_dict(
                    self._pending_model_handoff_state
                )
                # Ultralytics strips overrides['model'] after each .train();
                # without it the next train() raises KeyError: 'model'.
                self.model.overrides["model"] = self.model_size
            self.current_step = step_idx + 1
            self.current_step_patience = step["patience"]
            # step_idx is 0-based; +1 so step 1 trains the head (not
            # "freeze all").
            stage_stats = set_trainable_layers(
                self.model, num_layers=step_idx + 1
            )
            if (
                self._prev_stage_trainable_tensors is not None
                and stage_stats["trainable_tensors"]
                <= self._prev_stage_trainable_tensors
            ):
                raise RuntimeError(
                    f"Non-increasing staged unfreeze at step {step_idx + 1}: "
                    f"trainable tensors {stage_stats['trainable_tensors']} <= "
                    f"{self._prev_stage_trainable_tensors}"
                )
            self._prev_stage_trainable_tensors = stage_stats[
                "trainable_tensors"
            ]
            print(
                f"Trainable parameters: {stage_stats['trainable_params']:,} "
                f"across {stage_stats['trainable_tensors']} tensors "
                f"(blocks={stage_stats['n_blocks']}, "
                f"head_start={stage_stats['head_start']})"
            )
            self.total_epochs_so_far = total_epochs
            batches_per_epoch = num_train_images // step["batch"] + (
                1 if num_train_images % step["batch"] > 0 else 0
            )
            self.num_batches = batches_per_epoch
            print_freq = max(1, batches_per_epoch // self.prints_per_epoch)
            self._step_target_lr = scaled_lr
            self._step_warmup_iters = int(
                max(0.0, self.step_transition_warmup_epochs)
                * batches_per_epoch
            )
            self._step_warmup_iter_idx = 0
            self._manual_step_warmup_active = False
            self._current_step_best_snapshot = None
            self._current_step_best_model_state = None
            self._current_step_best_objective = float("inf")
            if self.current_step == 1:
                print(
                    f"Dataset: {num_train_images} images, "
                    f"{batches_per_epoch} batches per epoch"
                )
            print(
                f"Training with effective batch size "
                f"{int(step['batch'] * step['accumulate'])}, "
                f"lr={scaled_lr:.6f}"
            )
            print()
            # clear prior-step callbacks so re-adds don't stack on a
            # reused wrapper
            self.model.reset_callbacks()
            self.model.add_callback("on_train_start", self._on_train_start)
            self.model.add_callback(
                "on_train_batch_start", self._on_batch_start
            )
            self.model.add_callback(
                "on_train_batch_end",
                lambda trainer: self._on_batch_end(trainer, print_freq),
            )
            self.model.add_callback("on_train_epoch_end", self._on_epoch_end)
            self.model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)
            self.model.add_callback("on_val_end", self._on_val_end)
            self.model.add_callback("on_train_end", self._on_train_end)
            self.batch_idx = 0
            self._reset_loggers()
            if hasattr(self, "_last_val_epoch"):
                delattr(self, "_last_val_epoch")
            self.model.train(
                data=str(yaml_path),
                epochs=epochs_to_run,
                patience=step["patience"],
                batch=step["batch"],
                imgsz=self.imgsz,
                lr0=scaled_lr,
                lrf=config.get("lrf", 0.01),
                optimizer=config.get("optimizer", "AdamW"),
                nbs=effective_batch,
                # manual step-transition warmup overrides Ultralytics' warmup
                warmup_epochs=0.0,
                warmup_momentum=config.get("warmup_momentum", 0.8),
                warmup_bias_lr=config.get("warmup_bias_lr", 0.0005),
                weight_decay=config.get("weight_decay", 0.0005),
                momentum=config.get("momentum", 0.937),
                box=config.get("box_gain", 7.5),
                cls=config.get("cls_gain", 0.5),
                dfl=config.get("dfl_gain", 1.5),
                hsv_h=config.get("hsv_h", 0.015),
                hsv_s=config.get("hsv_s", 0.7),
                hsv_v=config.get("hsv_v", 0.4),
                degrees=config.get("degrees", 0),
                scale=config.get("scale", 0.5),
                shear=config.get("shear", 0),
                perspective=config.get("perspective", 0),
                # off -- shrinks small burs + stitches unnatural canopy
                # composites
                mosaic=config.get("mosaic", 0.0),
                mixup=config.get("mixup", 0),
                copy_paste=config.get("copy_paste", 0),
                # vertical flip on -- nadir imagery has no canonical "up"
                # (like fliplr)
                flipud=config.get("flipud", 0.5),
                dropout=config.get("dropout", 0),
                project=str(output_dir),
                name=f"train_step{step_idx + 1}",
                exist_ok=True,
                device=device,
                workers=0,
                plots=False,
                save=save,
                save_period=save_period,
                verbose=False,
            )
            self._pending_handoff_snapshot = deepcopy(
                self._current_step_best_snapshot
            )
            self._pending_model_handoff_state = (
                self._current_step_best_model_state
            )
            total_epochs += (
                self.current_epoch
            )  # actual epochs run this step, not the max_epochs cap
        print("\n" + "=" * 80)
        print(f"Training complete - {total_epochs} epochs")
        print("=" * 80)
        all_metrics_df = pd.DataFrame(self.metrics_history)
        self.training_metrics_path = output_dir / "training_metrics.csv"
        all_metrics_df.to_csv(self.training_metrics_path, index=False)

        # Only run outside of Ray Tune to avoid conflicts
        if self.ray_tune_callback is None:
            best_obj = float("inf")
            best_f1_at_best = 0.0
            best_step_dir = None
            best_epoch = None
            for step_dir in sorted(output_dir.glob("train_step*/")):
                results_csv = step_dir / "results.csv"
                weights_dir = step_dir / "weights"
                if results_csv.exists() and weights_dir.exists():
                    df = pd.read_csv(results_csv)
                    for _, row in df.iterrows():
                        precision = row.get("metrics/precision(B)", 0.0)
                        recall = row.get("metrics/recall(B)", 0.0)
                        map50 = row.get("metrics/mAP50(B)", 0.0)
                        val_loss = (
                            float(row.get("val/box_loss", 0.0) or 0.0)
                            + float(row.get("val/cls_loss", 0.0) or 0.0)
                            + float(row.get("val/dfl_loss", 0.0) or 0.0)
                        )
                        f1 = self._calculate_f1(precision, recall)
                        obj = compute_composite_objective(
                            val_loss, f1, map50, self.score_weights
                        )
                        if obj < best_obj:
                            best_obj = obj
                            best_f1_at_best = f1
                            best_step_dir = step_dir
                            best_epoch = int(row["epoch"])
            final_weights = None
            if best_step_dir is not None and best_epoch is not None:
                candidate = (
                    best_step_dir / "weights" / f"epoch{best_epoch - 1}.pt"
                )
                if candidate.exists():
                    final_weights = candidate
            best_model_path = str(final_weights) if final_weights else None
            print(
                f"Best model training path: {best_model_path} "
                f"(objective={best_obj:.4f}, F1={best_f1_at_best:.4f})"
            )
            final_weights_path = Path(output_dir) / "best_model_weights.pt"
            if final_weights:
                shutil.copy2(final_weights, final_weights_path)
            self.final_weights_path = final_weights_path
            self.best_model_path = (
                final_weights_path if final_weights else None
            )
            print(f"Final weights saved to: {final_weights_path}")

            if final_weights:
                self.test_preds = evaluate_test_set(
                    model_path=final_weights_path,
                    training_dir=Path(yolo_data_dir),
                    output_dir=output_dir,
                    plot_mode=plot_mode,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
            if self.test_preds:
                images_dir = Path(yolo_data_dir) / "images"
                labels_dir = Path(yolo_data_dir) / "labels"
                plot_dir = Path(output_dir) / "prediction_plots"

                if plot_mode == "none":
                    predictions_to_plot = []
                elif plot_mode == "subset":
                    sample_size = min(15, len(self.test_preds))
                    predictions_to_plot = random.sample(
                        self.test_preds, sample_size
                    )
                else:
                    predictions_to_plot = self.test_preds

                if predictions_to_plot:
                    plot_ground_truth_vs_predictions(
                        predictions=predictions_to_plot,
                        labels_dir=labels_dir,
                        original_images_dir=images_dir,
                        save_dir=plot_dir,
                        conf_threshold=conf_threshold,
                    )

    def _calculate_f1(self, precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _reset_loggers(self):
        self.train_metric_logger = MetricLogger(delimiter="  ")
        self.train_metric_logger.add_meter(
            "lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        self.validation_metrics = {}
        self.end_time = time.time()
        self.iter_time = SmoothedValue(fmt="{avg:.4f}")

    def _on_train_start(self, trainer):
        """Re-apply the staged freeze on the live trainer model.

        Ultralytics re-enables requires_grad during setup, so the
        pre-.train() freeze doesn't stick without this. Also restores
        the previous step's best-epoch optimizer state and arms the
        manual LR warmup.
        """
        stage = int(getattr(self, "current_step", 1))
        stats = set_trainable_layers(
            self.model, num_layers=stage, runtime_model=trainer.model
        )
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = self._original_stdout, self._original_stderr
        try:
            print(
                f"Runtime staged freeze (step {stage}): "
                f"{stats['trainable_params']:,} params across "
                f"{stats['trainable_tensors']} tensors trainable"
            )
            if stage > 1 and isinstance(self._pending_handoff_snapshot, dict):
                restored = self._restore_optimizer_by_name(
                    trainer, self._pending_handoff_snapshot
                )
                best_lr = self._pending_handoff_snapshot.get("best_lr")
                self._step_start_lr = (
                    float(best_lr) if best_lr is not None else 0.0
                )
                print(
                    f"Restored optimizer state for {restored} tensors; "
                    f"warmup LR {self._step_start_lr:.6f} -> "
                    f"{float(self._step_target_lr or 0.0):.6f}"
                )
            else:
                self._step_start_lr = 0.0
            for group in trainer.optimizer.param_groups:
                group["lr"] = self._step_start_lr
            self._step_warmup_iter_idx = 0
            self._manual_step_warmup_active = int(self._step_warmup_iters) > 0
            if hasattr(trainer, "args"):
                try:
                    trainer.args.warmup_epochs = 0.0
                    trainer.args.warmup_bias_lr = 0.0
                except Exception:
                    pass
        finally:
            self._pending_handoff_snapshot = None
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _on_train_end(self, trainer):
        """Write last.pt if missing to avoid a crash on post-train reload.

        With save=False, Ultralytics only writes last.pt on the final
        epoch, which an early stop skips.
        """
        try:
            last_path = getattr(trainer, "last", None)
            if last_path is not None and not Path(last_path).exists():
                trainer.save_model()
        except Exception:
            pass

    def _on_batch_start(self, trainer):
        """Linearly ramp the learning rate toward the step's target LR.

        Ramps from the step's start LR over the warmup iterations
        (manual step-transition warmup).
        """
        if not self._manual_step_warmup_active:
            return
        if self._step_start_lr is None or self._step_target_lr is None:
            self._manual_step_warmup_active = False
            return
        warmup = max(1, int(self._step_warmup_iters))
        alpha = min(1.0, self._step_warmup_iter_idx / max(1, warmup - 1))
        lr_now = self._step_start_lr + alpha * (
            self._step_target_lr - self._step_start_lr
        )
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr_now
        self._step_warmup_iter_idx += 1
        if self._step_warmup_iter_idx >= warmup:
            self._manual_step_warmup_active = False

    def _on_fit_epoch_end(self, trainer):
        """Snapshot optimizer state at the current step's best epoch.

        Captures the state at the best-objective epoch so it can be
        handed off to the next (more-unfrozen) step.
        """
        # The unfreeze LR ramp spikes val loss; hold EarlyStopping's "best"
        # anchor at the current epoch through warmup so it cannot end the step.
        stopper = getattr(trainer, "stopper", None)
        grace_epochs = max(
            1, int(math.ceil(self.step_transition_warmup_epochs))
        )
        if (
            stopper is not None
            and self.current_step > 1
            and self.current_epoch <= grace_epochs
        ):
            stopper.best_epoch = self.current_epoch
            stopper.best_fitness = float(
                getattr(trainer, "fitness", 0.0) or 0.0
            )
            trainer.stop = False
        vm = self.validation_metrics
        if not isinstance(vm, dict) or "val_loss" not in vm:
            return
        objective = compute_composite_objective(
            vm.get("val_loss", float("nan")),
            vm.get("val_f1", 0.0),
            vm.get("val_mAP50", 0.0),
            self.score_weights,
        )
        if not math.isfinite(objective):
            return
        if objective < self._current_step_best_objective - 1e-12:
            snapshot = self._snapshot_optimizer_by_name(trainer)
            if isinstance(snapshot, dict):
                self._current_step_best_objective = float(objective)
                self._current_step_best_snapshot = snapshot
                # Snapshot this epoch's weights with the optimizer state for
                # the next step's handoff.
                model = getattr(trainer, "model", None)
                if model is not None:
                    self._current_step_best_model_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }

    def _snapshot_optimizer_by_name(self, trainer):
        """Capture optimizer buffers keyed by parameter name.

        Also captures the current LR, so the snapshot can be restored
        into the next step's optimizer for overlapping params.
        """
        if trainer is None or getattr(trainer, "optimizer", None) is None:
            return None
        model = getattr(trainer, "model", None)
        if model is None:
            return None
        named = dict(model.named_parameters())
        opt_state = trainer.optimizer.state_dict()
        param_groups = opt_state.get("param_groups", [])
        state = opt_state.get("state", {})
        id_to_name = {}
        for group, gstate in zip(trainer.optimizer.param_groups, param_groups):
            for p_obj, pid in zip(
                group.get("params", []), gstate.get("params", [])
            ):
                for name, ref in named.items():
                    if ref is p_obj:
                        id_to_name[pid] = name
                        break
        state_by_name = {}
        for pid, vals in state.items():
            name = id_to_name.get(pid)
            if name is None:
                continue
            state_by_name[name] = {
                k: (
                    v.detach().cpu().clone()
                    if torch.is_tensor(v)
                    else deepcopy(v)
                )
                for k, v in vals.items()
            }
        best_lr = (
            float(trainer.optimizer.param_groups[0].get("lr", 0.0))
            if trainer.optimizer.param_groups
            else None
        )
        return {"state_by_name": state_by_name, "best_lr": best_lr}

    def _restore_optimizer_by_name(self, trainer, snapshot):
        """Restore optimizer buffers for trainable params in the snapshot.

        Only restores shape-compatible params whose names overlap the
        snapshot. Returns the number of params restored.
        """
        if not isinstance(snapshot, dict):
            return 0
        state_by_name = snapshot.get("state_by_name") or {}
        model = getattr(trainer, "model", None)
        if model is None or getattr(trainer, "optimizer", None) is None:
            return 0
        named = dict(model.named_parameters())
        new_sd = trainer.optimizer.state_dict()
        new_groups = new_sd.get("param_groups", [])
        new_state = new_sd.get("state", {})
        restored = 0
        for group, gstate in zip(trainer.optimizer.param_groups, new_groups):
            for p_obj, pid in zip(
                group.get("params", []), gstate.get("params", [])
            ):
                if not getattr(p_obj, "requires_grad", False):
                    continue
                name = next(
                    (n for n, ref in named.items() if ref is p_obj), None
                )
                if name is None:
                    continue
                buf = state_by_name.get(name)
                if not isinstance(buf, dict):
                    continue
                candidate, ok = {}, True
                for k, v in buf.items():
                    if torch.is_tensor(v):
                        if v.shape == p_obj.shape:
                            candidate[k] = v.to(
                                device=p_obj.device, dtype=p_obj.dtype
                            )
                        elif v.numel() == 1:
                            candidate[k] = v.to(device=p_obj.device)
                        else:
                            ok = False
                            break
                    else:
                        candidate[k] = deepcopy(v)
                if ok:
                    new_state[pid] = candidate
                    restored += 1
        new_sd["state"] = new_state
        trainer.optimizer.load_state_dict(new_sd)
        return restored

    def _on_batch_end(self, trainer, print_freq):
        self.batch_idx += 1
        self.current_epoch = trainer.epoch + 1
        batch_time = time.time() - self.end_time
        self.iter_time.update(batch_time)
        self.end_time = time.time()
        box_loss = (
            float(trainer.loss_items[0])
            if len(trainer.loss_items) > 0
            else 0.0
        )
        cls_loss = (
            float(trainer.loss_items[1])
            if len(trainer.loss_items) > 1
            else 0.0
        )
        dfl_loss = (
            float(trainer.loss_items[2])
            if len(trainer.loss_items) > 2
            else 0.0
        )
        # Replace NaN/inf with a large sentinel instead of crashing the
        # run/trial; the ASHA scheduler prunes genuinely-bad trials on its own.
        if math.isnan(box_loss) or math.isinf(box_loss):
            box_loss = 100.0
        if math.isnan(cls_loss) or math.isinf(cls_loss):
            cls_loss = 100.0
        if math.isnan(dfl_loss) or math.isinf(dfl_loss):
            dfl_loss = 100.0
        total_loss = box_loss + cls_loss + dfl_loss
        self.train_metric_logger.update(
            loss=total_loss,
            box_loss=box_loss,
            cls_loss=cls_loss,
            dfl_loss=dfl_loss,
        )
        self.train_metric_logger.update(
            lr=trainer.optimizer.param_groups[0]["lr"]
        )
        if (
            self.batch_idx == 1
            or self.batch_idx % print_freq == 0
            or self.batch_idx == self.num_batches
        ):
            total_batches = getattr(trainer, "nb", self.num_batches)
            remaining_batches = max(0, total_batches - self.batch_idx)
            eta_seconds = self.iter_time.global_avg * remaining_batches
            eta_string = str(timedelta(seconds=int(eta_seconds)))
            gpu_mem = ""
            mb = 1024.0 * 1024.0
            if torch.cuda.is_available():
                gpu_mem = (
                    f"max mem: {torch.cuda.max_memory_allocated() / mb:.0f}M"
                )
            elif torch.backends.mps.is_available():
                # mps has no max-allocated tracking
                gpu_mem = (
                    f"mem: {torch.mps.current_allocated_memory() / mb:.0f}M"
                )
            header = f"Epoch: [{self.current_epoch}/{self.epochs}] Training"
            progress = f"[{self.batch_idx}/{total_batches}]"
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = (
                self._original_stdout,
                self._original_stderr,
            )
            print(
                f"{header} {progress} eta: {eta_string} "
                f"{self.train_metric_logger} time: {self.iter_time} {gpu_mem}"
            )
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def _on_epoch_end(self, trainer):
        self.batch_idx = 0
        self.current_epoch = trainer.epoch + 1

    def _on_val_end(self, validator):
        try:
            val_metrics = validator.metrics
            if (
                hasattr(self, "_last_val_epoch")
                and self._last_val_epoch == self.current_epoch
            ):
                return
            self._last_val_epoch = self.current_epoch
            map50 = (
                float(val_metrics.box.map50)
                if hasattr(val_metrics.box, "map50")
                else 0.0
            )
            precision = (
                float(val_metrics.box.p[0])
                if hasattr(val_metrics.box, "p") and len(val_metrics.box.p) > 0
                else 0.0
            )
            recall = (
                float(val_metrics.box.r[0])
                if hasattr(val_metrics.box, "r") and len(val_metrics.box.r) > 0
                else 0.0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            fitness = getattr(val_metrics, "fitness", 0.0)
            if (
                hasattr(validator, "loss")
                and validator.loss is not None
                and hasattr(validator, "dataloader")
            ):
                num_batches = len(validator.dataloader)
                val_box_loss = float(validator.loss[0]) / num_batches
                val_cls_loss = float(validator.loss[1]) / num_batches
                val_dfl_loss = float(validator.loss[2]) / num_batches
            else:
                val_box_loss = 0.0
                val_cls_loss = 0.0
                val_dfl_loss = 0.0
            val_total_loss = val_box_loss + val_cls_loss + val_dfl_loss
            self.validation_metrics = {
                "val_mAP50": map50,
                "val_fitness": fitness,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "val_loss": val_total_loss,
                "val_box_loss": val_box_loss,
                "val_cls_loss": val_cls_loss,
                "val_dfl_loss": val_dfl_loss,
            }
            # Sanitize NaN/inf to 0.0 instead of crashing the run/trial.
            for key, value in list(self.validation_metrics.items()):
                if math.isnan(value) or math.isinf(value):
                    self.validation_metrics[key] = 0.0
            train_metrics = {
                "lr": self.train_metric_logger.meters["lr"].value,
                "train_loss": self.train_metric_logger.meters[
                    "loss"
                ].global_avg,
                "train_box_loss": self.train_metric_logger.meters[
                    "box_loss"
                ].global_avg,
                "train_cls_loss": self.train_metric_logger.meters[
                    "cls_loss"
                ].global_avg,
                "train_dfl_loss": self.train_metric_logger.meters[
                    "dfl_loss"
                ].global_avg,
            }
            actual_epoch = self.current_epoch + getattr(
                self, "total_epochs_so_far", 0
            )
            epoch_metrics = {
                **train_metrics,
                **self.validation_metrics,
                "epoch": self.current_epoch,
                "step": self.current_step,
                "training_iteration": actual_epoch,
                "step_patience": int(
                    getattr(self, "current_step_patience", 0)
                ),
            }
            self.metrics_history.append(epoch_metrics)
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = (
                self._original_stdout,
                self._original_stderr,
            )
            header = f"Epoch: [{self.current_epoch}/{self.epochs}] Validation"
            val_str = (
                f"mAP50: {map50:.4f}  fitness: {fitness:.4f}  "
                f"precision: {precision:.4f}  recall: {recall:.4f}  "
                f"f1: {f1:.4f}  loss: {val_total_loss:.4f}  "
                f"box loss: {val_box_loss:.4f}  cls loss: {val_cls_loss:.4f}  "
                f"dfl loss: {val_dfl_loss:.4f}"
            )
            print(f"{header}  {val_str}")
            print()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            if self.ray_tune_callback:
                self.ray_tune_callback(epoch_metrics)
        except Exception as e:
            with open("validation_errors.log", "a") as f:
                f.write(f"Error logging validation metrics: {str(e)}\n")
            print(f"Error in _on_val_end: {e}")
