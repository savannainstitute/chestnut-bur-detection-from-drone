"""Tune YOLO hyperparameters with Ray Tune and Optuna search."""

import os
import sys
import csv
import tempfile
import pickle
from pathlib import Path
import time
import math
import shutil
import json
import random
import types
from typing import cast
from collections.abc import Iterable

import torch
from ray import tune
from ray.tune import Callback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.air import session
from ultralytics import YOLO

from bur_detection.training import YOLOTrainer
from bur_detection.utils import (
    set_seed,
    is_notebook,
    convert_tuning_space,
    evaluate_test_set,
    plot_ground_truth_vs_predictions,
    compute_composite_objective,
    analyze_ray_results,
    pick_device,
)


def _should_report_on_trial_start(self, trials, done=False):
    """Trigger a status print once per trial start and once when done.

    A new trial's table already shows the previous trial's completed
    result, so this avoids a redundant second print at each trial
    boundary, and skips the per-epoch noise.
    """
    started = sum(1 for t in trials if str(t.status) != "PENDING")
    if done or started != getattr(self, "_last_started", 0):
        self._last_started = started
        return True
    return False


def _format_best_result_chunk(trial_name, m):
    """Format a compact summary of a trial's best epoch.

    Printed at completion in place of Ray's per-epoch result dump
    (which is silenced via verbose=1). `m` is the best-epoch metrics
    snapshot.
    """

    def g(k):
        return m.get(k, 0.0)

    return "\n".join(
        [
            "=" * 80,
            f"Best result for {trial_name} -- epoch "
            f"{int(g('epoch'))} (step {int(g('step'))}) "
            f"| model {m.get('model', '?')}",
            f"  objective    : {g('objective'):.6g}  "
            f"(best {m.get('objective_best', g('objective')):.6g})",
            f"  val_loss     : {g('val_loss'):.4f}   "
            f"box {g('val_box_loss'):.4f}  "
            f"cls {g('val_cls_loss'):.4f}  dfl {g('val_dfl_loss'):.4f}",
            f"  val_metrics  : precision {g('val_precision'):.4f}  "
            f"recall {g('val_recall'):.4f}  f1 {g('val_f1'):.4f}  "
            f"mAP50 {g('val_mAP50'):.4f}  fitness {g('val_fitness'):.4f}",
            f"  train_loss   : {g('train_loss'):.4f}   "
            f"box {g('train_box_loss'):.4f}  "
            f"cls {g('train_cls_loss'):.4f}  "
            f"dfl {g('train_dfl_loss'):.4f}   lr {g('lr'):.3e}",
            "=" * 80,
        ]
    )


class _CheckpointReaper(Callback):
    """Keep only the top-N trials' checkpoints, by composite objective.

    Reaps the checkpoints of finished trials ranked below the cut.
    Never touches a running trial.
    """

    def __init__(self, keep_top_n):
        self.keep_top_n = max(1, int(keep_top_n))
        self._best_obj = {}  # trial_id -> best (min) objective seen
        self._trial_path = {}  # trial_id -> trial dir

    def on_trial_result(self, iteration, trials, trial, result, **info):
        self._trial_path[trial.trial_id] = trial.path
        try:
            obj = float(result.get("objective"))
        except (TypeError, ValueError):
            return
        if math.isfinite(obj) and obj < self._best_obj.get(
            trial.trial_id, float("inf")
        ):
            self._best_obj[trial.trial_id] = obj

    def on_trial_complete(self, iteration, trials, trial, **info):
        self._trial_path[trial.trial_id] = trial.path
        self._best_obj.setdefault(
            trial.trial_id, float("inf")
        )  # objective-less trials rank last

        # Reap the checkpoints of every finished trial ranked below the top N.
        ranked = sorted(self._best_obj, key=lambda tid: self._best_obj[tid])
        for tid in ranked[self.keep_top_n :]:
            tpath = self._trial_path.get(tid)
            if not tpath:
                continue
            for ckpt in Path(tpath).glob("checkpoint_*"):
                shutil.rmtree(ckpt, ignore_errors=True)


class YOLOTuner:
    """Run YOLO hyperparameter tuning with Ray Tune and Optuna search."""

    def __init__(
        self,
        num_samples=50,
        max_concurrent_trials=1,
        yolo_data_dir=None,
        training_steps=None,
        points_to_evaluate=None,
        tuning_space=None,
        conf_threshold=0.5,
        iou_threshold=0.45,
        plot_mode="subset",
        score_weights=None,
        analysis_enabled=True,
        analysis_top_n=10,
        outputs_dir="bur_detection/sample_data/training/outputs",
        warmstart=False,
        tal_topk=None,
        step_transition_warmup_epochs=10.0,
        keep_top_n=5,
        registry_path=None,
    ):
        """Store the tuning configuration.

        Raises ValueError if training_steps or tuning_space is empty --
        both are required to run a search.
        """
        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.yolo_data_dir = (
            str(Path(yolo_data_dir).absolute()) if yolo_data_dir else None
        )
        self.training_steps = (
            training_steps if training_steps is not None else []
        )
        self.points_to_evaluate = (
            points_to_evaluate if points_to_evaluate is not None else []
        )
        self.tuning_space = tuning_space if tuning_space is not None else {}
        self.plot_mode = plot_mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.score_weights = score_weights or {
            "loss": 0.45,
            "f1": 0.35,
            "map50": 0.20,
        }
        self.analysis_enabled = analysis_enabled
        self.analysis_top_n = analysis_top_n
        self.outputs_dir = outputs_dir
        self.warmstart = warmstart
        self.tal_topk = tal_topk
        self.step_transition_warmup_epochs = step_transition_warmup_epochs
        self.keep_top_n = keep_top_n
        self.registry_path = registry_path

        self.results = None
        self.best_trial = None
        self.best_trial_preds = None
        self.best_output_dir = None
        self.best_model_path = None
        self.best_trial_config = None

        if not self.training_steps:
            raise ValueError(
                "training_steps schedule must be provided "
                "for hyperparameter tuning."
            )
        if not self.tuning_space:
            raise ValueError(
                "tuning_space config must be provided "
                "for hyperparameter search."
            )

    def train_yolo_with_ray(self, config):
        """Train one Ray Tune trial, resuming from a checkpoint if given.

        This is the Ray Tune trainable: it builds a YOLOTrainer, reports
        per-epoch metrics back to Ray via ray_tune_callback, and prints a
        best-epoch summary chunk when the trial ends.
        """
        from ultralytics.utils import SETTINGS

        SETTINGS["raytune"] = False

        os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
        os.chdir(str(Path(__file__).parent.parent))  # handle relative paths
        set_seed(666)

        checkpoint = tune.get_checkpoint()
        start_step = 0
        step_epochs_completed = 0
        total_epochs_so_far = 0
        yolo_checkpoint_path = None
        objective_best_so_far = float(
            "inf"
        )  # per-trial running best (best-vs-last)
        best_metrics = {}  # snapshot of the best epoch's reported metrics

        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                yolo_model_path = Path(checkpoint_dir) / "yolo_model.pt"

                if data_path.exists():
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    # current_step is 1-indexed;
                    # start_step is a 0-indexed step_idx.
                    start_step = max(0, checkpoint_state["current_step"] - 1)
                    step_epochs_completed = checkpoint_state[
                        "step_epochs_completed"
                    ]
                    total_epochs_so_far = checkpoint_state[
                        "total_epochs_so_far"
                    ]
                    objective_best_so_far = checkpoint_state.get(
                        "objective_best", float("inf")
                    )
                    best_metrics = checkpoint_state.get("best_metrics", {})

                if yolo_model_path.exists():
                    yolo_checkpoint_path = str(yolo_model_path)

        trial_dir = Path(session.get_trial_dir())
        # Ray's artifact staging removes large .pt files mid-trial on
        # Windows, breaking the step handoff/reload.
        output_dir = (
            Path(self.outputs_dir).resolve()
            / "_ultralytics_work"
            / trial_dir.name
        )
        # Point Ultralytics at a stable per-trial dir instead.
        # Resume + final selection ride the Ray checkpoints, not this dir.
        output_dir.mkdir(parents=True, exist_ok=True)

        def ray_tune_callback(metrics):
            nonlocal objective_best_so_far, best_metrics
            current_step = getattr(trainer, "current_step", 1)
            step_epoch = trainer.current_epoch
            total_epochs = (
                getattr(trainer, "total_epochs_so_far", 0) + step_epoch
            )

            report_metrics = {
                "model": Path(
                    config["model_size"]
                ).name,  # basename for a readable table column
                "training_iteration": total_epochs,
                "step": current_step,
                "epoch": metrics["epoch"]
                if "epoch" in metrics
                else total_epochs,
                "train_loss": metrics["train_loss"]
                if "train_loss" in metrics
                else 0.0,
                "train_box_loss": metrics["train_box_loss"]
                if "train_box_loss" in metrics
                else 0.0,
                "train_cls_loss": metrics["train_cls_loss"]
                if "train_cls_loss" in metrics
                else 0.0,
                "train_dfl_loss": metrics["train_dfl_loss"]
                if "train_dfl_loss" in metrics
                else 0.0,
                "lr": metrics["lr"] if "lr" in metrics else 0.0,
                "val_precision": metrics["val_precision"]
                if "val_precision" in metrics
                else 0.0,
                "val_recall": metrics["val_recall"]
                if "val_recall" in metrics
                else 0.0,
                "val_f1": metrics["val_f1"] if "val_f1" in metrics else 0.0,
                "val_mAP50": metrics["val_mAP50"]
                if "val_mAP50" in metrics
                else 0.0,
                "val_fitness": metrics["val_fitness"]
                if "val_fitness" in metrics
                else 0.0,
                "val_loss": metrics["val_loss"]
                if "val_loss" in metrics
                else 0.0,
                "val_box_loss": metrics["val_box_loss"]
                if "val_box_loss" in metrics
                else 0.0,
                "val_cls_loss": metrics["val_cls_loss"]
                if "val_cls_loss" in metrics
                else 0.0,
                "val_dfl_loss": metrics["val_dfl_loss"]
                if "val_dfl_loss" in metrics
                else 0.0,
            }

            report_metrics["objective"] = compute_composite_objective(
                report_metrics["val_loss"],
                report_metrics["val_f1"],
                report_metrics["val_mAP50"],
                self.score_weights,
            )

            # Hold reporting during the post-unfreeze grace (steps > 1,
            # first step_patience epochs).
            step_patience = int(metrics.get("step_patience", 0) or 0)
            step_grace_active = (
                current_step > 1 and step_epoch <= step_patience
            )
            # So ASHA can't prune a trial on the transition loss spike.
            if step_grace_active:
                return

            # Track running-min objective for ASHA/Optuna; snapshot the
            # best epoch for the terminal report.
            if report_metrics["objective"] < objective_best_so_far:
                objective_best_so_far = report_metrics["objective"]
                best_metrics = dict(
                    report_metrics
                )  # full snapshot of the best epoch
                best_metrics["objective_best"] = objective_best_so_far
            # Per-epoch rows keep current metrics for the live reporter.
            report_metrics["objective_best"] = objective_best_so_far

            checkpoint_data = {
                "current_step": current_step,
                "step_epochs_completed": step_epoch,
                "total_epochs_so_far": total_epochs,
                "training_iteration": total_epochs,
                "objective_best": objective_best_so_far,
                "best_metrics": best_metrics,
            }

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)

                if hasattr(trainer, "model") and trainer.model:
                    yolo_model_path = Path(checkpoint_dir) / "yolo_model.pt"
                    try:
                        trainer.model.save(str(yolo_model_path))
                    except Exception:
                        pass

                try:
                    tune.report(
                        report_metrics,
                        checkpoint=tune.Checkpoint.from_directory(
                            checkpoint_dir
                        ),
                    )
                except Exception:
                    pass

        trainer = YOLOTrainer(
            model_size=config["model_size"],
            prints_per_epoch=5,
            ray_tune_callback=ray_tune_callback,
            training_steps=self.training_steps,
            warmstart=self.warmstart,
            tal_topk=self.tal_topk,
            step_transition_warmup_epochs=self.step_transition_warmup_epochs,
        )

        # Wrap training so the best-epoch chunk
        # prints once at the end of every trial.
        try:
            # ASHA prunes by raising sys.exit(0) inside
            # tune.report() (SystemExit);
            if checkpoint:
                self._resume_training(
                    trainer,
                    self.yolo_data_dir,
                    config,
                    start_step,
                    step_epochs_completed,
                    total_epochs_so_far,
                    yolo_checkpoint_path,
                    ray_tune_callback,
                    output_dir,
                )
            else:
                trainer.train(
                    self.yolo_data_dir, config=config, output_dir=output_dir
                )

            # Terminal report (normal completion only): flip the reporter row
            # to the best epoch.
            if best_metrics:
                try:
                    # No checkpoint (don't shadow the selection checkpoint);
                    final_iter = getattr(
                        trainer, "total_epochs_so_far", 0
                    ) + getattr(trainer, "current_epoch", 0)
                    # training_iteration held so the epoch-sync reporter
                    # gate's running sum doesn't drop at termination.
                    tune.report(
                        {
                            **best_metrics,
                            "objective": objective_best_so_far,
                            "objective_best": objective_best_so_far,
                            "training_iteration": final_iter,
                        }
                    )
                    # On a prune the session is already exiting, so this is
                    # skipped -- the pruned row keeps its last epoch.
                except Exception:
                    pass
        # it propagates up through ultralytics (whose handlers are all
        # `except Exception`) and still runs `finally`.
        finally:
            # One result chunk per trial on the best epoch (not the last)
            if best_metrics:
                # -- runs whether the trial completed or was pruned (the
                # pruning SystemExit still triggers this).
                print(_format_best_result_chunk(trial_dir.name, best_metrics))

        # The stable Ultralytics work dir is intermediate only (the model for
        # resume and final selection rides the Ray checkpoints), so drop it on
        shutil.rmtree(output_dir, ignore_errors=True)
        # completion to bound disk growth. A pruned trial's SystemExit skips
        # this, so pruned/errored trials keep theirs for debugging.
        return {}

    def _resume_training(
        self,
        trainer,
        yolo_data_dir,
        config,
        start_step,
        step_epochs_completed,
        total_epochs_so_far,
        yolo_checkpoint_path,
        ray_tune_callback,
        output_dir=None,
    ):
        """Resume training from a checkpoint for this trial's trainer."""
        trainer.ray_tune_callback = ray_tune_callback
        trainer._original_stdout = sys.stdout
        trainer._original_stderr = sys.stderr

        if yolo_checkpoint_path and os.path.exists(yolo_checkpoint_path):
            trainer.model = YOLO(yolo_checkpoint_path)

        resume_config = {
            **config,
            "_resume_from_step": start_step,
            "_resume_step_epochs": step_epochs_completed,
            "_resume_total_epochs": total_epochs_so_far,
        }

        return trainer.train(
            yolo_data_dir, config=resume_config, output_dir=output_dir
        )

    def run(self, run_name=None):
        """Run hyperparameter tuning with Ray Tune.

        Builds the ASHA scheduler and Optuna search, fits the Ray
        Tuner, then finds the best trial by composite objective, saves
        its checkpoint and config, and evaluates it on the test set.
        """
        # Opt into the legacy reporter so our metric_columns/parameter_columns
        # are honored.
        os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = f"YOLO_Optuna_{timestamp}"
        Path(self.outputs_dir).mkdir(
            parents=True, exist_ok=True
        )  # Ray storage_path = run_<ts>/tune/

        param_space = convert_tuning_space(self.tuning_space)

        def trial_dirname_creator(trial):
            return f"trial_{trial.trial_id}"

        max_t = sum(step["max_epochs"] for step in self.training_steps)
        # Step 1's patience is the ASHA grace window; clamp to max_t to satisfy
        # grace <= max_t.
        first_patience = self.training_steps[0].get("patience", max_t)
        grace_period = max(1, min(first_patience, max_t))
        asha_scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=max_t,
            grace_period=grace_period,
            # canonical successive-halving factor (keep top third)
            reduction_factor=3,
        )

        optuna_search = OptunaSearch(
            metric="objective_best",
            mode="min",
            points_to_evaluate=self.points_to_evaluate,
        )
        optuna_search = ConcurrencyLimiter(
            optuna_search, max_concurrent=self.max_concurrent_trials
        )

        metric_columns = [
            # `model` (basename of model_size) is reported as a metric so the
            # table shows the readable arch name (yolo11s-p2.yaml) rather than
            "model",
            # the full config path.
            "step",
            # Current epoch for running trials; best epoch for completed ones
            # (terminal report).
            "epoch",
            "val_loss",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_mAP50",
            "objective",
            "objective_best",
        ]
        # Report every searched hyperparameter, in config order -- the table
        # mirrors the full trial config instead of a hand-picked subset that
        parameter_columns = [k for k in param_space if k != "model_size"]
        # silently drifts as the search space changes. model_size is surfaced
        # via the `model` metric column above (basename only), so it's dropped
        reporter_kwargs = dict(
            # here to keep the long path out of the table. Keys come straight
            # from param_space, so each column is present in every trial's
            metric_columns=metric_columns,
            # config (no best-trial KeyError).
            parameter_columns=parameter_columns,
            max_progress_rows=50,
            # values are now short (model basename ~15 chars, the rest numeric)
            max_column_length=20,
            # verbose=1 (below) silences Ray's per-epoch result dump but
            # also stops rendering the trial table;
            print_intermediate_tables=True,
            # force the table back on so the live grid still prints.
            sort_by_metric=True,
        )
        reporter = (
            tune.JupyterNotebookReporter(**reporter_kwargs)
            if is_notebook()
            else tune.CLIReporter(**reporter_kwargs)
        )
        reporter.should_report = types.MethodType(
            _should_report_on_trial_start, reporter
        )

        max_concurrent = self.max_concurrent_trials
        available_gpus = torch.cuda.device_count()
        available_cpus = os.cpu_count() or 2

        if available_gpus > 0:
            gpus_per_trial = available_gpus / max_concurrent
            cpus_per_trial = max(1, available_cpus // max_concurrent)
            resources = {
                "cpu": float(cpus_per_trial),
                "gpu": float(gpus_per_trial),
            }
        else:
            cpus_per_trial = max(1, available_cpus // max_concurrent)
            resources = {"cpu": float(cpus_per_trial)}
            # MPS lands here too, but trains on GPU --
            # only warn on real CPU-only
            if pick_device() == "cpu":
                print("GPU not available. Tuning with CPU only.")

        tuner = tune.Tuner(
            tune.with_resources(self.train_yolo_with_ray, resources=resources),
            tune_config=tune.TuneConfig(
                mode="min",
                metric="objective_best",
                search_alg=optuna_search,
                scheduler=asha_scheduler,
                num_samples=self.num_samples,
                trial_dirname_creator=trial_dirname_creator,
            ),
            run_config=tune.RunConfig(
                name=run_name,
                progress_reporter=reporter,
                # Silence Ray's per-epoch "Result for <trial>" dump; the
                # trial-end best-epoch chunk we print in train_yolo_with_ray
                verbose=1,
                # replaces it. The grid is kept alive via the reporter's
                # print_intermediate_tables.
                storage_path=str(
                    Path(self.outputs_dir).resolve()
                ),  # trials nest under run_<ts>/tune/
                # No retries: the in-memory step optimizer/LR handoff isn't
                # checkpointed.
                failure_config=tune.FailureConfig(max_failures=0),
                # Keep only each trial's best-objective checkpoint (what
                # winner-selection reads).
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="objective",
                    checkpoint_score_order="min",
                ),
                # ...and only the top-N trials' checkpoints across the run.
                callbacks=[_CheckpointReaper(keep_top_n=self.keep_top_n)],
            ),
            param_space=param_space,
        )

        self.results = tuner.fit()

        # Drop the Ultralytics scratch dir now that every trial is done.
        shutil.rmtree(
            Path(self.outputs_dir).resolve() / "_ultralytics_work",
            ignore_errors=True,
        )

        if self.analysis_enabled:
            try:
                analyze_ray_results(
                    self.results.experiment_path, top_n=self.analysis_top_n
                )
            except Exception as e:
                print(f"Tuning analysis skipped: {e}")

        try:
            self.best_output_dir = Path(self.outputs_dir)
            self.best_output_dir.mkdir(parents=True, exist_ok=True)
            results_df = self.results.get_dataframe(
                filter_metric="objective", filter_mode="min"
            )
            results_df.to_csv(
                self.best_output_dir / "all_tuning_history.csv", index=False
            )

            # Find the best trial (lowest composite objective of any epoch)
            best_overall_trial = min(
                list(
                    cast(Iterable, self.results)
                ),  # ResultGrid is iterable at runtime
                key=lambda trial: trial.metrics_dataframe["objective"].min(),
            )
            best_overall_trial.metrics_dataframe.to_csv(
                self.best_output_dir / "best_trial_training_history.csv",
                index=False,
            )

            # Find best epoch weights by composite objective
            best_obj_idx = best_overall_trial.metrics_dataframe[
                "objective"
            ].idxmin()
            best_obj_row = best_overall_trial.metrics_dataframe.loc[
                best_obj_idx
            ]
            checkpoint_dir_name = best_obj_row["checkpoint_dir_name"]
            model_size = Path(
                str(best_obj_row.get("config/model_size", "yolo_model"))
            ).stem

            best_trial_dir = Path(best_overall_trial.path)
            checkpoint_dir = best_trial_dir / checkpoint_dir_name
            yolo_model_path = checkpoint_dir / "yolo_model.pt"
            # Fall back to the trial's surviving checkpoint if a tie pruned the
            # looked-up one.
            if not yolo_model_path.exists():
                yolo_model_path = next(
                    (
                        c / "yolo_model.pt"
                        for c in sorted(best_trial_dir.glob("checkpoint_*"))
                        if (c / "yolo_model.pt").exists()
                    ),
                    yolo_model_path,
                )

            self.best_model_path = (
                self.best_output_dir / f"best_{model_size}_model.pt"
            )
            shutil.copy2(yolo_model_path, self.best_model_path)

            best_config_path = best_trial_dir / "params.json"
            if best_config_path.exists():
                shutil.copy2(
                    best_config_path,
                    self.best_output_dir / "best_trial_config.json",
                )
                with open(best_config_path, "r") as f:
                    self.best_trial_config = json.load(f)
            else:
                self.best_trial_config = {}

            self.best_trial = {
                "path": str(best_trial_dir),
                "config": self.best_trial_config,
                "metrics_dataframe": best_overall_trial.metrics_dataframe,
                "model_path": str(self.best_model_path),
                "output_dir": str(self.best_output_dir),
            }

            self._append_model_registry(best_obj_row, model_size)

            yolo_data_dir = cast(
                str, self.yolo_data_dir
            )  # always set for a real tuning run
            dataset_yaml = Path(yolo_data_dir) / "dataset.yml"
            if dataset_yaml.exists():
                self.best_trial_preds = evaluate_test_set(
                    model_path=self.best_model_path,
                    training_dir=Path(yolo_data_dir),
                    output_dir=self.best_output_dir,
                    plot_mode=self.plot_mode,
                    conf_threshold=self.conf_threshold,
                    iou_threshold=self.iou_threshold,
                )
            if self.best_trial_preds:
                images_dir = Path(yolo_data_dir) / "images"
                labels_dir = Path(yolo_data_dir) / "labels"
                plot_dir = Path(self.best_output_dir) / "prediction_plots"

                if self.plot_mode == "none":
                    predictions_to_plot = []
                elif self.plot_mode == "subset":
                    sample_size = min(15, len(self.best_trial_preds))
                    predictions_to_plot = random.sample(
                        self.best_trial_preds, sample_size
                    )
                else:  # 'all'
                    predictions_to_plot = self.best_trial_preds

                if predictions_to_plot:
                    plot_ground_truth_vs_predictions(
                        predictions=predictions_to_plot,
                        labels_dir=labels_dir,
                        original_images_dir=images_dir,
                        save_dir=plot_dir,
                        conf_threshold=self.conf_threshold,
                    )

        except Exception as e:
            print(f"Error in post-processing best trial: {e}")
            self.best_trial = None
            self.best_trial_preds = None

    def _append_model_registry(self, best_obj_row, model_size):
        """Append one row per tuning winner to the model registry CSV.

        Writes to registry_path (a cross-run CSV) and never raises -- a
        registry write must not abort an otherwise-finished tuning run.
        """
        if not self.registry_path:
            return
        try:
            cfg = self.best_trial_config or {}
            run_id = Path(self.outputs_dir).parent.name  # run_<ts>
            row = {
                "run_id": run_id,
                "model_size": model_size,
                "objective": round(
                    float(best_obj_row.get("objective", float("nan"))), 6
                ),
                "val_f1": round(float(best_obj_row.get("val_f1", 0.0)), 6),
                "val_mAP50": round(
                    float(best_obj_row.get("val_mAP50", 0.0)), 6
                ),
                "val_precision": round(
                    float(best_obj_row.get("val_precision", 0.0)), 6
                ),
                "val_recall": round(
                    float(best_obj_row.get("val_recall", 0.0)), 6
                ),
                "val_loss": round(float(best_obj_row.get("val_loss", 0.0)), 6),
                "best_epoch": int(best_obj_row.get("epoch", 0)),
                "num_trials": self.num_samples,
                "optimizer": cfg.get("optimizer", ""),
                "lr0": cfg.get("lr0", ""),
                "lrf": cfg.get("lrf", ""),
                "box_gain": cfg.get("box_gain", ""),
                "dfl_gain": cfg.get("dfl_gain", ""),
                "model_path": str(self.best_model_path),
                "config_json": str(
                    Path(self.best_output_dir) / "best_trial_config.json"
                ),
                "tune_dir": str(self.best_output_dir),
            }
            registry = Path(self.registry_path)
            registry.parent.mkdir(parents=True, exist_ok=True)
            write_header = not registry.exists()
            with open(registry, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(
                f"Model registry updated -> {registry} (run {run_id}, "
                f"objective={row['objective']}, model={model_size})"
            )
        except Exception as e:
            print(f"Model registry update skipped: {e}")
