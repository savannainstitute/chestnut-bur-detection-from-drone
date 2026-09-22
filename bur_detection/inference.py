"""Tiled YOLO inference over full-canopy trees for bur detection."""

from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import random
from ultralytics import YOLO

from bur_detection.utils import (
    plot_ground_truth_vs_predictions,
    apply_nms,
    get_output_dir,
    boxes_to_numpy,
)
from bur_detection.dataset import CanopyTiler


class YOLOInference:
    """Run tiled YOLO inference over canopy trees and aggregate results.

    For each tree in the image selections, crops and masks the
    canopy, tiles it, runs the model on the tiles, reconstructs and
    NMS-deduplicates detections, and writes a per-tree CSV plus
    optional prediction plots.
    """

    def __init__(
        self,
        model_path,
        image_selections_path,
        conf_threshold,
        iou_threshold,
        plot_mode="subset",
        global_nms_iou=0.3,
        tile_batch_size=96,
        tile_size=224,
        overlap=0.2,
        outputs_dir="bur_detection/sample_data/training/outputs",
        output_dir=None,
    ):
        """Load the model and selections, and set up the output dir."""
        self.outputs_dir = outputs_dir
        self.model_path = self._get_model_path(model_path)
        print(f"\nLoading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.image_selection_path = Path(image_selections_path)
        self.selections = self._load_selections(self.image_selection_path)
        self.tiler = CanopyTiler(tile_size=tile_size, overlap=overlap)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.global_nms_iou = global_nms_iou
        self.tile_batch_size = tile_batch_size
        self.plot_mode = plot_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else get_output_dir(
                "bur_detection/sample_data/inference/outputs",
                "inference",
                self.timestamp,
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_trees_dir = self.output_dir / "preprocessed_trees"
        self.preprocessed_trees_dir.mkdir(exist_ok=True)
        self.results_data = []
        self.all_predictions = []
        self.results_df = None
        self.csv_path = None

    def _get_model_path(self, model_path):
        if model_path is not None:
            return Path(model_path)
        # Auto-detect the latest model under outputs/run_<ts>/{train,tune}/:
        # newest run first, final train preferred over tuning best.
        base = Path(self.outputs_dir)
        candidate_dirs = []
        for run in sorted(base.glob("run_*"), reverse=True):
            candidate_dirs += [run / "train", run / "tune"]
        candidate_dirs += sorted(
            base.glob("tuning_*"), reverse=True
        )  # legacy flat layout
        candidate_dirs += sorted(base.glob("training_*"), reverse=True)
        for d in candidate_dirs:
            if d.is_dir():
                model_files = sorted(d.glob("best_*.pt"))
                if model_files:
                    print(f"Using model: {model_files[0]}")
                    return model_files[0]
        raise FileNotFoundError(
            f"No model found under {base} (looked in run_*/train, "
            f"run_*/tune). Run training or tuning first."
        )

    def _load_selections(self, selections_path):
        selections_path = Path(selections_path)
        print(f"Loading canopy selections: {selections_path}")
        with open(selections_path, "r") as f:
            return json.load(f)

    def run(self):
        """Detect burs for every tree, then write CSV, plots, and a summary."""
        print(f"\nProcessing {len(self.selections)} trees...")
        for tree_id, tree_data in self.selections.items():
            image_path = Path(tree_data["image_path"])
            if not image_path.exists():
                print(f"Warning: Image not found for {tree_id}: {image_path}")
                continue
            print(f"\nProcessing tree {tree_id}...")
            polygon_coords = tree_data["polygon_coords"]
            cropped_canopy = self.tiler.crop_canopy_from_polygon(
                image_path, polygon_coords
            )
            canopy_save_path = (
                self.preprocessed_trees_dir / f"tree_{tree_id}.jpg"
            )
            Image.fromarray(cropped_canopy).save(canopy_save_path)
            tiles, tile_info = self.tiler.tile_image(cropped_canopy)
            print(f"  {len(tiles)} tiles from cropped canopy")
            tile_detections = []
            batch_size = max(1, int(self.tile_batch_size))
            for start in range(0, len(tiles), batch_size):
                batch_imgs = [
                    Image.fromarray(t)
                    for t in tiles[start : start + batch_size]
                ]
                preds = self.model.predict(
                    batch_imgs,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    # native tile size -- no resize
                    imgsz=self.tiler.tile_size,
                    verbose=False,
                )
                for pred in preds:
                    if pred.boxes is not None and len(pred.boxes) > 0:
                        tile_detections.append(
                            {
                                "boxes": boxes_to_numpy(pred.boxes.xyxy),
                                "confidences": boxes_to_numpy(pred.boxes.conf),
                                "labels": np.zeros(len(pred.boxes)),
                            }
                        )
                    else:
                        tile_detections.append(
                            {
                                "boxes": np.array([]),
                                "confidences": np.array([]),
                                "labels": np.array([]),
                            }
                        )
            all_detections = self.tiler.reconstruct_detections_core(
                tile_detections, tile_info
            )
            raw_count = len(all_detections)
            filtered_detections = apply_nms(
                all_detections, self.global_nms_iou
            )
            nms_count = len(filtered_detections)
            print(
                f"  Post-NMS detections: {nms_count} "
                f"(removed {raw_count - nms_count} duplicates)"
            )
            if nms_count > 0:
                avg_conf = np.mean(
                    [d["confidence"] for d in filtered_detections]
                )
                print(f"  Average confidence: {avg_conf:.3f}")
            else:
                avg_conf = 0.0
                print("  Average confidence: no detections")
            self.all_predictions.append(
                (canopy_save_path, filtered_detections)
            )
            detection_coords = str(
                [
                    (
                        float(d["box"][0]),
                        float(d["box"][1]),
                        float(d["box"][2]),
                        float(d["box"][3]),
                        float(d["confidence"]),
                    )
                    for d in filtered_detections
                ]
            )
            self.results_data.append(
                {
                    "tree_id": tree_id,
                    "image_path": str(image_path),
                    "total_detections": nms_count,
                    "avg_confidence": avg_conf,
                    "detection_coords": detection_coords,
                }
            )
        self.results_df = pd.DataFrame(self.results_data)
        self.csv_path = self.output_dir / "tree_bur_detections.csv"
        self.results_df.to_csv(self.csv_path, index=False)

        if self.plot_mode == "none":
            predictions_to_plot = []
        elif self.plot_mode == "subset":
            sample_size = min(15, len(self.all_predictions))
            predictions_to_plot = random.sample(
                self.all_predictions, sample_size
            )
            print(f"\nSaving plots for {sample_size} random trees...")
        else:  # 'all'
            predictions_to_plot = self.all_predictions
            print("\nSaving plots for all trees...")

        if predictions_to_plot:
            plot_dir = self.output_dir / "prediction_plots"
            plot_dir.mkdir(exist_ok=True)
            plot_ground_truth_vs_predictions(
                predictions=predictions_to_plot,
                labels_dir=None,  # No ground truth for unlabeled data
                original_images_dir=self.preprocessed_trees_dir,
                save_dir=plot_dir,
                conf_threshold=self.conf_threshold,
            )

        processed_trees = len(self.results_data)
        total_burs = (
            int(self.results_df["total_detections"].sum())
            if processed_trees > 0
            else 0
        )
        avg_burs_per_tree = (
            float(self.results_df["total_detections"].mean())
            if processed_trees > 0
            else 0.0
        )
        min_burs = (
            int(self.results_df["total_detections"].min())
            if processed_trees > 0
            else 0
        )
        max_burs = (
            int(self.results_df["total_detections"].max())
            if processed_trees > 0
            else 0
        )
        avg_confidence = (
            float(self.results_df["avg_confidence"].mean())
            if processed_trees > 0
            else 0.0
        )

        summary_text = f"""
        Bur Detection Summary
        {"=" * 50}
        Processed Trees: {processed_trees}
        Total Burs Detected: {total_burs}
        Average Burs per Tree: {avg_burs_per_tree}
        Min Burs: {min_burs}
        Max Burs: {max_burs}
        Overall Average Confidence: {avg_confidence:.3f}

        Model: {self.model_path.name}
        Confidence Threshold: {self.conf_threshold}
        IoU Threshold: {self.iou_threshold}

        Results saved to: {self.csv_path}
        Plots saved to: {self.output_dir / "prediction_plots"}
        {"=" * 50}
        """
        summary_path = self.output_dir / "detection_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary_text)
        print(summary_text)
