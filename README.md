# chestnut-bur-detection-from-drone

Pipeline for estimating chestnut tree bur yield from drone imagery.

---

## Overview

This repository implements an end-to-end pipeline for estimating chestnut (*Castanea* spp.) bur yield at the individual-tree level using drone imagery and YOLO object detection. The workflow proceeds from raw flight data to per-tree bur counts through four sequential modules:

1. **Flight Reconstruction** — process raw drone images into georeferenced 3D products (DSM, DTM, CHM, orthomosaic) using Agisoft Metashape
2. **Canopy Segmentation** — delineate individual tree canopies from the Canopy Height Model using marker-controlled watershed segmentation
3. **Image Selection** — back-project each canopy polygon onto the raw drone image collection and select the highest-quality image per tree
4. **Bur Detection** — detect and count burs in each canopy image using YOLO; includes training, hyperparameter tuning, and inference modes

```
Raw drone images
      │
      ▼
[Flight Reconstruction]  ──►  DSM / DTM / CHM / Orthomosaic / Point Cloud
      │
      ▼
[Canopy Segmentation]  ──►  Per-tree canopy polygons + refined treetops
      │
      ▼
[Image Selection]  ──►  Best drone image per canopy (JSON)
      │
      ▼
[Bur Detection]  ──►  Per-tree bur count CSV
```

---

## Hardware Requirements

The full pipeline was developed and tested on Windows with the following hardware:

- **OS:** Windows (Metashape Python API is Windows-only)
- **GPU:** 8 GB VRAM minimum; 16+ GB recommended for Metashape depth map generation and YOLO training
- **RAM:** 128 GB recommended (Metashape depth map generation is memory-intensive; the reconstruction script monitors available RAM and can subdivide tasks automatically)
- **CPU:** 24-core recommended
- **Disk:** ~256 GB free space per orchard dataset (raw images + Metashape project files)

---

## Sample Data

A complete sample dataset for one chestnut orchard flight is available on Google Drive.

**Download (~42 GB):**  
[Download sample data (Google Drive)](https://drive.google.com/file/d/13qJbHO3ZU8EeesVM2tbPHkUUXSIAS_aN/view?usp=sharing)

The ZIP unpacks to a `chestnut_bur_sample_data/` folder containing one subfolder per module, each with its own `sample_data/`. After extracting, move each module's `sample_data/` (e.g. `chestnut_bur_sample_data/bur_detection/sample_data/`) into the matching module directory in your clone, so paths resolve as e.g., `bur_detection/sample_data/…`.

---

## Environment Setup

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

2. Clone this repository.

3. Create and activate the conda environment. This installs Python 3.13, the
   Agisoft Metashape wheel included in the repo, and PyTorch with CUDA 13.0:
   ```powershell
   conda env create -f bur-detection.yml
   conda activate bur-detection
   ```

4. Activate a Metashape Professional license. See [Agisoft documentation](https://agisoft.freshdesk.com/support/solutions/articles/31000153171-how-to-activate-metashape-license) for license activation instructions.

Key dependencies (from `bur-detection.yml`):

| Package | Purpose |
|---------|---------|
| Metashape | Flight reconstruction; its wheel supports Python up to 3.13 |
| gdal | Geospatial I/O |
| geopandas | Vector geometry |
| rasterio | Raster I/O |
| scikit-image | Image processing, watershed |
| shapely | Polygon operations |
| psutil | Memory estimation (Metashape subdivision) |
| easyidp | 3D back-projection, Metashape wrapper |
| torch, torchvision | CUDA 13.0 training and inference |
| ultralytics | YOLO training and inference |
| ray[tune] | Distributed hyperparameter tuning |
| optuna | Bayesian hyperparameter search |
| pyexiv2 | XMP metadata reading (DJI drone tags) |

---

## Repository Structure

```
chestnut-bur-detection-from-drone/
├── bur-detection.yml                  # Conda environment
├── yolo11{n,s,m,l}.pt, yolov8{n,s,m,l}.pt  # YOLO base weights (warm-start pool for tune/train)
│
├── flight_reconstruction/
│   ├── reconstruction.py               # Metashape automation pipeline
│   ├── utils.py                        # GPS, export, filtering utilities
│   ├── config.yml                      # Reconstruction parameters
│   ├── metashape-2.3.1-...-win_amd64.whl
│   └── sample_data/20230823_Orchard4/
│       ├── DJI_*.JPG                   # Raw drone images (DJI Mavic 3M)
│       ├── DJI_*_PPKNAV.nav/.obs/.bin  # PPK GNSS navigation files
│       ├── DJI_*_Timestamp.MRK
│       └── outputs/                    # Reconstruction products (see Step 1)
│
├── canopy_segmentation/
│   ├── segmentation.py                 # Watershed segmentation
│   └── sample_data/
│       ├── inputs/                     # Tree markers + boundary shapefiles
│       └── outputs/                    # Canopy polygons + treetops shapefiles
│
├── image_selection/
│   ├── canopy_to_image.py              # Best-image selection per canopy
│   └── sample_data/outputs/
│       └── best_image_selections.json
│
└── bur_detection/
    ├── detection.py                    # Entry point: preprocess / train / tune / inference
    ├── training.py                     # Multi-step progressive YOLO training
    ├── tuning.py                       # Ray Tune + Optuna hyperparameter search
    ├── inference.py                    # Tile-and-detect inference pipeline
    ├── dataset.py                      # Group-aware splitting, canopy tiling, polygon tiler
    ├── utils.py                        # NMS, evaluation, plotting, metrics
    ├── config.yml                      # Preprocess, training, tuning, and inference config
    ├── model_cfgs/                     # YOLO11 P2-head model YAML
    └── sample_data/
        ├── training/full_canopy/       # Source per-tree canopy images + polygon bur & canopy labels
        ├── training/tiled/             # YOLO detection tiles (built from full_canopy by --mode preprocess)
        └── training/outputs/           # One run_<timestamp>/ per invocation (created at runtime; not in download)
```

---

## Step 1: Flight Reconstruction

**Script:** `flight_reconstruction/reconstruction.py`  
**Config:** `flight_reconstruction/config.yml`

### What It Does

Automates Agisoft Metashape to generate georeferenced 3D products from raw drone images:

1. Loads images, computes per-image quality scores, and removes images below threshold (default 0.70)
2. Matches tie points across images, aligns cameras, and optimizes camera intrinsics and extrinsics using RTK/GPS reference coordinates
3. Applies USGS-recommended three-stage tie point filtering: reconstruction uncertainty → projection accuracy → reprojection error (percentile-based thresholds)
4. Builds dense depth maps and a colored, confidence-weighted point cloud
5. Classifies ground points (angle/distance/cell-size thresholds) and generates DSM, DTM, and CHM (DSM − DTM)
6. Produces an orthorectified mosaic with mosaic blending and ghosting filter
7. Exports all products as GeoTIFF/LAS with UTM projection auto-detected from the first image's GPS coordinates

Each processing step checks whether it already completed before running; the pipeline is **resumable** from any point by re-running the same command. GPU selection automatically restricts to devices with ≥8 GB VRAM and disables CPU during GPU-intensive steps to reduce memory fragmentation.

### Inputs

| Input | Path | Notes |
|-------|------|-------|
| Raw images | `<folder>/*.JPG` | DJI Mavic 3M; RTK accuracy read from XMP |
| PPK GNSS files | `<folder>/*.nav`, `*.obs`, `*.bin`, `*.MRK` | DJI PPK navigation |
| Config | `flight_reconstruction/config.yml` | See config reference below |

### Outputs

All written to `<folder>/outputs/`:

| File | Description |
|------|-------------|
| `project_<name>.psx` | Metashape project file (resumable) |
| `<name>_dsm.tif` | Digital Surface Model (GeoTIFF, UTM) |
| `<name>_dtm.tif` | Digital Terrain Model (ground points only), DSM grid |
| `<name>_chm.tif` | Canopy Height Model = DSM − DTM, DSM grid |
| `<name>_orthomosaic.tif` | Georeferenced aerial image (GeoTIFF) |
| `<name>_point_cloud.las` | Dense point cloud (RGB + confidence) |
| `<name>_camera_positions.txt` | Camera OPK exterior orientations |
| `<name>_report.pdf` | Metashape processing report |

### Usage

```powershell
conda activate bur-detection
$env:METASHAPE_LICENSE_KEY="XXXXX-XXXXX-XXXXX-XXXXX-XXXXX"

python -m flight_reconstruction.reconstruction `
    --config "flight_reconstruction/config.yml" `
    --folder "flight_reconstruction/sample_data/20230823_Orchard4"
```

### Sensor profiles

| Profile | Selected by | Overrides |
|---------|-------------|-----------|
| baseline / `m3m` | EXIF model `M3M` | none |
| `p1` | EXIF model `ZenmuseP1` | `sensor_profiles.p1` in `config.yml` |

### Key Config Parameters

| Key | Default | Description |
|-----|---------|-------------|
| `sensor` | `auto` | Sensor profile: `auto`, `m3m`, `p1` |
| `gps.use_rtk` | `true` | Read RTK accuracy from DJI XMP metadata |
| `image_quality.quality_threshold` | `0.70` | Discard images below this Metashape quality score (0–1) |
| `photo_matching.downscale` | `1` | Matching resolution; 1 = full resolution |
| `photo_matching.keypoint_limit` | `80000` | Max keypoints per image |
| `camera.align.retry_unaligned` | `false` | Re-run alignment on cameras left unaligned |
| `tie_point_filtering.reconstruction_uncertainty.percentile` | `20` | Remove worst 20% by reconstruction uncertainty |
| `tie_point_filtering.projection_accuracy.percentile` | `30` | Remove worst 30% by projection accuracy |
| `tie_point_filtering.reprojection_error.percentile` | `5` | Remove worst 5% by reprojection error |
| `boundary.enabled` / `boundary.buffer_m` | `false` / `12` | Buffered camera hull used to clip raster exports |
| `depth_maps.downscale` | `2` | Depth map resolution; 2 = half resolution |
| `depth_maps.filter_mode` | `disabled` | Depth filtering: `mild`, `moderate`, `aggressive`, or `disabled` |
| `point_cloud.confidence_filter.{enabled,min_confidence}` | `false` / `2` | Remove points below the confidence threshold |
| `classify_ground_points.max_angle` | `15.0` | Maximum slope angle (°) for ground classification |
| `model.surface_type` / `model.interpolation` | `arbitrary` / `enabled` | Mesh surface type and interpolation |
| `dem.source` | `model` | DSM source: `model`, `point_cloud`, or `depth_maps` |
| `dem.resolution` | `0` | DSM resolution; `0` = auto; DTM and CHM use the DSM grid |
| `orthomosaic.surface` | `model` | Orthorectification surface: `model` or `dem` |
| `orthomosaic.blending_mode` | `mosaic` | Blending algorithm for orthomosaic |

### Limitations

- Requires Agisoft Metashape Professional license (proprietary; not open-source)
- Windows-only (Metashape Python API)
- Assumes DJI drone with standard XMP tags for RTK accuracy; other manufacturers use different tag names
- Single-chunk processing; multi-chunk workflows are not automated
- `subdivide_task` is `false` by default in all config sections; enable for datasets that exceed available RAM

---

## Step 2: Canopy Segmentation

**Script:** `canopy_segmentation/segmentation.py`

### What It Does

Segments individual tree canopies from the CHM using **marker-controlled watershed segmentation** with an adaptive proximity penalty:

1. Loads CHM raster and tree marker points (manually digitized from imagery or collected with RTK GNSS)
2. Refines each marker to the local CHM maximum within a buffer radius (corrects imprecise field placement)
3. Constructs a watershed cost surface from: inverted CHM height, CHM gradient magnitude, and a proximity penalty that scales each tree's boundary distance by its height relative to neighboring trees — preventing over-segmentation in dense stands
4. Runs scikit-image watershed with each refined marker as a labeled seed
5. Removes segments smaller than 5 m² and removes any segments that clip the orchard boundary polygon by more than 0.5 m
6. Exports canopy polygons and refined treetop points as shapefiles

### Inputs

| Argument | Description |
|----------|-------------|
| `--chm` | CHM GeoTIFF from flight reconstruction |
| `--tree-markers` | Point shapefile of approximate tree locations |
| `--extent` (optional) | Boundary polygon; segments overlapping it by >0.5 m are removed |

### Outputs

Written to `--outdir`:

| File | Attributes |
|------|------------|
| `<name>_Canopies.shp` | `tree_id`, `area_m2`, `max_h` (m), `mean_h` (m) |
| `<name>_Treetops.shp` | `tree_id`, height at refined location |

### Usage

```powershell
conda activate bur-detection

python -m canopy_segmentation.segmentation `
    --chm "flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_chm.tif" `
    --tree-markers "canopy_segmentation/sample_data/inputs/20230823_Orchard4_tree_markers.shp" `
    --outdir "canopy_segmentation/sample_data/outputs/" `
    --min-height 1.75 `
    --buffer-size 0.5 `
    --id-column Tree_ID `
    --extent "canopy_segmentation/sample_data/inputs/20230823_Orchard4_boundary.shp"
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--chm` | yes | — | Path to CHM GeoTIFF |
| `--tree-markers` | yes | — | Tree marker point shapefile |
| `--outdir` | yes | — | Output directory |
| `--min-height` | no | `1.75` | Minimum CHM height (m) for the segmentation mask |
| `--buffer-size` | no | `1.0` | Buffer radius (m) for local maxima refinement |
| `--id-column` | no | first non-geometry column | Column name for tree IDs in marker file |
| `--extent` | no | — | Boundary polygon shapefile |

*Tree markers can be manually digitized using leaf-off imagery or collected with an RTK GNSS receiver in the field.*

### Limitations

- One marker = one canopy segment; touching or overlapping crowns are not automatically separated without separate markers
- Pairwise inter-tree distance computation scales O(n²); performance degrades above ~10,000 trees
- CHM and markers must share a coordinate reference system (geopandas will attempt auto-reprojection if they differ)

---

## Step 3: Image Selection

**Script:** `image_selection/canopy_to_image.py`

### What It Does

Selects the highest-quality raw drone image for each canopy polygon using Metashape's calibrated 3D camera model for back-projection:

1. Back-projects each canopy polygon onto all candidate raw images using the Metashape camera model (accounts for lens distortion, orientation, and 3D position)
2. Retains the 3 nearest images whose camera position is within 10 m (ground XY distance) of the canopy centroid
3. Scores each candidate image:
   - **Exposure:** mean pixel value in projected region; filtered to 0.15–0.85 (discards over/underexposed images)
   - **Gimbal pitch:** read from DJI XMP tag; images near nadir (~−90° ± 5°) are preferred
   - **Sharpness:** variance of the Laplacian over the projected canopy region
   - **Contrast:** standard deviation of pixel values in the projected region
4. Selects the image with highest sharpness (within 85% of the per-canopy maximum), breaking ties by contrast

### Inputs

| Argument | Description |
|----------|-------------|
| `--canopy_shapefile` | Canopy polygon shapefile from segmentation |
| `--dsm` | DSM raster (for 3D Z-value assignment during back-projection) |
| `--metashape_project` | Metashape `.psx` project with aligned cameras |
| `--raw_images` | Folder containing raw drone images |
| `--output` | Output directory |

### Output

`best_image_selections.json` — maps each tree ID to its selected image and projected canopy polygon:

```json
{
  "338": {
    "image_path": "flight_reconstruction/sample_data/20230823_Orchard4/DJI_20230823122017_0102_D.JPG",
    "polygon_coords": [[2521.04, 2099.90], [2508.38, 2099.97], ...]
  }
}
```

Polygon coordinates are in **image pixel space** (not geographic coordinates).

### Usage

```powershell
conda activate bur-detection

python -m image_selection.canopy_to_image `
    --canopy_shapefile "canopy_segmentation/sample_data/outputs/20230823_Orchard4_Canopies.shp" `
    --dsm "flight_reconstruction/sample_data/20230823_Orchard4/outputs/20230823_Orchard4_dsm.tif" `
    --metashape_project "flight_reconstruction/sample_data/20230823_Orchard4/outputs/project_20230823_Orchard4.psx" `
    --raw_images "flight_reconstruction/sample_data/20230823_Orchard4" `
    --output "image_selection/sample_data/outputs"
```

### Limitations

- Requires a fully-aligned Metashape project (cameras must already be positioned from Step 1)
- Assumes DJI XMP tags for gimbal pitch; other drone brands use different metadata schemas
- The 10 m image-plane distance threshold may need adjustment for different flight altitudes or focal lengths
- Does not check for occlusion by neighboring trees or branches

---

## Step 4: Bur Detection

**Entry point:** `bur_detection/detection.py`  
**Config:** `bur_detection/config.yml`

Four modes are available: `preprocess`, `train`, `tune`, and `inference`. All are accessed through the same entry point:

```powershell
conda activate bur-detection

python -m bur_detection.detection `
    --mode <preprocess|train|tune|inference> `
    --config "bur_detection/config.yml" `
    --data-root "<your dataset root>" `   # optional; overrides the sample-data paths
    --plot-mode <all|subset|none>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `inference` | `preprocess`, `train`, `tune`, or `inference` |
| `--config` | `bur_detection/config.yml` | Path to YAML config file (the committed config points at the bundled sample data) |
| `--data-root` | _(none)_ | Point the pipeline at your own dataset *without editing the committed config*; derives the `tiled/`, `full_canopy/`, `outputs/` layout (see below) |
| `--plot-mode` | `subset` | `all` (every image), `subset` (15 random), `none` |

Each invocation writes to one run folder, `<outputs_dir>/run_<timestamp>/`, with a subfolder per mode it ran: `preprocess/`, `tune/`, `train/`, `inference/`. `outputs_dir` is `data.outputs_dir` in `config.yml` (`bur_detection/sample_data/training/outputs`), or `<data-root>/outputs` with `--data-root`.

**Typical workflow:** `preprocess` (build or split the tiled training set) → `tune` (search hyperparameters) → copy the best hyperparameters into `training_params` → `train` (final model) → `inference`. No trained weights ship with the sample data (withheld pending publication), so run `preprocess → tune → train` to produce a model; `inference` can then run standalone.

---

### Dataset Format and Preprocessing (`--mode preprocess`)

`--mode preprocess` builds the training set and is the first step before `train`/`tune`. Given
full-resolution per-tree canopy images plus **polygon** (segmentation) bur labels, it:

1. Crops/masks each image to its canopy polygon and tiles it into 224×224 patches (20% overlap),
   clipping bur polygons to each tile and deriving bounding boxes; mostly-background tiles are dropped.
2. De-duplicates overlapping (double-annotated) boxes.
3. Creates a **group-aware** train/val/test split (70/20/10) where all tiles cut from one source tree
   stay in the same split — preventing the tree-level leakage a plain per-tile shuffle would cause.
4. Saves QA overlays so you can confirm labels align with the imagery.

> **Where the data comes from:** Steps 1–3 produce the per-tree canopy *images*, and the canopy polygon comes from segmentation (Step 2) — but the **bur polygon labels are not produced by the pipeline.** You create them by hand-annotating the canopy images (e.g., in Roboflow), since the detector learns from human-drawn labels.

Expected dataset layout (produced by your annotation/export step; pass its root with `--data-root`):

```
<data-root>/
├── full_canopy/
│   ├── images/    # full-resolution per-tree canopy images
│   ├── labels/    # YOLO-segment polygon bur labels (one .txt per image)
│   └── canopy/    # YOLO-segment canopy polygon per image (used for masking)
└── tiled/                           # written by --mode preprocess (the training set)
```

```powershell
python -m bur_detection.detection --mode preprocess `
    --data-root "<data-root>" --plot-mode subset
```

The bundled sample ships ready-to-train tiles in `…/training/tiled/`, plus the source images and
polygon labels in `…/training/full_canopy/` for reference. With the committed config, `--mode
preprocess` only builds the group-aware train/val/test split over those tiles — it does not re-tile.
To tile your own data, pass a `full_canopy/` layout via `--data-root <root>`. The full dataset is
distributed via Google Drive, not committed to git.

You can also chain the whole pipeline in one command — `tune` hands its winning hyperparameters
directly to `train`:

```powershell
python -m bur_detection.detection --mode preprocess,tune,train,inference `
    --data-root "<data-root>" --plot-mode subset
```

---

### 4a. Training Mode

Trains a YOLO model using **multi-step progressive gradient accumulation**, which simulates large effective batch sizes without exceeding GPU memory:

| Step | Layers Unfrozen |
|------|-----------------|
| 1 | Detection head (predictor) |
| 2 | Full head (neck + predictor) |
| 3 | Head + last ⅓ of backbone |
| 4 | Full model |

Per-step physical batch, gradient accumulation, max-epochs, and patience are set in `config.yml` (`training_steps`); physical batch is capped and accumulation reaches the large effective batches.

The learning rate is scaled per step as `lr = min(lr0, max_lr0) × (effective_batch / 64)^0.5`, capped at `max_scaled_lr`. Progressive unfreezing is **architecture-aware** (read from the model YAML) and re-applied on the live trainer at the start of each step, so the curriculum actually takes effect. The best-epoch optimizer state is **carried across each step boundary** (momentum is preserved through the unfreeze), and the learning rate is **warmed up** over a few epochs at each transition. The best checkpoint across all steps/epochs is selected by the composite objective described under Tuning Mode and saved as `best_model_weights.pt`.

```powershell
python -m bur_detection.detection --mode train `
    --config "bur_detection/config.yml" `
    --plot-mode subset
```

Outputs, written to `<outputs_dir>/run_<timestamp>/train/`:

| File | Description |
|------|-------------|
| `best_model_weights.pt` | Best checkpoint across all steps (by the composite objective) |
| `training_metrics.csv` | Per-epoch: learning rate, losses, precision, recall, mAP50, F1 |
| `test_results.csv` | Held-out test set: precision, recall, F1, mAP50, inference time |
| `train_step{1-4}/` | Per-step YOLO run directories (weights, results.csv, plots) |
| `prediction_plots/` | Side-by-side ground truth vs. prediction visualizations |

**Performance**

The production detector is **YOLO11m with a P2 (stride-4) head** for small, dense burs, warm-started from pretrained YOLO11m weights. The tuning search covers YOLOv8 and YOLO11 **medium and large** P2 variants — see `tuning_space.model_size` in `config.yml`.

Production model — YOLO11m-P2 (progressive 4-step) on the held-out test split (404 tiles at 224 px, 5,114 burs), ~1.3 ms/img inference on an RTX 4090:

| Precision | Recall | F1 | mAP50 | mAP50-95 |
|-----------|--------|----|-------|----------|
| 0.878 | 0.864 | 0.871 | 0.935 | 0.642 |

*Trained on the full multi-orchard set (proprietary, not distributed here). The bundled single-orchard sample is much smaller, so metrics from a sample-only run will be lower and vary noticeably run-to-run.*

**Key `training_params` in `config.yml`** (the production model's tuned hyperparameters):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_size` | `yolo11m-p2.yaml` | YOLO11m + P2 head (warm-started from `yolo11m.pt`) |
| `optimizer` | `SGD` | — |
| `lr0` / `lrf` | `0.0037` / `0.0327` | Initial LR; final-LR fraction (decays toward `lrf·lr0`) |
| `momentum` | `0.943` | SGD momentum |
| `weight_decay` | `0.00835` | — |
| `box_gain` / `cls_gain` / `dfl_gain` | `4.43` / `0.56` / `0.80` | Loss gains (box/cls pinned during tuning; dfl tuned) |
| `hsv_h` / `hsv_s` / `hsv_v` | `0.0070` / `0.491` / `0.156` | HSV augmentation |
| `degrees` / `scale` / `flipud` | `49.9` / `0.386` / `0.081` | Geometric augmentation (nadir imagery) |

---

### 4b. Tuning Mode

Searches the hyperparameter space using **Ray Tune with Optuna (Bayesian) search** and ASHA early stopping:

- Trial count and concurrency are set in `config.yml` (`ray_tune.num_samples`, `max_concurrent_trials`)
- ASHA scheduler prunes underperforming trials (reduction factor 3), with a grace period equal to the first training step's `patience`
- Optimization metric, used for both scheduling and best-model selection: a composite objective, lower is better, = `loss`·val_loss + `f1`·(1−F1)·10 + `map50`·(1−mAP50)·10, with the weights from `score_weights` in `config.yml`. The ×10 puts the quality terms on the scale of a typical val_loss. Raw val_loss is not used because tuning the box/cls/dfl loss gains would confound it
- Trials rank by their running best objective (`objective_best`), so a trial's best epoch counts, not its last
- A degenerate trial (non-finite or non-positive val_loss, or F1 and mAP50 both near zero) gets a sentinel objective of 1e6 and is pruned, not crashed
- Failed trials are not retried (`max_failures=0`), because the optimizer handoff between steps is not checkpointed
- The `training_params` entry seeds Optuna's first trial only when `ray_tune.warm_start` is `true` (off in the committed config)

```powershell
python -m bur_detection.detection --mode tune `
    --config "bur_detection/config.yml" `
    --plot-mode subset
```

**Tuning space** (from `bur_detection/config.yml`):

The search covers `model_size` (YOLOv8m, YOLOv8l, YOLO11m, and YOLO11l, all with the P2 head), learning rate (`lr0`/`lrf`), `momentum`, `weight_decay`, the DFL loss gain (`dfl_gain`), and augmentation (`hsv_*`, `degrees`, `scale`, `flipud`). `optimizer` is fixed to SGD, and `box_gain` and `cls_gain` are pinned to single values because they change only the gain-scaled val_loss, not F1 or mAP50. See `tuning_space` in `config.yml` for the exact set and ranges.

Outputs, written to `<outputs_dir>/run_<timestamp>/tune/`:

| File | Description |
|------|-------------|
| `best_<model>_model.pt` | Best model weights |
| `best_trial_config.json` | Hyperparameters for the best trial |
| `all_tuning_history.csv` | Metrics for all trials |
| `best_trial_training_history.csv` | Per-epoch metrics for the best trial |
| `test_results.csv` | Test set evaluation of best model |
| `prediction_plots/` | Prediction visualizations |

Each tuning run also appends one row to `model_registry.csv` at the `outputs/` root — a cross-run index of every winner (run, model, composite objective, key metrics, and paths to its weights + config). Sort by `objective` (lower is better) to find the best run; promotion into `config.yml` (`training_params`) stays a manual copy.

Ray trial folders nest under the same `tune/` folder. Each trial keeps only its best-objective checkpoint, and only the top `ray_tune.keep_top_n` trials' checkpoints stay on disk; the rest are deleted as trials finish. Hyperparameter-importance and top-trial curves (`hp_importance.png`, `top_trial_curves.png`, `trial_summary.csv`) are written to the Ray experiment directory at the end of the run.

---

### 4c. Inference Mode

Detects burs on unlabeled canopy images using a tile-and-reconstruct strategy:

For each tree in `best_image_selections.json`:
1. Crop the canopy polygon region from the full drone image (mask outside region to black)
2. Tile the cropped canopy into 224×224 patches with 20% overlap (stride = 179 px); skip all-black tiles
3. Run YOLO inference on the tiles in batches (`inference.tile_batch_size`), confidence threshold 0.5 by default
4. Reconstruct tile-space detections to full canopy coordinates, keeping only detections whose box **center** lies in each tile's non-overlapping **core** region — so a bur in the overlap seam is counted once, not double-counted
5. Apply a light global NMS (`inference.global_nms_iou`, default 0.3) to resolve any residual cross-tile duplicates
6. Aggregate per tree: total detections, average confidence, bounding box coordinates

When `inference.model_path` is `null`, the model is auto-detected: the newest `run_*/` under `outputs_dir` is searched first, its `train/` before its `tune/`, for a `best_*.pt` file. Older flat `tuning_*/` and `training_*/` folders are searched after that. To use a specific model, set `inference.model_path` to an explicit path.

```powershell
python -m bur_detection.detection --mode inference `
    --config "bur_detection/config.yml" `
    --plot-mode subset
```

**Inference config keys:**

```yaml
inference:
  model_path: null          # null = auto-detect most recent training/tuning output
  conf_threshold: 0.5
  iou_threshold: 0.45        # per-tile NMS during predict()
  global_nms_iou: 0.3        # light cross-tile NMS after core-region filtering
  tile_batch_size: 96        # tiles per batched predict() call

data:
  image_selections: image_selection/sample_data/outputs/best_image_selections.json
```

Outputs, written to `<outputs_dir>/run_<timestamp>/inference/`:

| File | Description |
|------|-------------|
| `tree_bur_detections.csv` | Per tree: `tree_id`, `image_path`, `total_detections`, `avg_confidence`, `detection_coords` |
| `detection_summary.txt` | Summary stats: mean/min/max burs per tree, overall confidence, model info |
| `preprocessed_trees/` | Cropped canopy images (for visual verification of masking) |
| `prediction_plots/` | Detection visualizations with bounding boxes |

---

## Limitations and Known Assumptions

- **Platform:** Windows-only due to the Agisoft Metashape Python API (Steps 1 and 3)
- **Drone hardware:** DJI Mavic 3M (baseline) and DJI Zenmuse P1 (`p1` profile); RTK accuracy parsing and gimbal pitch metadata depend on DJI XMP tags. Bur-detection tile size (224 px) was set at M3M GSD
- **Single class:** The detector is configured for one object class (bur); multi-class use requires label and config changes
- **Canopy segmentation:** One-to-one mapping between markers and canopies; touching or overlapping crowns are not automatically split without separate markers per crown and, as such, outputs usually require manual cleanup. ## TODO: instance segmentation from point cloud
- **Image selection:** Does not account for occlusion of a canopy by adjacent trees or branches
- **Tiling:** Tile size (224×224) and overlap (20%) match across preprocessing and inference; both are configurable via the `data.tiling` and `inference` keys in `config.yml`
- **Metashape license:** Agisoft Metashape Professional is required for Steps 1 and 3; the included wheel is version 2.3.1

---

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

---

## Acknowledgments

This project uses [easyidp](https://easyidp.readthedocs.io/en/latest/index.html) for 3D back-projection in the image selection step. If you use this pipeline in your research, please cite:

Wang, Haozhou and Duan, Yulin and Shi, Yun and Kato, Yoichiro and Ninomiya, Seish and Guo, Wei. "EasyIDP: A Python Package for Intermediate Data Processing in UAV-Based Plant Phenotyping." *Remote Sensing* 13, no. 13 (2021): 2622. https://doi.org/10.3390/rs13132622
