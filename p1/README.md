# P1 bur detection, as trained in the 2026 learning-curve campaign

This folder holds the bur detector for DJI Zenmuse P1 hover frames: the training recipe Nathan Miller developed (his
v-Blue settings), the code that trained 5,150 models with it on VT ARC and UW CHTC in October 2026, and the code that
scored them. The M3M pipeline in `bur_detection/` is unchanged; this sits beside it.

# How it differs from `bur_detection/`

| | `bur_detection/` (M3M) | `p1/` |
| --- | --- | --- |
| model | yolo11m-p2, trained in four progressive batch steps with tuned hyperparameters | yolov8n-p2 from COCO weights, trained in one stage |
| tiles | 224 px tiles of the masked canopy, 20 percent overlap | 640 px tiles of whole frames, stride 320, half as many bur-free tiles as tiles with burs |
| augmentation | tuned color, rotation, scale and vertical flip | a 90-degree rotation of every image, horizontal and vertical flips at 0.5, Ultralytics defaults otherwise |
| scoring | confidence 0.5 | a confidence floor of 0.15, then each model's own threshold, set on its 18 validation trees; boxes over 130 px dropped |
| scale | native | the six dense older-tree Ohio frames enlarged 1.5x (2.35x in one arm) |

`config.yml` records every setting as run.

# What is in `tools/`

- `build_tile_store.py` cuts the training tiles from annotated frames; `cut_infer_tiles.py` cuts the inference tiles
  of each test frame, its edges included.
- `sweep_one.py` trains one model (one cell of the design) and scores its fold's test and validation frames.
  `--score-only` scores kept weights on frames annotated later, with the same scoring code.
- `sweep_pool.py` runs several cells on one GPU, sized from the card's free memory, and withdraws any cell whose log
  shows a silent out-of-memory fallback.
- `make_sweep_design.py`, `append_tranche6.py` and `append_upscale_arm.py` build the design of 5,150 runs;
  `partition_sweep_tasks.py`, `verify_partition.py`, `verify_partition_multi.py` and `verify_stores.py` cut it into
  cluster tasks and check the cut and the tile stores.
- `analyze_sweep.py` and `make_figures.py` compute per-tree count accuracy, the learning curves and the figures.
- `extend_test_trees.py`, `pool_rescores.py` and `check_pooled_rescores.py` add trees annotated after the design to the
  test set and score the trained models on them without retraining.
- `annotation_vs_field.py` compares the annotation of each tree with the crew's field count of it.
- `requirements-sweep.txt` pins the analysis environment; training ran in the Ultralytics 8.4.76 image.

These are the files as they ran. `sweep_pool.py` and `analyze_sweep.py` differ from the run copies only in the
wording of a few comments, and their parsed code is identical. The scripts find their data by paths relative to the
project folder they ran in (`work/`, `inputs/`), which is not part of this repository.

# The design

Seven tranches varied how many annotated Clarks Run 2026 trees (0 to 103) and Ohio 2025 frames (0 to 36) each model
trained on, with the recipe held fixed: the main grid over five folds blocked by drone flight and three seeds; the same
grid with folds split by tree; which Clarks Run trees to annotate; tile sizes of 448, 960 and 1,280 px; which Ohio
frames transfer; a dense curve with no Ohio frames over six seeds; and the older-tree frames enlarged 2.35x instead of
1.5x.

# Data and weights

The weights of every model, the two training tile stores (each holding its tiles and labels), the container image and
the run tables are on the project's S3 bucket: endpoint `https://web.s3.wisc.edu`, bucket `chestnut`, prefix
`chestnut-imagery/UAV/sweeps/2026-10-02_arc_sweep/`, with a README there. `metadata/design.json` gives each run id's
training set. The test frames and their annotation are held by the project.

# Running one model

    python tools/sweep_one.py --design design.json --run-id r00102 \
        --tilestore <folder holding train640/> --infer <folder holding 640/FRAMES.json and its tiles> \
        --labels <Clarks Run label folder> --work <scratch folder> --out <results folder> --keep-weights

# Status

This is a first cut, the code as it ran. The publication pass would read the settings from `config.yml`, fold P1 into
`bur_detection/config.yml` under a sensor key, replace the project-folder paths, and settle two known defects of the
recipe for any new training: the rotation is unseeded, so two runs of one cell differ, and the stride 16 and 32 heads,
which no bur is large enough to train, produce the oversized boxes that the 130 px gate removes.
