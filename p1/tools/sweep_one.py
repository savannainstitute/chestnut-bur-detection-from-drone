"""Train and score ONE cell of the learning-curve sweep, on one GPU, inside a shared allocation.

Called by `sweep_pool.py`, which packs several of these onto a single GPU because yolov8n-p2 at 640
uses a small fraction of an A30 or A100 and the sweep is throughput-bound rather than
memory-bound. Packing also means the campaign asks the scheduler for a few dozen GPUs instead of a
few thousand job slots, which matters on a partition that is fully allocated.

THE RECIPE IS FROZEN AND ONLY THE DATA VARIES. Nathan's v-Blue settings, including the
RandomRotate90 injection that is worth 6.4 points of recall, are reproduced exactly as in
`work/2026-10-01_training_experiments/configs/nathan_vblue_train.py`. Changing anything here breaks
comparability with the N1 result the sweep is anchored to.

THE VALIDATION SET MIRRORS THE TRAINING COMPOSITION, which is the one design choice in this file
worth arguing. Early stopping on the Clarks Run validation trees for every cell would be a cleaner
objective, and it would also hand eighteen trees of Clarks Run model-selection signal to the arms
that are supposed to have no Clarks Run data at all. Since the headline comparison is exactly
"with and without her annotations", those arms have to be genuinely without, so a run trains and
stops on whichever sources it was given.

Predictions are written with their scores rather than a verdict, so every operating point can be
swept afterward without re-running inference. That is what made the matched-precision comparison
possible on the first six arms.
"""

import argparse
import gzip
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path


def device_name():
    """Name the GPU for the record, and never let that question kill a finished run.

    The call that reports it sits in the result dict, which is built AFTER the training and the
    inference, so an exception there discards hours of completed work. `torch.cuda.get_device_name`
    raises outright when torch has no CUDA, which is how this was found.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        return f"unknown ({type(e).__name__})"
    return "cpu"


def host_environment():
    """Record the HOST CPU, its vector instruction sets, and the GPU driver, for every cell.

    This exists because two facilities running identical bytes, identical wheels and
    `deterministic=True` on the same GPU GENERATION still produced per-tree counts that differed by
    about as much as two seeds do. The augmentation pipeline (Albumentations over OpenCV and numpy,
    with the injected `RandomRotate90`) runs on the CPU, and both libraries dispatch on the vector
    instruction sets available, so a different CPU make can change augmented pixels from the first
    batch. MEASURED 2026-10-03: falcon's nodes are Intel Xeon Platinum 8462Y+ with AVX-512, while
    tinkercliffs' DGX A100 nodes are AMD and report no AVX-512. Tranches 3 to 6 are training on
    falcon while tranches 1 and 2 ran on tinkercliffs, so this is a live confound in this campaign
    and not a hypothetical.

    Recording it costs nothing and turns the question from one needing a probe job into a grouping
    over the results already collected. Everything is wrapped, because this runs in the result dict
    AFTER training and must never discard a finished cell.
    """
    env = {}
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and "cpu_model" not in env:
                env["cpu_model"] = line.split(":", 1)[1].strip()
            elif line.startswith("flags") and "cpu_vector" not in env:
                fl = set(line.split(":", 1)[1].split())
                env["cpu_vector"] = " ".join(sorted(
                    f for f in fl if f.startswith("avx") or f in ("sse4_2", "fma")))
            if "cpu_model" in env and "cpu_vector" in env:
                break
    except Exception:
        pass
    try:
        import torch
        env["driver"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            maj, mnr = torch.cuda.get_device_capability(0)
            env["gpu_capability"] = f"{maj}.{mnr}"
    except Exception:
        pass
    try:                                   # the real driver, which torch does not expose
        import subprocess
        env["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        pass
    for k in ("SWEEP_CONCURRENT", "SWEEP_GPU_MIB", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID",
              "_CONDOR_SLOT"):
        v = os.environ.get(k)
        if v:
            env[k.lower()] = v
    return env


def read_boxes(path, keep):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        f = line.split()
        if len(f) == 5 and int(f[0]) in keep:
            out.append(tuple(map(float, f[1:5])))
        elif len(f) >= 6 and int(f[0]) in keep:
            out.append(tuple(map(float, f[2:6])))
    return out


def score_frames(mdl, test_stems, val_stems, infer_dir, tile, labels_dir, keep, conf, infer_batch,
                 device=0):
    """Score a trained model at the TREE level, frame by frame: the ONE scoring path.

    Training and re-scoring both call this, so a re-score of kept weights on new trees runs exactly
    the code that produced the filed results: tiles predicted at `conf` with overlap removal at 0.45
    inside a tile, overlap removal at 0.3 across tiles of a frame, and truth read from the label
    files filtered to `keep`, the tile store's own class list (0, 2 and 4, the three in-crown bur
    classes; class 3, a bur on the ground, is never truth). Validation stems absent from the tile
    grid are skipped, as they always were; a missing TEST stem raises, as it always did.
    """
    import torch
    from PIL import Image
    from torchvision.ops import nms

    Image.MAX_IMAGE_PIXELS = None
    grids = json.load(open(Path(infer_dir) / str(tile) / "FRAMES.json"))
    test_set = set(test_stems)
    t1 = time.time()
    trees = []
    for stem in list(test_stems) + [s for s in val_stems if s in grids["frames"]]:
        g = grids["frames"][stem]
        d = Path(infer_dir) / str(tile) / stem
        origins = g["origins"]
        boxes, scores = [], []
        for i in range(0, len(origins), infer_batch):
            chunk = origins[i:i + infer_batch]
            paths = [str(d / f"{x}_{y}.jpg") for x, y in chunk]
            out = mdl.predict(paths, imgsz=tile, conf=conf, iou=0.45, device=device, verbose=False)
            for r, (ox, oy) in zip(out, chunk):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xy, cf):
                    boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
                    scores.append(float(c))
        if boxes:
            kp = nms(torch.tensor(boxes, dtype=torch.float32),
                     torch.tensor(scores, dtype=torch.float32), 0.3).tolist()
            boxes = [[round(float(v), 1) for v in boxes[i]] for i in kp]
            scores = [round(float(scores[i]), 4) for i in kp]
        W, H = g["W"], g["H"]
        truth = [[round(b[0] * W - b[2] * W / 2, 1), round(b[1] * H - b[3] * H / 2, 1),
                  round(b[0] * W + b[2] * W / 2, 1), round(b[1] * H + b[3] * H / 2, 1)]
                 for b in read_boxes(os.path.join(labels_dir, stem + ".txt"), keep)]
        trees.append(dict(tree=stem.split("__")[0], frame=stem, truth=truth,
                          boxes=boxes, scores=scores,
                          split="test" if stem in test_set else "val"))
    return trees, time.time() - t1


def md5_of(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--tilestore", required=True, help="dir holding train<tile>/{images,labels}")
    ap.add_argument("--infer", required=True, help="dir holding <tile>/FRAMES.json and tile jpgs")
    ap.add_argument("--labels", required=True, help="Clarks Run label dir, for the truth boxes")
    ap.add_argument("--work", required=True, help="scratch dir for this run, deleted at the end")
    ap.add_argument("--out", required=True, help="results dir")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-tiles-per-source", type=int, default=250,
                    help="cap on validation tiles from EACH source; see the comment where "
                         "the validation set is built for why this is per source")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--infer-batch", type=int, default=64)
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--score-only", default=None, metavar="WEIGHTS",
                    help="skip training: score these kept weights with the same scoring code and "
                         "write a re-score result. Frames default to the fold's own test and "
                         "validation frames, which is the reproduction test.")
    ap.add_argument("--frames", default=None,
                    help="score-only: a file of frame stems, one a line, each optionally followed "
                         "by 'test' or 'val' (default test). New trees must be assigned to folds "
                         "by drone mission before they are listed here.")
    ap.add_argument("--filed-result", default=None,
                    help="score-only: the run's filed result, whose training fields are copied in")
    ap.add_argument("--device", default="0", help="score-only: the device to predict on")
    ap.add_argument("--pretrained", default="yolov8n.pt",
                    help="COCO starting weights, as an absolute path on a cluster so three "
                         "thousand runs do not each fetch them")
    ap.add_argument("--amp-model", default=None,
                    help="a specific AMP-check checkpoint to link into the run directory")
    ap.add_argument("--amp-model-dirs", action="append",
                    default=["/ultralytics"],
                    help="directories whose *.pt are ALL linked into the run directory, so the "
                         "AMP check finds whatever this ultralytics version asks for instead of "
                         "downloading it. The container carries /ultralytics/yolo26n.pt.")
    a = ap.parse_args()
    # the run changes directory so ultralytics can find its AMP check weights locally, so every
    # path has to be absolute before that happens
    for k in ("design", "tilestore", "infer", "labels", "work", "out", "pretrained", "amp_model",
              "score_only", "frames", "filed_result"):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(os.path.expanduser(getattr(a, k))))

    design = json.load(open(a.design))
    run = next(r for r in design["runs"] if r["run_id"] == a.run_id)
    scheme = design["schemes"][run["scheme"]]
    fold = scheme["folds"][run["fold"]]
    tile = run["tile"]
    man = json.load(open(Path(a.tilestore) / f"train{tile}" / "MANIFEST.json"))["frames"]
    keep = tuple(json.load(open(Path(a.tilestore) / f"train{tile}" / "MANIFEST.json"))["classes"])

    if a.score_only:
        return score_only(a, run, fold, tile, keep)

    # ---- dataset: a text file of tile paths, so nothing is copied or linked ----------
    work = Path(a.work) / a.run_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    img = Path(a.tilestore).resolve() / f"train{tile}" / "images"

    def tiles_for(frames):
        return [str(img / f"{t}.jpg") for s in frames for t in man[s]["tiles"]]

    train = tiles_for(run["ren_train"] + run["nathan_train"])

    # THE VALIDATION SET IS CAPPED PER SOURCE, and the reason is measured rather than aesthetic.
    # Ultralytics validates every epoch. The fold's 18 Clarks Run validation trees plus the 7 Ohio
    # validation frames come to about 1,600 tiles, which is 100 batches, while the smallest cell in
    # the grid trains on 12 tiles, or ONE batch. So at the small end of the learning curve the
    # validation pass costs a hundred times the training pass and sets the whole cell's wall clock.
    # Capping each source at a fixed number of tiles, drawn once per fold with a fixed seed, makes
    # the per-epoch validation cost constant across the grid and leaves early stopping with ample
    # signal. It must be per SOURCE rather than a single total, so that a run with no Ohio data and
    # a run with no Clarks Run data each still validate on the same tiles as their peers.
    # AND CAPPED PER FRAME, because one frame otherwise owns the objective. Measured: of the seven
    # Ohio validation frames, `DJI_20250905134657_0011_point10` holds 1,169 of the 1,430 boxes,
    # which is 81.7 percent of them and 55.7 percent of the tiles. Early stopping and best.pt
    # selection for every mixed cell in the campaign would therefore be decided by one frame of
    # dense mature canopy, which is the regime LEAST like the site being predicted.
    val, val_frames = [], []
    for present, frames, tag in ((run["n_ren"] > 0, fold["val_ren"], "ren"),
                                 (run["n_nathan"] > 0, scheme["val_nathan"], "nathan")):
        if not present:
            continue
        val_frames += frames
        take = max(1, a.val_tiles_per_source // max(1, len(frames)))
        picked = []
        for s in sorted(frames):
            pool = sorted(f"{t}" for t in man[s]["tiles"])
            random.Random(f"{tag}|{s}|{run['scheme']}|{run['fold']}|{tile}").shuffle(pool)
            picked += pool[:take]
        val += sorted(str(img / f"{t}.jpg") for t in picked)
    if not train or not val:
        raise SystemExit(f"{a.run_id}: empty train ({len(train)}) or val ({len(val)})")

    # EACH CELL NEEDS ITS OWN LABEL CACHE PATH, and this is why the tile paths go through a
    # per-cell symlinked view instead of pointing at the store directly. Ultralytics derives the
    # cache path as `Path(label_files[0]).parent.with_suffix(".cache")`, so every cell sharing one
    # label directory writes the SAME `labels.cache`. Its writer unlinks the file and then streams
    # a pickle into it, with no temporary file and no rename, so with five cells running on one GPU
    # there is a window in which the file exists and is half written. Each cell's hash check fails
    # against the previous cell's file list anyway, so every cell rewrites it: a guaranteed race
    # rather than an unlucky one. Two views per cell, one for train and one for val, also stop a
    # cell's own two datasets from overwriting each other's cache.
    lab = Path(a.tilestore).resolve() / f"train{tile}" / "labels"
    paths = {}
    for split, lst in (("tr", train), ("va", val)):
        v = work / split
        v.mkdir(parents=True, exist_ok=True)
        (v / "images").symlink_to(img, target_is_directory=True)
        (v / "labels").symlink_to(lab, target_is_directory=True)
        paths[split] = [str(v / "images" / os.path.basename(p)) for p in lst]

    (work / "train.txt").write_text("\n".join(paths["tr"]) + "\n")
    (work / "val.txt").write_text("\n".join(paths["va"]) + "\n")
    (work / "data.yaml").write_text(
        f"path: {work}\ntrain: {work}/train.txt\nval: {work}/val.txt\nnc: 1\nnames: [bur]\n")

    # ---- train ----------------------------------------------------------------------
    import albumentations as A
    import ultralytics.data.augment as aug

    _orig = aug.Albumentations.__init__

    def _patched(self, p=1.0, transforms=None):
        _orig(self, p, transforms)
        bp = A.BboxParams(format="yolo", label_fields=["class_labels"])
        try:
            existing = list(self.transform.transforms) if getattr(self, "transform", None) else []
        except Exception:
            existing = []
        self.transform = A.Compose([A.RandomRotate90(p=1.0), *existing], bbox_params=bp)
        self.contains_spatial = True

    aug.Albumentations.__init__ = _patched

    from ultralytics import YOLO

    # LINK EVERY CHECKPOINT THE IMAGE CARRIES, not one named guess. Ultralytics runs an AMP check
    # that loads a small model by name through `check_file`, which looks in the working directory
    # first and otherwise DOWNLOADS. The name changes between versions: the 8.3 releases opened
    # `yolo11n.pt` and 8.4.76 opens `yolo26n.pt`, at `utils/checks.py` in `check_amp`. Passing
    # `--amp-model yolo11n.pt` therefore satisfied nothing here, and the measured cost was 871
    # downloads of yolo26n.pt plus 89 of Arial.ttf across 452 cells, which is the exact waste the
    # argument existed to prevent. Linking the whole directory is version-proof: whatever the next
    # release asks for, it is already beside the run.
    # LINK ONLY WHAT THIS PROCESS CAN ACTUALLY READ, PROVEN BY READING IT. The first version of this
    # block linked every `*.pt` it could SEE, and that broke every cell it touched: the weights in
    # the image are owned by root with no world read bit, Apptainer runs as the invoking user rather
    # than as root, and `glob`, `is_file` and `resolve` all succeed on a file you cannot open. So a
    # valid symlink was planted to an unreadable file, ultralytics stopped downloading its own copy
    # and died on `PermissionError: [Errno 13] Permission denied: 'yolo26n.pt'` 15 seconds into
    # training. Measured cost: three cells a task on falcon, and one task failed outright before the
    # cause was read out of the cell log. `os.access` is not the test either, since it answers about
    # the real uid and can disagree with the kernel's decision on the actual open; opening the file
    # and reading a byte is the only check that asks the question the library will ask.
    def readable(p):
        try:
            with open(p, "rb") as fh:
                return fh.read(1) != b""
        except OSError:
            return False

    for src in sorted(a.amp_model_dirs or []):
        d = Path(src)
        if not d.is_dir():
            continue
        for pt in sorted(d.glob("*.pt")):
            link = work / pt.name
            if link.exists() or not readable(pt):
                continue
            try:
                os.symlink(pt.resolve(), link)
            except OSError:
                pass
    if a.amp_model and readable(a.amp_model):
        link = work / os.path.basename(a.amp_model)
        if not link.exists():
            os.symlink(os.path.abspath(a.amp_model), link)

    # The plotting font lives at /root/.config/Ultralytics inside the image, which is not where a
    # non-root job's settings directory is, so it is fetched over the network unless it is seeded.
    # Written to a temporary name and renamed, because `shutil.copy` is not atomic and a truncated
    # font is worse than an absent one: an absent one is downloaded, a truncated one is parsed.
    cfg = os.environ.get("YOLO_CONFIG_DIR")
    if cfg:
        for cand in ("/root/.config/Ultralytics/Arial.ttf",
                     "/root/.config/Ultralytics/Arial.Unicode.ttf"):
            dst = Path(cfg) / os.path.basename(cand)
            if dst.exists() or not readable(cand):
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.part")
                shutil.copy(cand, tmp)
                os.replace(tmp, dst)
            except OSError:
                pass
    os.chdir(work)

    t0 = time.time()
    m = YOLO("yolov8n-p2.yaml").load(a.pretrained)
    # the returned metrics object describes only the LAST epoch, so the run's history is read back
    # from results.csv below and the best epoch is chosen there
    m.train(data=str(work / "data.yaml"), imgsz=tile, epochs=a.epochs, batch=a.batch,
                  patience=a.patience, project=str(work), name="t", exist_ok=True,
                  degrees=0.0, fliplr=0.5, flipud=0.5, translate=0.1, scale=0.5,
                  mosaic=1.0, erasing=0.4, device=0, seed=run["seed"], deterministic=True,
                  workers=a.workers, verbose=False, plots=False, val=True, save_period=-1)
    train_s = time.time() - t0

    import csv
    rows = list(csv.DictReader(open(work / "t" / "results.csv")))
    def col(r, *names):
        for n in names:
            for k in r:
                if k.strip() == n:
                    return float(r[k])
        return float("nan")
    val_curve = [dict(epoch=int(col(r, "epoch")),
                      mAP50=col(r, "metrics/mAP50(B)"),
                      mAP50_95=col(r, "metrics/mAP50-95(B)"),
                      precision=col(r, "metrics/precision(B)"),
                      recall=col(r, "metrics/recall(B)")) for r in rows]
    best = max(val_curve, key=lambda d: d["mAP50_95"]) if val_curve else {}

    # ---- score at the TREE level on the fold's held-out frames -----------------------
    import torch

    best_w = work / "t" / "weights" / "best.pt"
    mdl = YOLO(str(best_w))

    # The confidence threshold is a free parameter, and picking it on the same trees the recall is
    # reported on is selection on the test set. So the fold's validation trees are scored too, and
    # the operating point is chosen there. The earlier six-arm table matched precision on the test
    # trees themselves, which is the comparison to reproduce but not the one to rely on.
    trees, infer_s = score_frames(mdl, fold["test"], fold["val_ren"], a.infer, tile, a.labels, keep,
                                  a.conf, a.infer_batch, device=0)

    result = dict(
        run=run, val_composition=dict(ren=run["n_ren"] > 0, nathan=run["n_nathan"] > 0,
                                      n_val_frames=len(val_frames), n_val_tiles=len(val),
                                      val_tiles_per_source=a.val_tiles_per_source),
        n_train_tiles=len(train), epochs_run=len(val_curve), best_epoch=best.get("epoch"),
        best_val=best, val_curve=val_curve, train_seconds=round(train_s, 1),
        infer_seconds=round(infer_s, 1), conf_floor=a.conf, trees=trees,
        host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
        ultralytics=__import__("ultralytics").__version__,
        torch=torch.__version__, device=device_name(), host_env=host_environment(),
    )

    out = Path(a.out)
    (out / "done").mkdir(parents=True, exist_ok=True)
    tmp = out / f".{a.run_id}.json.gz.part"
    with gzip.open(tmp, "wt") as f:
        json.dump(result, f)
    os.replace(tmp, out / f"{a.run_id}.json.gz")          # atomic, so a partial file never looks done
    if a.keep_weights:
        (out / "weights").mkdir(exist_ok=True)
        shutil.copy(best_w, out / "weights" / f"{a.run_id}.pt")
    (out / "done" / a.run_id).write_text(
        f"{os.environ.get('SLURM_JOB_ID','local')} {train_s:.0f}s+{infer_s:.0f}s "
        f"{len(train)} tiles {len(val_curve)} epochs best {best.get('epoch')}\n")
    os.chdir(a.out)                      # never delete the directory you are standing in
    shutil.rmtree(work, ignore_errors=True)
    print(f"{a.run_id} OK  train {train_s:.0f}s  infer {infer_s:.0f}s  "
          f"{len(train)} tiles  {len(val_curve)} epochs  "
          f"best mAP50-95 {best.get('mAP50_95')}", flush=True)


def score_only(a, run, fold, tile, keep):
    """Re-score kept weights. The result goes to its OWN results tree, never among training results.

    The filed result's training fields are copied in, the trees and inference time are replaced, and
    a `rescore` block records what was scored with what: the weights' md5, the frames, the classes,
    the scoring settings and the md5 of this file.
    """
    import datetime
    import torch
    from ultralytics import YOLO

    if a.frames:
        test_stems, val_stems = [], []
        for line in open(a.frames):
            f = line.split()
            if not f:
                continue
            (val_stems if len(f) > 1 and f[1] == "val" else test_stems).append(f[0])
    else:
        test_stems, val_stems = list(fold["test"]), list(fold["val_ren"])
    dev = int(a.device) if str(a.device).isdigit() else a.device
    trees, infer_s = score_frames(YOLO(a.score_only), test_stems, val_stems, a.infer, tile,
                                  a.labels, keep, a.conf, a.infer_batch, device=dev)
    base = json.load(gzip.open(a.filed_result, "rt")) if a.filed_result else dict(run=run)
    if base.get("run", {}).get("run_id") != a.run_id:
        raise SystemExit(f"{a.run_id}: the filed result is for {base.get('run', {}).get('run_id')}")
    drop = ("trees", "infer_seconds", "host", "slurm_job", "host_env", "device", "ultralytics",
            "torch", "rescore")
    result = {k: v for k, v in base.items() if k not in drop}
    result.update(trees=trees, infer_seconds=round(infer_s, 1), conf_floor=a.conf,
                  host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
                  ultralytics=__import__("ultralytics").__version__, torch=torch.__version__,
                  device=device_name(), host_env=host_environment())
    result["rescore"] = dict(
        weights=a.score_only, weights_md5=md5_of(a.score_only), filed_result=a.filed_result,
        frames_file=a.frames, n_test=sum(t["split"] == "test" for t in trees),
        n_val=sum(t["split"] == "val" for t in trees), classes=list(keep), conf=a.conf,
        infer_batch=a.infer_batch, iou_tile=0.45, iou_frame=0.3, tile=tile,
        code_md5=md5_of(os.path.abspath(__file__)),
        when=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    out = Path(a.out)
    (out / "done").mkdir(parents=True, exist_ok=True)
    tmp = out / f".{a.run_id}.json.gz.part"
    with gzip.open(tmp, "wt") as f:
        json.dump(result, f)
    os.replace(tmp, out / f"{a.run_id}.json.gz")
    (out / "done" / a.run_id).write_text(f"rescore {len(trees)} frames {infer_s:.0f}s\n")
    print(f"{a.run_id} RESCORED  {len(trees)} frames  {infer_s:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
