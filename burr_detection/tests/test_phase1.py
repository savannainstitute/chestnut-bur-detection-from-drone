"""Lightweight CPU-only checks for the Phase-1 burr_detection changes.

These need the `burr-detection` conda env (the modules import torch/ultralytics),
but no GPU and no training run. The two model-dependent checks skip cleanly if
ultralytics is unavailable.

Run from the repo root:
    python -m pytest burr_detection/tests/ -q
  or (no pytest needed):
    python burr_detection/tests/test_phase1.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the repo root importable when run as a plain script.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from burr_detection.dataset import burr_tile_group_key, prepare_dataset_splits, CanopyTiler
from burr_detection.utils import compute_composite_objective


def test_burr_tile_group_key():
    assert burr_tile_group_key("route9_orchard3_115_537_537") == "route9_orchard3_115"
    assert burr_tile_group_key(Path("/x/route9_orchard4_70_1074_537.jpg")) == "route9_orchard4_70"
    assert burr_tile_group_key("tile_5_10") == "tile"
    # No trailing "_<int>_<int>" -> falls back to the whole stem.
    assert burr_tile_group_key("weird_name") == "weird_name"


def test_group_split_no_leakage():
    """No source group may appear in more than one split (the key leakage regression guard)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        images, labels = td / "images", td / "labels"
        images.mkdir()
        labels.mkdir()
        # 12 source groups x 4 tiles; even groups foreground, odd groups background.
        for g in range(12):
            for t in range(4):
                stem = f"orchardX_{g}_{t * 179}_0"
                (images / f"{stem}.jpg").write_bytes(b"")  # split logic never opens images
                n = 2 if g % 2 == 0 else 0
                (labels / f"{stem}.txt").write_text(
                    "\n".join("0 0.5 0.5 0.1 0.1" for _ in range(n))
                )
        out = td / "out"
        prepare_dataset_splits(images, labels, out, splits=(0.7, 0.2, 0.1),
                               seed=666, group_key_fn=burr_tile_group_key)

        def keys(fn):
            lines = [l.strip() for l in (out / fn).read_text().splitlines() if l.strip()]
            return {burr_tile_group_key(Path(l)) for l in lines}

        tr, va, te = keys("train.txt"), keys("val.txt"), keys("test.txt")
        assert tr and va and te, (len(tr), len(va), len(te))          # all non-empty
        assert not (tr & va), tr & va                                 # no leakage
        assert not (tr & te), tr & te
        assert not (va & te), va & te


def test_composite_objective():
    w = {"loss": 0.45, "f1": 0.35, "map50": 0.20}
    # Perfect quality -> only the loss term remains.
    assert abs(compute_composite_objective(2.0, 1.0, 1.0, w) - 0.45 * 2.0) < 1e-9
    # Degenerate short-circuits.
    assert compute_composite_objective(0.0, 0.5, 0.5, w) == 1e6   # val_loss <= 0
    assert compute_composite_objective(2.0, 0.0, 0.0, w) == 1e6   # f1<.01 and map50<.01
    # Worse quality -> larger objective.
    assert compute_composite_objective(2.0, 0.5, 0.5, w) > compute_composite_objective(2.0, 0.9, 0.9, w)
    # The point of the composite: a low-loss/low-quality config must score WORSE
    # than a higher-loss/high-quality one (defeats the loss-gain confound).
    low_loss_bad_quality = compute_composite_objective(1.0, 0.30, 0.30, w)
    high_loss_good_quality = compute_composite_objective(2.0, 0.85, 0.85, w)
    assert high_loss_good_quality < low_loss_bad_quality


def test_lr_cap_formula():
    """Documents the capped LR scaling (mirrors the inline formula in training.py)."""
    def scaled(lr0, eff, ref=64, power=0.5, max_lr0=0.01, cap=0.03):
        return min(min(lr0, max_lr0) * (eff / ref) ** power, cap)

    assert scaled(0.0075, 8) < scaled(0.0075, 512)          # grows with effective batch
    assert scaled(0.0075, 512) <= 0.03 + 1e-12             # never exceeds the ceiling
    # Old reference (/4) gave ~11.3x at eff 512; new reference (/64) gives ~2.83x.
    assert abs(scaled(0.0075, 512) - 0.0075 * (512 / 64) ** 0.5) < 1e-9
    # The max_scaled_lr ceiling binds only for very large effective batches.
    assert abs(scaled(1.0, 2304) - 0.03) < 1e-9            # lr0 clamped to 0.01, 0.01*sqrt(36)=0.06 -> cap


def test_core_region_dedup():
    """Core-region filtering assigns a seam burr to exactly one tile."""
    tiler = CanopyTiler(tile_size=224, overlap=0.2)  # stride 179, margin 22.5
    info = [
        {"tile_x": 0,   "tile_y": 0, "original_width": 403, "original_height": 224},
        {"tile_x": 179, "tile_y": 0, "original_width": 403, "original_height": 224},
    ]
    # One burr (global box center x=200) detected in BOTH overlapping tiles.
    det = [
        {"boxes": np.array([[190., 90., 210., 110.]]), "confidences": np.array([0.9]), "labels": np.array([0])},
        {"boxes": np.array([[11., 90., 31., 110.]]),   "confidences": np.array([0.9]), "labels": np.array([0])},
    ]
    old = tiler.reconstruct_detections(det, info)
    new = tiler.reconstruct_detections_core(det, info)
    assert len(old) == 2, len(old)   # plain reconstruction double-counts the seam
    assert len(new) == 1, len(new)   # core-region keeps it once

    # A burr near the right image boundary must be retained (boundary side untrimmed).
    info_edge = [{"tile_x": 179, "tile_y": 0, "original_width": 403, "original_height": 224}]
    det_edge = [{"boxes": np.array([[210., 90., 230., 110.]]), "confidences": np.array([0.9]), "labels": np.array([0])}]
    assert len(tiler.reconstruct_detections_core(det_edge, info_edge)) == 1


def test_set_trainable_layers_monotonic():
    try:
        from ultralytics import YOLO
    except Exception:
        print("SKIP test_set_trainable_layers_monotonic (ultralytics unavailable)")
        return
    from burr_detection.training import set_trainable_layers
    model = YOLO(str(_REPO / "yolo11n.pt"))
    counts = [set_trainable_layers(model, num_layers=n)["trainable_tensors"] for n in (1, 2, 3, 4)]
    assert counts == sorted(counts) and len(set(counts)) == len(counts), counts  # strictly increasing
    total_tensors = sum(1 for _ in model.model.parameters())
    assert counts[-1] == total_tensors, (counts[-1], total_tensors)              # stage 4 == all


def test_nan_sentinel():
    try:
        from ultralytics import YOLO  # noqa: F401
    except Exception:
        print("SKIP test_nan_sentinel (ultralytics unavailable)")
        return
    from types import SimpleNamespace
    from burr_detection.training import YOLOTrainer
    tr = YOLOTrainer(model_size=str(_REPO / "yolo11n.pt"))
    tr._original_stdout, tr._original_stderr = sys.stdout, sys.stderr
    tr.batch_idx = 5          # not 1, not a multiple of freq, not == num_batches -> skips the print path
    tr.num_batches = 1000
    fake = SimpleNamespace(
        epoch=0,
        loss_items=[float("nan"), 1.0, float("inf")],
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.01}]),
        nb=1000,
    )
    tr._on_batch_end(fake, print_freq=1000)
    assert tr.train_metric_logger.meters["box_loss"].value == 100.0
    assert tr.train_metric_logger.meters["dfl_loss"].value == 100.0
    assert tr.train_metric_logger.meters["cls_loss"].value == 1.0


def test_step_warmup_math():
    """The manual step-transition LR ramp (mirrors training._on_batch_start)."""
    def ramp(start, target, iters):
        return [start + min(1.0, idx / max(1, iters - 1)) * (target - start) for idx in range(iters)]
    seq = ramp(0.0, 0.02, 5)
    assert abs(seq[0] - 0.0) < 1e-9 and abs(seq[-1] - 0.02) < 1e-9
    assert all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))      # monotonic
    assert abs(ramp(0.001, 0.01, 3)[1] - 0.0055) < 1e-9               # midpoint


def test_optimizer_snapshot_restore():
    try:
        import torch
        from ultralytics import YOLO  # noqa: F401
    except Exception:
        print("SKIP test_optimizer_snapshot_restore (torch/ultralytics unavailable)")
        return
    from types import SimpleNamespace
    from burr_detection.training import YOLOTrainer

    tr = YOLOTrainer(model_size=str(_REPO / "yolo11n.pt"))
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(3):  # populate exp_avg / exp_avg_sq
        opt.zero_grad()
        model(torch.randn(8, 4)).sum().backward()
        opt.step()

    snap = tr._snapshot_optimizer_by_name(SimpleNamespace(model=model, optimizer=opt))
    assert snap and snap["state_by_name"], "snapshot is empty"
    name0 = next(iter(snap["state_by_name"]))
    ref = snap["state_by_name"][name0]["exp_avg"].clone()

    opt2 = torch.optim.AdamW(model.parameters(), lr=0.01)  # fresh, empty state
    restored = tr._restore_optimizer_by_name(SimpleNamespace(model=model, optimizer=opt2), snap)
    assert restored >= 1, restored
    p0 = dict(model.named_parameters())[name0]
    assert "exp_avg" in opt2.state.get(p0, {}), "restore did not populate optimizer state"
    assert torch.allclose(opt2.state[p0]["exp_avg"].cpu(), ref), "restored exp_avg mismatch"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
