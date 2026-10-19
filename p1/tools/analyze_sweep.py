"""Turn the sweep's per-run predictions into the curve that answers when to stop annotating.

FOUR THINGS IN HERE ARE DECISIONS RATHER THAN MECHANICS, and each is the answer to an objection.

THE OPERATING POINT IS CHOSEN LEAVE-ONE-FOLD-OUT. A detector's confidence threshold is a free
parameter, and picking it on the same trees the recall is reported on is selection on the test set.
The earlier six-arm table matched precision on its own thirty test trees, which is the comparison to
reproduce but not one to rely on. Here the threshold applied to fold f is chosen on the other four
folds' test trees, so it is out of sample, and it costs no reserved trees, which leaves the top of
the learning curve where it is.

MATCHING RUNS ONCE PER TREE, NOT ONCE PER THRESHOLD. Greedy matching in descending score order has
a property worth exploiting: lowering the confidence threshold only admits lower-scored
predictions, and those cannot take a truth box away from a higher-scored one. So whether a given
prediction matches is invariant to the threshold, and one pass yields every threshold's counts
exactly. That turns a 193,000-tree sweep over 31 thresholds from hours into seconds.

THE DECISION METRIC IS PER-TREE COUNT ACCURACY, AND TOP-DECILE GAIN CAPTURE IS A SECOND READING
THAT SATURATES TOO EARLY TO DECIDE ANYTHING. The counts are a quantitative trait that goes to
Breedbase and into genomic selection, so the measurement has to quantify actual crop load and not
only sort the trees. Capture was the original choice, on the argument that the model exists to rank
trees for selection and that set overlap ("how many of the true top 15 are in the model's top 15")
has a standard error near 0.10 on 154 trees, too coarse to read a curve from. Capture does fix the
coarseness, but it was never checked for headroom, and it has almost none: measured over the
finished cells it reaches about 0.97 by ten to fifteen annotated trees and then moves by less than
its own replicate noise, which is sd 0.040 on the pooled folds. Per-tree count error keeps falling
across the whole range, so that is what the curve is fitted to. Capture, overlap, recall, precision
and Spearman are all still reported.

WITHIN THE COUNT ERROR, THE AFFINE PART DOES NOT COUNT AGAINST THE DETECTOR AND THE SCATTER DOES.
A detector that undercounts every tree by a third, or compresses the heavy trees by a fixed slope,
costs a genomic model nothing: an intercept and a scale are absorbed by the model's own intercept
and scale, the ranking and the correlation are untouched, and the correction is recoverable from
the annotated trees after the fact. What is not recoverable is the scatter around that line,
because it attenuates the heritability of the measured trait by exactly the correlation. So
`mae`, `bias` and `slope` are reported as calibration diagnostics, while `resid_sd` (that scatter,
in burs a tree) and `pearson` are the figures the stopping rule is read from.

THE NOISE IS NOT CONSTANT ALONG THE CURVE, AND WEIGHTING FOR THAT DOES NOT HELP, WHICH IS A
MEASURED RESULT AGAINST THE OBVIOUS ARGUMENT. Over the 57 grid positions holding two or more
cells, a position's spread is proportional to its distance from a perfect detector: the correlation
between the mean and the spread is +0.66 for `resid_sd` and +0.79 for `mae`, where perfect is zero,
and -0.84 for capture, -0.84 for the correlation, -0.80 for Spearman and -0.76 for recall, where
perfect is one. So `fit_curves` can weight each point by one pooled coefficient of variation times
its level's distance from perfect, and over 120 simulated campaigns drawn from a power law with
that measured multiplicative noise it made the MARGINAL VALUE slightly worse, sd 0.0374 against
0.0326 burs per ten trees, while making the floor marginally better. The marginal is what the
stopping rule reads, so weighting is OFF by default and `weight=True` keeps it available. The
promising treatment is not a smooth weight: about 11 percent of cells produce a detector that
over-counts per tree, and dropping those alone cuts the pooled coefficient of variation from 0.300
to 0.204.

THE X AXIS IS HOURS, measured rather than assumed: seven working sessions in Ren's own
`annotation_stats.json` cover 130 images in 7.07 hours, which is 3.26 minutes an image and 33.4
seconds a bur. Ohio frames have no comparable rate, so they stay on their own frame and box axes.
"""

import argparse
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_PER_TREE = 3.26          # measured, see the module docstring
IOU = 0.3
# 0.15 to 0.95. The upper end was 0.90 until 2026-10-04, which silently CLIPPED every run whose
# own count-matching threshold sat above it; measured over 606 cells those thresholds reach 0.95,
# so a grid ending at 0.90 reports the edge of the grid rather than the run's operating point.
THRESHOLDS = np.round(np.arange(0.15, 0.9501, 0.025), 4)

# A PREDICTED BOX LARGER THAN ANY POSSIBLE BUR IS NOT A BUR. This is a validity filter, not a
# modelling choice, and it is the fix for the over-counting runs that no threshold could reach.
#
# MEASURED over 41,212 annotated burs in 121 filed results: the larger side of a hand-annotated bur
# has median 68.4 px, p99 90.4, and a MAXIMUM of 108.0. The largest prediction that ever matched a
# bur is 116.3. Meanwhile the predictions that match NOTHING have p99 624 px and a maximum of 640,
# which is the whole tile edge: some runs emit a box spanning an entire tile. A 130 px gate
# therefore removes 0 of 41,212 true burs, 0 of 35,026 matched predictions, and 1,339 of 19,196
# unmatched predictions, 7.0 percent. Nothing true is lost at any cut from 116 px to 300 px.
#
# WHY IT MATTERS MORE THAN THE THRESHOLD. The five-fold rerun sets on CHTC, 60 runs over
# six design points, show the bad runs emit these large boxes rather than extra bur-sized ones: in
# one pair of the same cell, boxes matching no bur were 750 against 21, at median 192 by 352 px and
# median confidence 0.95. The training's own validation cannot see them, because validation runs on
# object tiles plus sampled background while inference covers the whole frame; that run reported
# HIGHER validation precision than its good twin. Per run over all 60: off by more than 2 burs a
# tree 25 at a fixed 0.25, 3 at each run's own threshold, 0 with the gate; runs with a per-tree
# correlation under 0.85, 10, then 7, then 0. The rerun noise of the pooled metric falls from 1.99
# to 1.95 to 0.24 burs, so the gate is worth about eight times what the threshold is worth.
#
# THE GATE IS IN PIXELS AND HERE IS THE PHYSICAL FORM OF IT (corrected 2026-10-05 and 2026-10-08). The
# P1's nominal 35 mm gives an angular pixel of 4.3823 um / 35 mm = 1.2521e-4 rad, but self-calibration in
# six ODM reconstructions puts the effective focal length near 35.8 mm, an angular pixel of about
# 1.225e-4 rad, and the 65R-to-P1 scale measured on same-waypoint pairs agrees to within 1.5 percent. So
#
#     GSD in mm/px = 0.1225 * standoff_m          gate_px = 1023 / standoff_m
#
# where the standoff is CAMERA TO CROWN, not camera to ground. These hover missions hold it at a
# design constant of 7.87 m over all 4,028 waypoints of all 13 missions, sd 0.00, so the GSD is about
# 0.96 mm/px (0.985 with the nominal focal length) and 130 px is 12.5 cm, roughly twice a bur.
#
# IT WAS STATED HERE AS 14 cm AT 1.11 mm/px AND THAT WAS WRONG BY 13 PERCENT, because 1.11 came
# from `drone-dji:RelativeAltitude`, which is height above the TAKE-OFF POINT rather than above
# ground. Clarks Run's per-frame values run -4.29 to +14.18 m, and a negative height above ground
# is impossible, which is what proves the quantity rather than the reading was wrong. THE GATE
# ITSELF DOES NOT MOVE: it was set in pixels from the measured annotation sizes, so only its
# physical reading changes.
#
# For deployment the rule is the same shape as the threshold: a count needs its box sizes checked
# against what a bur can be at that standoff. Take the standoff from the mission manifest rather
# than from EXIF, and for a mission not planned from crown heights take it from the surface, as
# camera absolute altitude minus crown elevation. The angular pixel belongs to the camera and lens,
# not to this campaign.
MAX_BOX_PX = 130

# The within-fold per-tree correlation below which a run counts as FAILED rather than noisy. The
# 30 rerun pairs on CHTC put both runs of a cell either side of about 0.94, and every run that
# emitted oversized boxes fell under 0.85, so the cut separates two populations rather than slicing
# one. See `folds_lowr` in emit().
LOW_R = 0.85


def iou_matrix(pred, truth):
    if not pred or not truth:
        return np.zeros((len(pred), len(truth)))
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    ix = np.maximum(0.0, np.minimum(p[:, None, 2], t[None, :, 2])
                    - np.maximum(p[:, None, 0], t[None, :, 0]))
    iy = np.maximum(0.0, np.minimum(p[:, None, 3], t[None, :, 3])
                    - np.maximum(p[:, None, 1], t[None, :, 1]))
    inter = ix * iy
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    at = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    union = ap[:, None] + at[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def match_once(boxes, scores, truth):
    """Greedy match in descending score order. Returns each prediction's score and whether it hit.

    Threshold-invariant, which is the point: a lower-scored prediction can never steal a truth box
    from a higher-scored one, so the single pass answers every threshold.
    """
    if not boxes:
        return np.zeros(0), np.zeros(0, dtype=bool)
    order = np.argsort(-np.asarray(scores))
    m = iou_matrix([boxes[i] for i in order], truth)
    taken = np.zeros(len(truth), dtype=bool)
    hit = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if not len(truth):
            break
        row = np.where(taken, -1.0, m[i])
        j = int(np.argmax(row)) if row.size else -1
        if j >= 0 and row[j] >= IOU:
            taken[j] = True
            hit[i] = True
    return np.asarray(scores)[order], hit


def per_tree_curves(trees, max_box_px=None):
    """For each tree, the arrays needed to evaluate every threshold in O(1).

    `max_box_px` drops any PREDICTED box whose larger side exceeds it, before matching. The truth is
    never filtered. See MAX_BOX_PX for why this is a validity filter rather than a modelling choice.
    """
    out = []
    for t in trees:
        boxes, scores = t["boxes"], t["scores"]
        if max_box_px and boxes:
            keep = [i for i, b in enumerate(boxes)
                    if max(b[2] - b[0], b[3] - b[1]) <= max_box_px]
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
        s, hit = match_once(boxes, scores, t["truth"])
        out.append(dict(tree=t["tree"], split=t.get("split", "test"), n_truth=len(t["truth"]),
                        scores=s, hit=hit))
    return out


def counts_at(tc, thr):
    """Return (tp, fp, n_pred) for one tree at one threshold."""
    k = int(np.searchsorted(-tc["scores"], -thr, side="right")) if tc["scores"].size else 0
    if k == 0:
        return 0, 0, 0
    tp = int(tc["hit"][:k].sum())
    return tp, k - tp, k


def aggregate(tcs, thr):
    tp = fp = npred = ntruth = 0
    pred_c, true_c = [], []
    for tc in tcs:
        a, b, n = counts_at(tc, thr)
        tp += a
        fp += b
        npred += n
        ntruth += tc["n_truth"]
        pred_c.append(n)
        true_c.append(tc["n_truth"])
    return dict(tp=tp, fp=fp, n_pred=npred, n_truth=ntruth,
                precision=tp / npred if npred else 0.0,
                recall=tp / ntruth if ntruth else 0.0,
                count_ratio=npred / ntruth if ntruth else 0.0,
                pred=np.array(pred_c), true=np.array(true_c))


def spearman(a, b):
    if len(a) < 3:
        return float("nan")

    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        # average ties, which matter here: many trees share a low bur count
        _, start, cnt = np.unique(x[o], return_index=True, return_counts=True)
        for s, c in zip(start, cnt):
            if c > 1:
                r[o[s:s + c]] = r[o[s:s + c]].mean()
        return r

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d else float("nan")


def affine_fit(x, y):
    """Least squares y ~ a + b*x, returning (a, b, se_b).

    `se_b` is what makes the calibration usable: applying a fitted slope to a held-out fold
    divides by that slope, so a slope the data do not determine produces an arbitrarily large
    calibrated error rather than a large one. Returns a zero slope and an infinite standard error
    when x does not vary, which happens for a cell whose detector predicts the same count on
    every tree.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 3:
        return (float(y.mean()) if y.size else 0.0, 0.0, float("inf"))
    vx = float(((x - x.mean()) ** 2).sum())
    if vx <= 0:
        return (float(y.mean()), 0.0, float("inf"))
    b = float(((x - x.mean()) * (y - y.mean())).sum() / vx)
    a = float(y.mean() - b * x.mean())
    resid = y - (a + b * x)
    s2 = float((resid ** 2).sum()) / (x.size - 2)
    return (a, b, math.sqrt(s2 / vx) if vx > 0 else float("inf"))


def fn_load(n, tp):
    """How a tree's MISS RATE moves with its CROP LOAD, the covariance the breeding use turns on.

    A detector that cannot tell one bur from a cluster of burs misses a larger share of a heavy tree,
    which compresses the top of the distribution where a breeder selects. The bur-weighted (pooled)
    miss rate and the mean of the per-tree miss rates differ by EXACTLY that covariance:

        pooled - mean_i(f_i) = Cov_pop(n_i, f_i) / mean_i(n_i),   f_i = (n_i - tp_i) / n_i

    so `fn_gap` is the covariance in the units of a miss rate, and the criterion set for the
    detector is a pooled rate of at most 0.10 with |fn_gap| at most 0.05. A positive gap means
    heavier trees lose a larger share of their burs.

    Matched boxes against annotation, so f_i lies in [0, 1]. A COUNT-based rate does not: on a
    light tree a detector that over-counts gives f_i far below zero, and on 780 field-counted trees
    one tree with a single counted bur and 17 detections contributed -16 to the mean on its own.
    Here tp_i can never exceed n_i, so that instability cannot arise. What this cannot see is a bur
    hidden from the camera, which neither the annotator nor the detector records; that part of the
    load dependence needs field counts.
    """
    nan = float("nan")
    keys = ("fn_pooled", "fn_tree_mean", "fn_gap", "fn_cov_n", "fn_load_slope", "fn_load_corr")
    ok = n > 0
    n, tp = n[ok], tp[ok]
    if n.size < 3:
        return {**{k: nan for k in keys}, "fn_trees": int(n.size)}
    f = 1.0 - tp / n
    pooled = 1.0 - tp.sum() / n.sum()
    cov = float(np.mean((n - n.mean()) * (f - f.mean())))       # population, so the identity is exact
    gap = float(pooled - f.mean())
    assert abs(gap - cov / n.mean()) < 1e-9, (gap, cov / n.mean())
    var = float(np.mean((n - n.mean()) ** 2))
    return dict(fn_pooled=float(pooled), fn_tree_mean=float(f.mean()), fn_gap=gap, fn_cov_n=cov,
                fn_load_slope=cov / var if var > 0 else nan, fn_load_corr=pearson(n, f),
                fn_trees=int(n.size))


def count_errors(pred, true, cal_pred=None):
    """Per-tree count error in burs, which is the unit the yield phenotype is recorded in.

    `slope` regresses PREDICTED on TRUE, so a value below one is the undercount on heavy trees.

    `resid_sd` IS THE FIGURE THE STOPPING RULE READS, in burs a tree: the scatter that survives
    the best affine correction, computed as sd(true) * sqrt(1 - r^2), which is exactly the
    residual standard deviation of that correction. It is BOUNDED by the between-tree spread of
    the truth, so a detector with no signal reads as "recovers none of the spread" rather than as
    a large number. The first version of this computed it by fitting the correction on four folds
    and applying it to the fifth, which is the purer design and is numerically useless at the left
    end of the curve: the correction divides by the fitted slope, and on the 20 of 163 cells whose
    detector had almost no signal the slope was not determined and the calibrated error ran up to
    47 burs a tree against a raw mean absolute error of 8.6. `cal_rmse` keeps that out-of-fold
    reading as a check, blanked unless the slope is determined to better than two standard errors
    on the folds it was fitted from; where both are finite they should agree.
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    n = int(true.size)
    if n == 0:
        # every key is still emitted, because the CSV writer takes its field list from the first
        # row and a row with a shorter key set would silently drop columns for every later row
        return dict(mean_true=float("nan"), true_sd=float("nan"), mae=float("nan"),
                    rmse=float("nan"), bias=float("nan"), slope=float("nan"),
                    pearson=float("nan"), r2=float("nan"), nmae=float("nan"),
                    resid_sd=float("nan"), resid_frac=float("nan"),
                    cal_mae=float("nan"), cal_rmse=float("nan"))
    err = pred - true
    r = pearson(pred, true)
    mt = float(true.mean())
    sdt = float(true.std())
    frac = math.sqrt(max(0.0, 1.0 - r * r)) if np.isfinite(r) else float("nan")
    out = dict(
        mean_true=round(mt, 4),
        true_sd=round(sdt, 4),
        mae=round(float(np.abs(err).mean()), 4),
        rmse=round(float(math.sqrt(float((err ** 2).mean()))), 4),
        bias=round(float(err.mean()), 4),
        slope=round(affine_fit(true, pred)[1], 4),
        pearson=round(r, 4) if np.isfinite(r) else float("nan"),
        r2=round(r * r, 4) if np.isfinite(r) else float("nan"),
        nmae=round(float(np.abs(err).mean()) / mt, 4) if mt else float("nan"),
        resid_sd=round(sdt * frac, 4) if np.isfinite(frac) else float("nan"),
        # the fraction of the between-tree spread in true crop load that the detector does NOT
        # recover, which is one minus the attenuation a genomic model would suffer
        resid_frac=round(frac, 4) if np.isfinite(frac) else float("nan"),
    )
    if cal_pred is None:
        out.update(cal_mae=float("nan"), cal_rmse=float("nan"))
        return out
    cerr = np.asarray(cal_pred, float) - true
    ok = np.isfinite(cerr)
    if not ok.all():
        out.update(cal_mae=float("nan"), cal_rmse=float("nan"))
        return out
    out.update(
        cal_mae=round(float(np.abs(cerr).mean()), 4),
        cal_rmse=round(float(math.sqrt(float((cerr ** 2).mean()))), 4),
    )
    return out


def gain_capture(pred, true, frac):
    """Mean true count of the model's top k over the mean true count of the true top k."""
    n = len(true)
    k = max(1, int(round(frac * n)))
    # break prediction ties by true count DESCENDING would flatter the model, so break at random
    # with a fixed seed instead
    rng = np.random.default_rng(0)
    jitter = rng.random(n) * 1e-6
    sel = np.argsort(-(pred + jitter))[:k]
    best = np.sort(true)[::-1][:k]
    got = true[sel]
    overlap = len(set(sel.tolist()) & set(np.argsort(-(true + jitter))[:k].tolist()))
    return dict(k=k, capture=float(got.mean() / best.mean()) if best.mean() else float("nan"),
                overlap=overlap / k, mean_true_selected=float(got.mean()),
                mean_true_best=float(best.mean()))


def discover_results_trees(roots):
    """Every directory under `roots` holding result files, classified so repeats are not merged.

    THE PROBLEM DISCOVERY HAS TO SOLVE. Some trees are DELIBERATE REPEATS: the cross-check and
    bridge cells were run again on a second facility under the SAME run id, specifically to measure
    the facility difference, and merging them into the pooled analysis would silently average two
    different machines' answers for one cell. The loader is right to refuse a duplicate run id, so
    naive discovery would simply always fail.

    THE RULE: trees are taken largest first, and a tree that contributes NO new run id is a repeat
    and is reported and skipped. A tree contributing some new ids AND some duplicates is a genuine
    conflict and is raised, because that is two submissions racing on one cell rather than a
    designed re-run. Largest first is not arbitrary: a repeat tree is a small deliberate re-run of
    cells that already exist elsewhere, so it is contained in what is already loaded.
    """
    # AN EXPLICIT OPT OUT, because the size heuristic below is the wrong thing to rely on for a
    # tree somebody MEANT to keep separate. A directory holding a file named NOT_CAMPAIGN_RESULTS,
    # or sitting under one, is never loaded and never classified. Deliberate second runs of cells
    # that already exist belong here: they are a measurement of run-to-run variation, not more
    # campaign cells, and the repeat rule would reach the same answer only as long as the first-run
    # tree stays the larger of the two. An order-dependent accident is not a guarantee.
    opted_out = set()
    for root in roots:
        for m in Path(root).rglob("NOT_CAMPAIGN_RESULTS"):
            opted_out.add(m.parent.resolve())

    def excluded(d):
        rd = d.resolve()
        return any(rd == o or o in rd.parents for o in opted_out)

    trees, skipped_marker = {}, set()
    for root in roots:
        for f in Path(root).rglob("*.json.gz"):
            if excluded(f.parent):
                skipped_marker.add(f.parent)
                continue
            trees.setdefault(f.parent, set()).add(f.name[:-8])
    for d in sorted(skipped_marker):
        print(f"  EXCLUDED {d}  (NOT_CAMPAIGN_RESULTS marker)")
    primary, seen, repeats, conflicts = [], set(), [], []
    for d, ids in sorted(trees.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        new, dup = ids - seen, ids & seen
        if not new:
            repeats.append((d, len(ids)))
        elif dup:
            conflicts.append((d, sorted(dup)[:5], len(dup)))
        else:
            primary.append(d)
            seen |= ids
    print(f"discovered {len(trees)} results tree(s) under {len(roots)} root(s): "
          f"{len(primary)} loaded, {len(repeats)} deliberate repeat(s) skipped")
    for d in primary:
        print(f"  load   {d}  ({len(trees[d])} results)")
    for d, n in repeats:
        print(f"  SKIP   {d}  ({n} results, every id already present: a designed re-run of cells "
              f"that exist in another tree, such as the cross-facility bridge)")
    if conflicts:
        lines = "; ".join(f"{d} shares {n} id(s) e.g. {ex}" for d, ex, n in conflicts)
        raise SystemExit(
            f"CONFLICTING results trees, which is two submissions claiming one cell rather than a "
            f"designed re-run: {lines}. Resolve it rather than letting one win.")
    return [str(d) for d in primary]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--discover-under", action="append",
                    help="a directory to search for results trees, recursively. Repeatable. Use "
                         "this INSTEAD of listing --results by hand. THREE TIMES on 2026-10-04 the "
                         "analysis turned out to be reading a subset of the campaign because a "
                         "results path was not passed: 838 of 1,593 when the VT trees were absent, "
                         "then 1,593 of 1,957 when tranches 6, 3-and-5 and 2 each sat in their own "
                         "directory. An --expect-runs list does not catch that, because it only "
                         "covers what somebody remembered to list. Discovery cannot be defeated by "
                         "forgetting an argument, which is the only version of this guard that "
                         "holds.")
    ap.add_argument("--results", action="append",
                    help="repeatable: one results directory per submission. The campaign is "
                         "submitted as more than one array and each writes its own tree, so the "
                         "analysis reads them all. Run ids are globally unique, so there is no "
                         "collision and a cell appearing twice would be a real duplicate.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixed-threshold", type=float, default=0.25)
    ap.add_argument("--expect-runs", action="append",
                    help="file of run ids, one per line, that this analysis MUST find. Repeatable. "
                         "THE LOADER CANNOT CHECK ITSELF: it reported 'loaded 838 of 838' while "
                         "reading 838 of 1,593 files, because both numbers came from the same glob "
                         "and a count compared to itself can only ever succeed. That omitted 44 "
                         "percent of the campaign's measurements and hid an entire top-level Ohio "
                         "treatment, every cell of which lived on the cluster whose results were "
                         "not being read. Pass the done-flag or task lists here and the expectation "
                         "comes from outside the loader.")
    ap.add_argument("--max-box-px", type=float, default=MAX_BOX_PX,
                    help="drop a predicted box whose larger side exceeds this, before matching. "
                         "See MAX_BOX_PX: no annotated bur reaches 110 px and nothing true is lost "
                         "at any cut from 116 to 300. Pass 0 to disable and reproduce the ungated "
                         "numbers.")
    ap.add_argument("--nathan-level", action="append",
                    help="Ohio frame level to print the Clarks Run curve at; repeatable. The "
                         "levels are never pooled, because Ohio data moves the count error by "
                         "more than several Clarks Run levels do. Default: every level that has "
                         "at least 5 Clarks Run levels present, highest first.")
    a = ap.parse_args()

    if not a.results and not a.discover_under:
        ap.error("pass --discover-under (preferred) or --results")
    if a.discover_under:
        a.results = list(a.results or []) + discover_results_trees(a.discover_under)

    design = json.load(open(a.design))
    runs = {r["run_id"]: r for r in design["runs"]}

    # ---- load, and match once per tree ------------------------------------------------
    loaded = {}
    # RECURSIVE, because a results tree pulled off a cluster keeps its per-submission subdirectories
    # (`t1/`, `rest/`, `fal_a30/`, `fal_l40s/`) while one assembled on an access point is flat. A
    # top-level glob read 838 of 1,593 files and said "838 of 838", which is the shape of silent
    # omission that looks like success: the count it reports is the count it found. Duplicate run
    # ids are already refused below, so a recursive walk cannot double-count.
    files = sorted(f for d in a.results for f in Path(d).rglob("*.json.gz"))
    seen = {}
    for i, f in enumerate(files):
        rid = f.name[:-8]
        if rid not in runs:
            continue
        if rid in seen:
            raise SystemExit(f"{rid} appears in two results trees, {seen[rid]} and {f}; "
                             f"two arrays ran the same cell and one result would be discarded")
        seen[rid] = f
        with gzip.open(f, "rt") as fh:
            d = json.load(fh)
        loaded[rid] = dict(run=runs[rid], tcs=per_tree_curves(d["trees"]),
                           tcsg=per_tree_curves(d["trees"], max_box_px=a.max_box_px),
                           epochs=d.get("epochs_run"), best_epoch=d.get("best_epoch"),
                           train_s=d.get("train_seconds"), best_val=d.get("best_val", {}),
                           n_train_tiles=d.get("n_train_tiles"))
        if (i + 1) % 200 == 0:
            print(f"  matched {i+1}/{len(files)}", flush=True)
    # SAY WHERE THE DENOMINATOR CAME FROM. "loaded N of N" invited exactly the error it then
    # concealed: both numbers are the same glob, so the line reads as a reconciliation and is not
    # one.
    print(f"loaded {len(loaded)} result files, {len(files)} found by the glob under "
          f"{len(a.results)} results path(s), {len(runs)} runs in the design")
    if a.expect_runs:
        want = set()
        for path in a.expect_runs:
            ids = [x for x in Path(path).read_text().split() if re.fullmatch(r"r\d+", x)]
            want |= set(ids)
            print(f"  expecting {len(ids)} run ids from {path}")
        missing = sorted(want - set(loaded))
        if missing:
            raise SystemExit(
                f"EXPECTED {len(want)} run ids and {len(missing)} were NOT loaded, for example "
                f"{missing[:8]}. Either a results path is absent or the glob did not reach it; a "
                f"results tree pulled off a cluster keeps per-submission subdirectories while one "
                f"assembled on an access point is flat.")
        print(f"  VERIFIED: all {len(want)} expected run ids were loaded")

    # ---- group into cells: everything but the fold ------------------------------------
    def cellkey(r):
        return (r["tranche"], r["arm"], r["scheme"], r["seed"], r["rule"],
                r.get("nathan_rule", "random"), r["n_ren"], r["n_nathan"], r["tile"])

    cells = defaultdict(dict)
    for rid, L in loaded.items():
        cells[cellkey(L["run"])][L["run"]["fold"]] = L

    rows = []
    for key, byfold in sorted(cells.items()):
        tranche, arm, scheme, seed, rule, nat_rule, n_ren, n_nat, tile = key
        folds = sorted(byfold)
        if len(folds) < 2:
            continue                                  # leave-one-fold-out needs at least two

        test = {f: [t for t in byfold[f]["tcs"] if t["split"] == "test"] for f in folds}
        vtrees = {f: [t for t in byfold[f]["tcs"] if t["split"] != "test"] for f in folds}
        testg = {f: [t for t in byfold[f]["tcsg"] if t["split"] == "test"] for f in folds}
        vtreesg = {f: [t for t in byfold[f]["tcsg"] if t["split"] != "test"] for f in folds}

        # the operating point for fold f, chosen on the OTHER folds by best F1
        thr_for = {}
        for f in folds:
            other = [t for g in folds if g != f for t in test[g]]
            bestf1, bestthr = -1.0, THRESHOLDS[0]
            for thr in THRESHOLDS:
                ag = aggregate(other, thr)
                p, rc = ag["precision"], ag["recall"]
                f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
                if f1 > bestf1:
                    bestf1, bestthr = f1, thr
            thr_for[f] = float(bestthr)

        # EACH RUN'S OWN THRESHOLD, SET ON ITS OWN RESERVED VALIDATION TREES. This is the operating
        # point the campaign should read, and the reason is measured rather than argued.
        #
        # The leave-one-fold-out rule above gives fold f a threshold fitted to four OTHER models,
        # and a model's confidence SCALE is not reproducible: over 606 cells a run's count-matching
        # threshold has median 0.40 and sd 0.12, and between two runs of the SAME cell it
        # correlates only +0.46. So about one run in nine lands with its extra boxes between 0.25
        # and 0.5 and over-counts per tree by more than 2 burs, at a per-tree correlation near 0.73
        # against 0.94 for the rest. That is not a property of the training set: rerunning the same
        # cell flips the outcome, and the bias of two runs of one cell correlates -0.10, which is
        # what independent draws give. It is not visible in mAP either, since those runs validate
        # slightly ABOVE the rest, and the cause is most likely the unseeded rotation stream.
        #
        # Measured over 606 cells on the test trees, per-tree bias: a fixed 0.25 gives median +1.47
        # with sd 5.08 and 222 runs over by more than 2 burs; the best SHARED threshold chosen with
        # hindsight on the test trees gives -0.27, sd 2.10, 28 over; each run's OWN validation
        # trees give -0.06, sd 1.07, 18 over, and above 32 Clarks Run trees leave zero of 211 runs
        # off by more than 2 in either direction. The own rule beats a threshold chosen with
        # hindsight on the very trees being scored, which is the result that settles it.
        #
        # The validation trees are not pristine, and that is stated rather than hidden: they also
        # choose the checkpoint. They are never trained on and never scored as test, and the
        # checkpoint is chosen by tile-level mAP while this is a per-tree count match, so the two
        # uses are close to orthogonal. In field terms the rule says a deployed model's threshold
        # must be set on a few annotated trees from the target planting, and 18 were enough here.
        def own_threshold(vt):
            """The confidence at which this run's predicted total matches the truth on its own
            reserved validation trees. One number from an aggregate of about 140 burs, which is why
            it is well determined where a two-parameter per-fold calibration on the same 18 trees
            is not."""
            if not vt:
                return float("nan")
            best, bestthr = None, float("nan")
            for thr in THRESHOLDS:
                ag = aggregate(vt, thr)
                dd = abs(ag["n_pred"] - ag["n_truth"])
                if best is None or dd < best:
                    best, bestthr = dd, float(thr)
            return bestthr

        thr_own = {f: own_threshold(vtrees[f]) for f in folds}
        # The gate changes which boxes exist, so the threshold must be re-chosen on the gated
        # predictions rather than reused. Reusing it would score a gated detector at an operating
        # point fitted to an ungated one.
        thr_gate = {f: own_threshold(vtreesg[f]) for f in folds}

        # Pooled out-of-fold predictions: every one of the 154 trees, each scored by a model that
        # never saw it, at a threshold that never saw it either. Emitted once for all folds, and
        # once per fold with that fold's trees left OUT, which is the jackknife replicate the
        # error bar is built from.
        r0 = byfold[folds[0]]["run"]

        def emit(use, omit):
            tp = fp = npred = ntruth = 0
            pp, pt = [], []
            for f in use:
                ag = aggregate(test[f], thr_for[f])
                pp.append(ag["pred"])
                pt.append(ag["true"])
                tp += ag["tp"]
                fp += ag["fp"]
                npred += ag["n_pred"]
                ntruth += ag["n_truth"]
            pred = np.concatenate(pp) if pp else np.zeros(0)
            true = np.concatenate(pt) if pt else np.zeros(0)
            # THE CALIBRATION IS FITTED LEAVE-ONE-FOLD-OUT, exactly as the threshold is. Fold f's
            # predicted counts are rescaled by an affine fit from the OTHER folds in this pool, so
            # no tree contributes to its own correction and `cal_rmse` is not a two-parameter fit
            # scored on its own data. With a single fold there is nothing to fit on, so the
            # calibrated columns stay absent rather than being fitted in sample.
            # TWO CONDITIONS, AND THE SECOND IS THE ONE THAT MATTERS. The correction multiplies the
            # held-out fold's predictions by the fitted slope, so it also multiplies that fold's
            # prediction noise. A slope the fitting folds do not determine to two standard errors
            # is not usable, which is the obvious condition and turns out to catch almost nothing:
            # the slope is b = r * sd(true) / sd(pred), so the blow-up case is a detector whose
            # predictions barely vary, which gives a slope that is LARGE and perfectly well
            # determined. Guarding only on significance left a cell reading 47.3 burs a tree at
            # r-squared 0.032. The amplification itself is therefore bounded: a correction is not
            # applied if it would stretch the predictions by more than a factor of three, since
            # past that the calibrated value is an extrapolation from a detector with no range.
            # A blanked fold blanks the whole cell, rather than leaving a pool half corrected.
            cal = None
            if len(use) >= 2:
                parts = []
                for i in range(len(use)):
                    ox = np.concatenate([pp[j] for j in range(len(use)) if j != i])
                    oy = np.concatenate([pt[j] for j in range(len(use)) if j != i])
                    a0, b0, se0 = affine_fit(ox, oy)
                    if not np.isfinite(se0) or abs(b0) <= 2 * se0 or abs(b0) > 3.0:
                        parts.append(np.full(pp[i].shape, np.nan))
                    else:
                        parts.append(a0 + b0 * pp[i])
                cal = np.concatenate(parts)
            # a selection-free second reading at one fixed threshold for every cell alike
            fx = aggregate([t for f in use for t in test[f]], a.fixed_threshold)
            row = dict(
                tranche=tranche, arm=arm, scheme=scheme, seed=seed, rule=rule,
                nathan_rule=nat_rule, n_ren=n_ren, n_nathan=n_nat, tile=tile,
                omit_fold=omit, folds=len(use),
                ren_train_boxes=r0["ren_train_boxes"],
                nathan_train_boxes=r0["nathan_train_boxes"],
                ren_hours=round(n_ren * MIN_PER_TREE / 60, 2),
                n_trees_scored=int(len(true)), n_burs_scored=int(true.sum()),
                thr_median=float(np.median([thr_for[f] for f in use])),
                precision=round(tp / npred, 4) if npred else 0.0,
                recall=round(tp / ntruth, 4) if ntruth else 0.0,
                count_ratio=round(npred / ntruth, 4) if ntruth else 0.0,
                spearman=round(spearman(pred, true), 4),
                fixed_precision=round(fx["precision"], 4),
                fixed_recall=round(fx["recall"], 4),
                fixed_spearman=round(spearman(fx["pred"], fx["true"]), 4),
                mean_epochs=round(np.mean([byfold[f]["epochs"] or 0 for f in use]), 1),
                mean_best_epoch=round(np.mean([byfold[f]["best_epoch"] or 0 for f in use]), 1),
                mean_train_minutes=round(
                    np.mean([(byfold[f]["train_s"] or 0) / 60 for f in use]), 1),
                mean_train_tiles=int(np.mean([byfold[f]["n_train_tiles"] or 0 for f in use])),
            )
            row.update(count_errors(pred, true, cal))

            # THE SAME COUNTS UNDER EACH RUN'S OWN THRESHOLD, prefixed `own_`, so the two operating
            # points sit side by side in one table and neither has to be taken on trust.
            if all(np.isfinite(thr_own[f]) for f in use):
                op, ot = [], []
                for f in use:
                    ag = aggregate(test[f], thr_own[f])
                    op.append(ag["pred"])
                    ot.append(ag["true"])
                oc = count_errors(np.concatenate(op), np.concatenate(ot))
                row.update({f"own_{k}": v for k, v in oc.items()
                            if k not in ("mean_true", "true_sd")})
                row["thr_own_median"] = float(np.median([thr_own[f] for f in use]))
            else:
                for k in ("mae", "rmse", "bias", "slope", "pearson", "r2", "nmae",
                          "resid_sd", "resid_frac", "cal_mae", "cal_rmse"):
                    row[f"own_{k}"] = float("nan")
                row["thr_own_median"] = float("nan")

            # THE SAME COUNTS WITH THE SIZE GATE IN, prefixed `gate_`, at a threshold re-chosen on
            # the gated predictions. This is the reading to lead with: it is the only one of the
            # three under which a rerun of the same cell reproduces its own per-tree bias.
            if a.max_box_px and all(np.isfinite(thr_gate[f]) for f in use):
                gp, gt = [], []
                for f in use:
                    ag = aggregate(testg[f], thr_gate[f])
                    gp.append(ag["pred"])
                    gt.append(ag["true"])
                gc = count_errors(np.concatenate(gp), np.concatenate(gt))
                row.update({f"gate_{k}": v for k, v in gc.items()
                            if k not in ("mean_true", "true_sd")})
                row["thr_gate_median"] = float(np.median([thr_gate[f] for f in use]))
                # the miss rate against crop load, tree by tree, at each run's own gated threshold
                nn, tt = [], []
                for f in use:
                    for tc in testg[f]:
                        tp_i, _, _ = counts_at(tc, thr_gate[f])
                        nn.append(tc["n_truth"])
                        tt.append(tp_i)
                row.update({f"gate_{k}": v for k, v in
                            fn_load(np.array(nn, float), np.array(tt, float)).items()})
            else:
                for k in ("mae", "rmse", "bias", "slope", "pearson", "r2", "nmae",
                          "resid_sd", "resid_frac", "cal_mae", "cal_rmse", "fn_pooled",
                          "fn_tree_mean", "fn_gap", "fn_cov_n", "fn_load_slope", "fn_load_corr",
                          "fn_trees"):
                    row[f"gate_{k}"] = float("nan")
                row["thr_gate_median"] = float("nan")

            # WITHIN-FOLD READINGS, WHICH SEPARATE A REAL DETECTOR EFFECT FROM A POOLING ARTIFACT.
            # Everything above pools five models' predictions into one array of trees, so a single
            # fold whose model sits a few burs high depresses the POOLED correlation even when
            # every model is internally well ordered. The mean of the per-fold correlations does
            # not have that property, and the gap between the two is the size of the between-fold
            # offset. `folds_off2_*` counts how many folds are off by more than 2 burs a tree,
            # which is the figure to compare between the two operating points.
            def per_fold(thr_map, which=None):
                src = which if which is not None else test
                rs, offs, low, lowf = [], 0, 0, []
                for f in use:
                    if not np.isfinite(thr_map[f]):
                        return float("nan"), float("nan"), float("nan"), ""
                    ag = aggregate(src[f], thr_map[f])
                    r = pearson(ag["pred"], ag["true"])
                    if np.isfinite(r):
                        rs.append(r)
                        # A FAILED RUN, not a slightly worse one. The CHTC rerun sets put every
                        # pair of a cell either side of about 0.94, so a within-fold per-tree
                        # correlation under 0.85 marks a run whose ranking inside its own fold has
                        # broken rather than one that is merely noisy. The FRACTION of runs in that
                        # state is a learning-curve output in its own right: measured over 838
                        # filed cells it is 31 percent at zero annotated trees, 20 percent at one,
                        # and ZERO of 351 cells at 32 trees or more, so it is a real cost of
                        # annotating less and appears in no averaged metric.
                        if r < LOW_R:
                            low += 1
                            lowf.append(str(f))
                    if abs(float(ag["pred"].mean() - ag["true"].mean())) > 2.0:
                        offs += 1
                return ((float(np.mean(rs)) if rs else float("nan")), offs, low, "|".join(lowf))

            wr, off, lw, lf = per_fold(thr_for)
            row["within_pearson"] = round(wr, 4) if np.isfinite(wr) else float("nan")
            row["folds_off2"], row["folds_lowr"] = off, lw
            wro, offo, lwo, lfo = per_fold(thr_own)
            row["own_within_pearson"] = round(wro, 4) if np.isfinite(wro) else float("nan")
            row["own_folds_off2"], row["own_folds_lowr"] = offo, lwo
            wrg, offg, lwg, lfg = per_fold(thr_gate, testg)
            row["gate_within_pearson"] = round(wrg, 4) if np.isfinite(wrg) else float("nan")
            row["gate_folds_off2"], row["gate_folds_lowr"] = offg, lwg
            # WHICH folds failed, not just how many: 21 of 47 failures were in fold 3
            # against 11, 2 and 13 in folds 1, 2 and 4, and the fold is the jackknife's block, so a
            # failure rate concentrated in one fold widens the interval for a reason worth naming.
            row["gate_lowr_folds"] = lfg
            for frac, name in ((0.05, "top05"), (0.10, "top10"), (0.20, "top20")):
                g = gain_capture(pred, true, frac)
                row[f"{name}_capture"] = round(g["capture"], 4)
                row[f"{name}_overlap"] = round(g["overlap"], 4)
                row[f"{name}_k"] = g["k"]
            rows.append(row)

        emit(folds, None)
        if len(folds) >= 3:
            for drop in folds:
                emit([f for f in folds if f != drop], drop)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import csv
    if rows:
        keys = list(rows[0])
        with open(out / "cells.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(rows)} cells -> {out/'cells.csv'}")

    # ---- the learning curve on Ren's trees, ONE TABLE PER OHIO LEVEL ------------------
    # NEVER POOLED ACROSS OHIO LEVELS. Nathan's Ohio frames move the per-tree count error by more
    # than several Clarks Run annotation levels do (5.79 burs a tree with no Ohio frames against
    # 1.89 with 25, both at one Clarks Run tree), so a curve that mixes them is not a curve in
    # Ren's trees at all and its apparent flatness is the Ohio axis showing through.
    #
    # The default used to be the single highest Ohio level. That is also the slowest-returning
    # stratum, because those are the largest cells, so the table printed NOTHING for the first
    # 551 returned cells rather than printing a curve at a lower level: a filter that is correct
    # in principle and silent in practice.
    lv_present = sorted({r["n_nathan"] for r in rows if r["arm"] == "grid"}, reverse=True)
    if a.nathan_level:
        want_lv = [int(x) for x in a.nathan_level]
        missing = [x for x in want_lv if x not in lv_present]
        if missing:
            print(f"\nno cells at Ohio level(s) {missing}; present: {lv_present}")
    else:
        want_lv = [lv for lv in lv_present
                   if len({r["n_ren"] for r in rows
                           if r["arm"] == "grid" and r["n_nathan"] == lv}) >= 5]
        print(f"\nOhio levels with at least 5 Clarks Run levels present: {want_lv} "
              f"(of {lv_present}); pass --nathan-level to choose")
    for scheme in ("flight", "tree"):
      for lv in want_lv:
        cur = [r for r in rows if r["arm"] == "grid" and r["scheme"] == scheme
               and r["n_nathan"] == lv and r["rule"] == "random"]
        if not cur:
            continue
        nb = max([r["nathan_train_boxes"] for r in cur] or [0])
        print(f"\n=== learning curve on Ren's trees, {scheme} folds, "
              f"{lv} Ohio frames ({nb} Ohio boxes) ===")
        print(f"{'trees':>6} {'hours':>6} {'burs':>6} | {'residSD':>8} {'mae':>6} {'r':>6} "
              f"{'slope':>6} {'bias':>6} | {'noGate':>7} {'recall':>7} {'prec':>6} {'rho':>6} "
              f"{'top10cap':>9} | {'epochs':>7} {'min':>6}")
        print(f"{'':>6} {'':>6} {'':>6} | {'<-- with the size gate, at each run\'s own threshold -->':<40}"
              f" | {'residSD':>7}")
        by = defaultdict(list)
        for r in cur:
            by[r["n_ren"]].append(r)
        for n in sorted(by):
            g = [r for r in by[n] if r["omit_fold"] is None]    # the table shows the full pool
            def m(k):
                v = [x[k] for x in g
                     if isinstance(x[k], (int, float)) and np.isfinite(x[k])]
                return float(np.mean(v)) if v else float("nan")
            print(f"{n:6d} {m('ren_hours'):6.1f} {m('ren_train_boxes'):6.0f} | "
                  f"{m('gate_resid_sd'):8.2f} {m('gate_mae'):6.2f} {m('gate_pearson'):6.3f} "
                  f"{m('gate_slope'):6.2f} {m('gate_bias'):+6.2f} | {m('resid_sd'):7.2f} "
                  f"{m('recall'):7.3f} {m('precision'):6.3f} {m('spearman'):6.3f} "
                  f"{m('top10_capture'):9.3f} | "
                  f"{m('mean_epochs'):7.0f} {m('mean_train_minutes'):6.0f}   (n={len(g)})")

        # cal_rmse and mae lead, because they are the metrics with headroom left; capture is kept
        # so the saturation is on the record rather than asserted
        for metric, falling in (("gate_resid_sd", True), ("gate_mae", True),
                                ("own_resid_sd", True), ("resid_sd", True),
                                ("gate_r2", False), ("top10_capture", False),
                                ("spearman", False), ("recall", False)):
            fit = fit_curves(by, metric, decreasing=falling)
            if not fit:
                continue
            arrow = "falling to" if falling else "rising to"
            print(f"\n  --- saturating fits to {metric}, {arrow} its limit ---")
            for form, v in fit["forms"].items():
                print(f"    {form:14s} limit {v['A']:.4f} "
                      f"[{v['A_lo']:.4f}, {v['A_hi']:.4f}]  rmse {v['rmse']:.4f}  "
                      f"within 1% of it at n={v['n99']}")
            print(f"    LIMIT SPREAD ACROSS FORMS {fit['A_spread']:.4f} "
                  f"({fit['A_min']:.4f} to {fit['A_max']:.4f}); best fitting form "
                  f"{fit['best_form']}")
            b = fit["forms"][fit["best_form"]]
            print(f"    MARGINAL VALUE, interpolated inside the data at n={b['mid']}: "
                  f"{b['per10_mid']:+.4f} per 10 trees ({b['per10_hours']} hours)")
            print(f"    MARGINAL VALUE, at the edge of the data n={b['per10_at']}: "
                  f"{b['per10']:+.4f} per 10 trees")
            print(f"    trees to come within 1% of the limit, across forms: {fit['n99_range']} "
                  f"(an extrapolation past n={b['n_at']}; treat the spread as the answer)")
            if np.isfinite(fit["noise"]):
                print(f"    the cell-to-cell spread at one level is {fit['noise']:.4f}, so a "
                      f"marginal gain smaller than that cannot be read from a single cell")

    # The run ids this analysis read, so anything built from cells.csv (the report's tranche status above
    # all) counts the SAME snapshot rather than whatever has been filed since.
    (out / "loaded_runs.txt").write_text("".join(r + "\n" for r in sorted(loaded)))
    json.dump(dict(min_per_tree=MIN_PER_TREE, iou=IOU,
                   thresholds=[float(x) for x in THRESHOLDS],
                   fixed_threshold=a.fixed_threshold, n_cells=len(rows)),
              open(out / "ANALYSIS.json", "w"), indent=1)


# Three functional forms, all of them saturating, fitted to the same points. The SPREAD between
# their asymptotes is the honest uncertainty on the extrapolation, and a bootstrap within one form
# is not, which a test caught before any of this ran: on synthetic data generated from an exact
# power law with A = 0.900, the single power-law fit returned 0.8922 with a bootstrap interval of
# [0.8859, 0.8977], so the truth sat OUTSIDE its own interval. The interval measured replicate
# noise while the real error was in how weakly an asymptote is determined by data that stop at 103
# trees while the curve is still rising. Reporting one fit's interval would have understated the
# uncertainty by more than the effect being claimed.
FORMS = {
    "power": (lambda n, A, B, c: A - B * np.power(n, -c),
              lambda y: [y.max(), 1.0, 0.5], ([0, 0, 0.01], [2, 50, 3])),
    "power_shifted": (lambda n, A, B, c: A - B * np.power(n + 5.0, -c),
                      lambda y: [y.max(), 1.0, 0.5], ([0, 0, 0.01], [2, 200, 3])),
    "exponential": (lambda n, A, B, tau: A - B * np.exp(-n / tau),
                    lambda y: [y.max(), 0.5, 30.0], ([0, 0, 1.0], [2, 5, 5000])),
}


def fit_curves(by_level, metric, n_boot=400, fold_key="omit_fold", decreasing=False,
               weight=False):
    """Fit every saturating form and report the limit spread, not one form's interval.

    The jackknife resamples FOLDS, not replicates, because the fold is the independent block: two
    seeds on the same fold score the same 31 held-out trees, so resampling replicates would treat
    correlated observations as independent and shrink the interval for the wrong reason.

    `fold_key` DEFAULTS TO `omit_fold`, WHICH IS THE ONLY FOLD-SHAPED KEY A ROW CARRIES. It used to
    default to `fold`, a key `emit()` has never written, so `r.get(fold_key)` was None on every row:
    the jackknife found no replicates and returned nan for every interval, and the point fit's
    `... is None or by_level[n]` fallback silently swept in all six rows of every cell (the full
    pool plus its five leave-one-fold-out replicates), fitting six correlated copies of each point
    as though they were independent. It was invisible because `len(levels) < 5` had short-circuited
    the function on every run so far, so no fit had yet been printed from real data.

    `decreasing` handles a metric that falls toward a floor, which per-tree count error does. The
    three forms all rise toward their asymptote and their A bound is [0, 2], so such a metric is
    mapped onto [0, 1] before fitting and every fitted quantity is mapped back, rather than fitting
    the negated metric against bounds that exclude its answer.
    """
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:
        # SAY SO. Returning None here removes every curve fit, every limit and every marginal
        # value from the run while the table above still prints normally, so a silent skip is
        # indistinguishable from an analysis that had nothing to fit. A system Python often
        # has no scipy; `requirements-sweep.txt` pins 1.18.0.
        print(f"  NO CURVE FITS: scipy is not importable here ({exc}). Every limit, marginal "
              f"value and interval below is ABSENT, not zero.", file=sys.stderr)
        return None
    levels = sorted(n for n in by_level if n > 0)
    if len(levels) < 5:
        print(f"  no fit to {metric}: {len(levels)} annotation levels present, 5 needed",
              file=sys.stderr)
        return None

    allv = np.array([r[metric] for n in levels for r in by_level[n]
                     if isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])], float)
    if allv.size < 6:
        return None
    if decreasing:
        top = float(allv.max())
        span = float(allv.max() - allv.min()) or 1.0

        def fwd(y):
            return (top - y) / span

        def inv(z):
            return top - z * span
    else:
        span = 1.0

        def fwd(y):
            return y

        def inv(z):
            return z

    # THE FIT MUST BE WEIGHTED, AND THE REASON IS MEASURED RATHER THAN ASSUMED (2026-10-04).
    # The cell-to-cell spread at one grid position is not constant along the curve: it is
    # proportional to how far the detector is from perfect. Measured over the 57 grid positions
    # with two or more cells, the correlation between a position's mean and its own spread is
    # +0.66 for resid_sd and +0.79 for mae, where perfect is zero, and -0.84 for top-decile
    # capture, -0.84 for the correlation, -0.80 for Spearman and -0.76 for recall, where perfect
    # is one. The same fact in both signs. For resid_sd the median spread runs 0.25 burs where the
    # metric is under 2 and 1.99 burs where it is 3 to 5, so an unweighted fit lets the noisiest
    # points, which are also the ones that set the curve's shape, dominate the residual sum of
    # squares. The 2026-10-03 note already said the analysis should weight by the measured
    # per-band spread; this is that, with one parameter instead of a table.
    #
    # ONE pooled coefficient of variation is estimated from the replicates, 0.326 for resid_sd on
    # 91 degrees of freedom, and the per-point sigma is that times the level's distance from
    # perfect. A per-level sigma from two or three replicates would be a worse estimate than a
    # pooled one applied proportionally.
    perf = 0.0 if decreasing else 1.0
    pool = [r for n in levels for r in by_level[n] if r.get(fold_key) is None]
    cvs, cvdf = 0.0, 0
    dists = []
    for n in levels:
        v = np.array([r[metric] for r in by_level[n] if r.get(fold_key) is None
                      and isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])],
                     float)
        if v.size:
            dists.append(abs(perf - v.mean()))
        if v.size >= 2 and abs(perf - v.mean()) > 0:
            cvs += (v.size - 1) * (v.std(ddof=1) / abs(perf - v.mean())) ** 2
            cvdf += v.size - 1
    cv = math.sqrt(cvs / cvdf) if cvdf >= 3 else None
    if not weight:
        # DEFAULT OFF, AND THIS IS A MEASURED RESULT AGAINST MY OWN ARGUMENT (2026-10-04).
        # The heteroscedasticity above is real, so weighting by it ought to help, and it does not.
        # Measured over 120 simulated campaigns generated from a power law with the measured
        # multiplicative noise, 13 levels x 3 seeds, weighting made the MARGINAL VALUE estimate
        # slightly WORSE (sd 0.0374 against 0.0326 burs per ten trees, about two standard errors
        # of the difference) while making the floor marginally better (sd 1.855 against 1.903).
        # The marginal is the quantity the stopping rule reads, so the thing that matters came out
        # the wrong way and the argument for weighting does not survive its own test. The
        # capability stays, with `weight=True`, because the negative result is specific to a
        # Gaussian multiplicative model and the real noise is partly a mixture: about 11 percent
        # of cells produce a detector that OVER-COUNTS per tree, and excluding those cuts the
        # pooled coefficient of variation from 0.300 to 0.204, so a treatment aimed at that
        # subgroup is more promising than a smooth weight.
        cv = None

    # a soft floor, so a level sitting exactly at perfect does not acquire infinite weight
    dfloor = 0.25 * float(np.median(dists)) if dists else 0.0
    # the level mean is used rather than each point's own value, so a point is not down-weighted
    # for being a high draw
    mu = {}
    for n in levels:
        v = np.array([r[metric] for r in by_level[n] if r.get(fold_key) is None
                      and isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])],
                     float)
        mu[n] = float(v.mean()) if v.size else float("nan")

    # THE LIMIT MUST BE PHYSICALLY POSSIBLE, AND IT WAS NOT (found and fixed 2026-10-04).
    # `FORMS` bounds the asymptote parameter to [0, 2] on whatever scale the fit runs on. For a
    # DECREASING metric the fit runs on a transformed scale where A = top - A_t * span, so an A_t of
    # 2 maps to a NEGATIVE floor whenever span exceeds half of top. Measured on the real data: the
    # power form returned a floor of -1.59, -4.13 and -2.69 burs a tree in three of four Ohio
    # columns. A per-tree count scatter cannot be negative, and `n99` computed from a negative
    # limit is meaningless, so the form was free to fit the data by placing its asymptote somewhere
    # impossible and then reporting a marginal from that geometry.
    #
    # The bound is therefore expressed in the METRIC's own units and mapped back: a decreasing
    # metric's floor lies in [0, top], and an increasing metric's ceiling lies in [0, perfect],
    # which is 1 for every rising metric here since they are all correlations or captured
    # fractions. Caught by a judge of four independent model proposals, one of which noticed the
    # negative asymptote that this analysis had been printing.
    if decreasing:
        a_lo_t, a_hi_t = max(0.0, (top - top) / span), (top - 0.0) / span
    else:
        a_lo_t, a_hi_t = 0.0, 1.0

    def fit(form, sample):
        f, p0, bounds = FORMS[form]
        lo, hi = list(bounds[0]), list(bounds[1])
        lo[0] = max(lo[0], a_lo_t)
        hi[0] = min(hi[0], a_hi_t)
        if not hi[0] > lo[0]:                      # degenerate range, leave the form's own bound
            lo[0], hi[0] = bounds[0][0], bounds[1][0]
        bounds = (lo, hi)
        x = np.array([n for n in levels for _ in sample[n]], float)
        y = np.array([r[metric] for n in levels for r in sample[n]], float)
        ok = np.isfinite(y)
        if ok.sum() < 6:
            raise ValueError("too few finite points")
        yt = fwd(y[ok])
        sig = None
        if cv:
            d = np.array([max(abs(perf - mu[n]), dfloor) if np.isfinite(mu[n]) else dfloor
                          for n in levels for _ in sample[n]], float)[ok]
            sig = cv * d / abs(span)        # sigma on the TRANSFORMED scale the fit works on
            if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
                sig = None
        init = list(p0(yt))
        init[0] = min(max(init[0], bounds[0][0]), bounds[1][0])   # p0 must lie inside the bounds
        p, _ = curve_fit(f, x[ok], yt, p0=init, bounds=bounds, maxfev=40000,
                         sigma=sig, absolute_sigma=False)
        resid = (yt - f(x[ok], *p)) * span      # back into the metric's own units
        rmse = float(np.sqrt((resid ** 2).mean()))
        # the reduced chi-square is what compares forms when the variances differ; the plain rmse
        # would prefer whichever form happens to fit the noisiest points
        if sig is not None:
            w = ((yt - f(x[ok], *p)) / sig) ** 2
            chi2 = float(w.sum() / max(1, ok.sum() - len(p)))
        else:
            chi2 = rmse
        return p, rmse, chi2

    # A JACKKNIFE OVER FOLDS, not a bootstrap over replicates. A cell is already pooled across the
    # five folds, so its replicates are the three seeds and there is nothing fold-shaped left to
    # resample. What the analysis emits instead is, for each cell, the same metrics recomputed with
    # one fold's trees left out of the pool. Five such curves give a leave-one-out error bar whose
    # unit is the fold, which is the independent block, and the jackknife needs no resampling.
    lofo = sorted({r[fold_key] for n in levels for r in by_level[n]
                   if r.get(fold_key) is not None})
    out = {}
    for form in FORMS:
        f = FORMS[form][0]
        try:
            p, rmse, chi2 = fit(form, {n: [r for r in by_level[n] if r.get(fold_key) is None]
                                       or by_level[n] for n in levels})
        except Exception:
            continue
        A = inv(float(p[0]))
        n_at = max(levels)
        per10 = float(inv(f(n_at + 10, *p)) - inv(f(n_at, *p)))
        # the marginal value INSIDE the observed range, which is interpolation rather than
        # extrapolation and is therefore the number to lead with
        mid = levels[len(levels) // 2]
        per10_mid = float(inv(f(mid + 10, *p)) - inv(f(mid, *p)))
        # "within one percent of the limit", evaluated in the metric's own units so it means the
        # same thing for a metric that rises and one that falls
        tol = 0.01 * abs(A)
        n99 = next((n for n in range(1, 1000000)
                    if abs(inv(f(n, *p)) - A) <= tol), None)
        jk = []
        for fo in lofo:
            try:
                jk.append(inv(fit(form, {n: [r for r in by_level[n] if r.get(fold_key) == fo]
                                         for n in levels})[0][0]))
            except Exception:
                pass
        if len(jk) >= 3:
            m = float(np.mean(jk))
            se = math.sqrt((len(jk) - 1) / len(jk) * float(np.sum((np.array(jk) - m) ** 2)))
            lo, hi = A - 1.96 * se, A + 1.96 * se
        else:
            se = lo = hi = float("nan")
        out[form] = dict(A=A, params=[float(v) for v in p], rmse=rmse, chi2_red=chi2,
                         weighted=bool(cv), cv=float(cv) if cv else None, n_at=int(n_at),
                         per10=per10, per10_at=int(n_at), per10_mid=per10_mid, mid=int(mid),
                         per10_hours=round(10 * MIN_PER_TREE / 60, 2), n99=n99,
                         A_se=float(se), A_lo=float(lo), A_hi=float(hi), n_jackknife=len(jk))
    if not out:
        return None
    # The resolution floor: the pooled within-level standard deviation of the metric across the
    # cells at one annotation level, which mixes the seed draw and the unseeded rotation stream.
    # A fitted marginal gain smaller than this is below what a single cell can show, so it is
    # printed beside the gain rather than left for the reader to work out.
    within = []
    for n in levels:
        v = np.array([r[metric] for r in by_level[n] if r.get(fold_key) is None
                      and isinstance(r.get(metric), (int, float)) and np.isfinite(r[metric])],
                     float)
        if v.size >= 2:
            within.append(((v - v.mean()) ** 2).sum() / (v.size - 1))
    noise = float(math.sqrt(float(np.mean(within)))) if within else float("nan")
    Avals = [v["A"] for v in out.values()]
    lo = [v["A_lo"] for v in out.values() if np.isfinite(v["A_lo"])]
    hi = [v["A_hi"] for v in out.values() if np.isfinite(v["A_hi"])]
    return dict(forms=out, n_forms=len(out), noise=noise, decreasing=bool(decreasing),
                A_min=float(min(Avals)), A_max=float(max(Avals)),
                A_spread=float(max(Avals) - min(Avals)),
                A_envelope_lo=float(min(lo)) if lo else float("nan"),
                A_envelope_hi=float(max(hi)) if hi else float("nan"),
                # the reduced chi-square, not the plain rmse: with unequal variances the rmse
                # would prefer whichever form happens to fit the noisiest points best
                best_form=min(out, key=lambda k: out[k]["chi2_red"]),
                n99_range=[min((v["n99"] for v in out.values() if v["n99"]), default=None),
                           max((v["n99"] for v in out.values() if v["n99"]), default=None)])


if __name__ == "__main__":
    main()
