#!/usr/bin/env python3
"""Ren's annotation of each Clarks Run tree against the crew's 2026 field count of the same tree.

The campaign scores detectors against the annotation and never against field counts, because a field
count includes burs the camera cannot see. This module measures that part directly: how the number of
burs annotated in a tree's hover frame relates to the number the crew counted on the tree.

Two rules, both read off the data and both reported beside the result:

  1. Only annotated burs within RADIUS_M of the frame's center count toward the tree. A hover frame
     shows the target tree at its center and the edges of neighboring crowns, and Ren annotates burs
     on the neighbors too. On 2026-10-08, 83 of 1,539 annotated crown burs lay more than 3 m from the
     center and none between 2.5 and 3 m, so any radius from 2.0 to 3.0 m gives the same counts.
  2. Trees whose two counts differ at least threefold (the smaller floored at 1) and by at least five
     burs are left out as probable recording errors; the correlation with them is reported too.

The share missing from the annotation is 1 - annotated / field, pooled over the trees of a load class,
so it is negative where the annotation holds more burs than the crew counted.

    python3 tools/annotation_vs_field.py [--radius 2.5]
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import optimize, stats

W, H = 8192, 5460                 # every Clarks Run hover frame, Zenmuse P1 at 35 mm
M_PER_PX = 0.1225e-3 * 7.87       # P1 self-calibrated angular pixel at the 7.87 m camera-to-crown standoff;
                                  # the nominal 1.2521e-4 gives identical results here (no bur lies 2.5 to 3 m out)
RADIUS_M = 2.5
CLASSES_FROM = "work/2026-10-02_arc_sweep/design/train640_MANIFEST.json"   # the truth classes training used
LOAD_CLASSES = ((1, 3), (4, 10), (11, 22), (23, 44), (45, 10 ** 6))


def annotated(proj):
    """Per tree: the distance in meters from the frame's center of every annotated crown bur."""
    keep = set(json.load(open(Path(proj) / CLASSES_FROM))["classes"])
    out = {}
    for p in sorted(glob.glob(str(Path(proj) / "inputs/ren-annotations/ClarksRun_set_*/labels/detect/*.txt"))):
        tree = os.path.basename(p).split("__")[0]
        if tree in out:
            raise SystemExit("tree %s has two annotated frames: %s" % (tree, p))
        d = []
        for line in open(p):
            x = line.split()
            if x and int(x[0]) in keep:
                d.append(math.hypot((float(x[1]) - 0.5) * W, (float(x[2]) - 0.5) * H) * M_PER_PX)
        out[tree] = d
    return out


def field(proj):
    """Per tree: the crew's count (cluster_size summed over cluster events; ground events left out) and
    the number of visits, a visit being a run of consecutive rows for the tree in one observer's sheet."""
    rows = collections.defaultdict(list)
    for p in glob.glob(str(Path(proj) / "inputs/clarks-run-bur-count/20*/*.csv")):
        for r in csv.DictReader(open(p)):
            rows[r["observer"]].append(r)
    count, visits = collections.Counter(), collections.Counter()
    for rs in rows.values():
        rs.sort(key=lambda r: r["recorded_at_utc"])
        prev = None
        for r in rs:
            u = r["observationunit_name"]
            if u != prev:
                visits[u] += 1
                prev = u
            if r["event"] == "cluster":
                count[u] += int(r["cluster_size"])
    return count, visits


def power_fit(x, y):
    """y = c * x**b as a Poisson regression of the count y on log x, which weights each tree by the
    variance a count has rather than letting the heaviest trees set the curve."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nll = lambda p: -(stats.poisson.logpmf(y, np.exp(p[0] + p[1] * np.log(x)))).sum()
    p = optimize.minimize(nll, [0.0, 1.0], method="Nelder-Mead",
                          options=dict(xatol=1e-9, fatol=1e-9, maxiter=5000)).x
    return math.exp(p[0]), float(p[1])


INVERSION_BANDS = ((1, 3), (4, 7), (8, 12), (13, 20), (21, 30), (31, 45))


def inversion(counts, field_counts, bands=INVERSION_BANDS):
    """For each band of an image count (annotated or model), the field counts of those trees: the
    median and the middle 90 percent. The scatter grows with load, so a band table is the honest
    error bar where one standard deviation about a line is not."""
    x, f = np.asarray(counts, float), np.asarray(field_counts, float)
    out = []
    for lo, hi in bands:
        k = (x >= lo) & (x <= hi)
        if k.sum():
            out.append(dict(lo=lo, hi=hi, trees=int(k.sum()), median=float(np.median(f[k])),
                            p5=float(np.percentile(f[k], 5)), p95=float(np.percentile(f[k], 95)),
                            fmin=int(f[k].min()), fmax=int(f[k].max())))
    return out


def outlier(a, n):
    lo, hi = sorted((max(a, 1), max(n, 1)))
    return hi >= 3 * lo and abs(a - n) >= 5


def compare(proj, radius=RADIUS_M):
    ann = annotated(proj)
    count, visits = field(proj)
    trees = sorted(ann)
    bad = [(t, visits[t]) for t in trees if visits[t] != 1]
    if bad:
        raise SystemExit("every annotated tree needs exactly one field visit; these do not: %s" % bad[:10])
    allb = [d for t in trees for d in ann[t]]
    a_all = {t: len(ann[t]) for t in trees}
    a = {t: sum(d <= radius for d in ann[t]) for t in trees}
    n = {t: count[t] for t in trees}
    out = [t for t in trees if outlier(a[t], n[t])]
    kept = [t for t in trees if t not in out]

    def fit(ts, aa):
        x = np.array([n[t] for t in ts], float)
        y = np.array([aa[t] for t in ts], float)
        slope, icept = np.polyfit(x, y, 1)
        resid = y - (slope * x + icept)
        rng = np.random.default_rng(20261008)          # fixed, so the report's interval is reproducible
        boot = []
        for _ in range(2000):
            k = rng.integers(0, len(x), len(x))
            boot.append(np.polyfit(x[k], y[k], 1)[0])
        return dict(pearson=float(stats.pearsonr(x, y)[0]), spearman=float(stats.spearmanr(x, y)[0]),
                    slope=float(slope), intercept=float(icept),
                    slope_lo=float(np.percentile(boot, 5)), slope_hi=float(np.percentile(boot, 95)),
                    resid_sd=float(math.sqrt((resid ** 2).sum() / (len(x) - 2))),
                    missing=float(1 - y.sum() / x.sum()), n=len(ts), field_max=int(x.max()))

    kf = fit(kept, a)
    xk = np.array([n[t] for t in kept], float)
    yk = np.array([a[t] for t in kept], float)
    pc, pb = power_fit(xk, yk)
    rng = np.random.default_rng(20261008)
    bb = []
    for _ in range(1000):
        k = rng.integers(0, len(xk), len(xk))
        bb.append(power_fit(xk[k], yk[k])[1])
    pres = yk - pc * xk ** pb
    power = dict(c=pc, b=pb, b_lo=float(np.percentile(bb, 5)), b_hi=float(np.percentile(bb, 95)),
                 scatter=float(math.sqrt((pres ** 2).sum() / (len(xk) - 2))))
    rows = []
    for lo, hi in LOAD_CLASSES:
        ts = [t for t in kept if lo <= n[t] <= hi]
        fa, aa = sum(n[t] for t in ts), sum(a[t] for t in ts)
        on_line = sum(kf["slope"] * n[t] + kf["intercept"] for t in ts)
        on_curve = sum(pc * n[t] ** pb for t in ts)
        rows.append(dict(lo=lo, hi=hi, trees=len(ts), field=fa, annotated=aa,
                         missing=1 - aa / fa if fa else float("nan"),
                         line_missing=1 - on_line / fa if fa else float("nan"),
                         power_missing=1 - on_curve / fa if fa else float("nan")))
    heavy = [t for t in kept if n[t] >= 23]
    return dict(
        radius=radius, n_trees=len(trees), n_burs=len(allb),
        beyond3=sum(d > 3.0 for d in allb), band=sum(radius < d <= 3.0 for d in allb),
        outliers=[(t, a[t], n[t]) for t in out],
        # trees the whole-frame count would call recording errors while the target-tree count does not
        frame_only=[(t, a_all[t], a[t], n[t]) for t in trees if outlier(a_all[t], n[t]) and not outlier(a[t], n[t])],
        kept=kf, all_trees=fit(trees, a), whole_frame=fit(kept, a_all),
        classes=rows, power=power,
        inversion=inversion([a[t] for t in kept if a[t] > 0], [n[t] for t in kept if a[t] > 0]),
        heavy=dict(trees=len(heavy), missing=1 - sum(a[t] for t in heavy) / sum(n[t] for t in heavy)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--radius", type=float, default=RADIUS_M)
    ap.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    r = compare(a.project, a.radius)
    print("%d trees, %d annotated crown burs: %d more than 3 m from the center, %d between %.1f and 3 m"
          % (r["n_trees"], r["n_burs"], r["beyond3"], r["band"], r["radius"]))
    print("left out as probable recording errors: %s" % r["outliers"])
    for k in ("kept", "all_trees", "whole_frame"):
        f = r[k]
        print("%-12s n %3d  r %.3f  rho %.3f  annotated = %.2f x field + %.2f  scatter %.2f burs  missing %+.3f"
              % (k, f["n"], f["pearson"], f["spearman"], f["slope"], f["intercept"], f["resid_sd"], f["missing"]))
    for c in r["classes"]:
        print("  field %s: %3d trees, %4d field burs, %4d annotated, missing %+.3f"
              % ("%d to %d" % (c["lo"], c["hi"]) if c["hi"] < 10 ** 6 else "%d and above" % c["lo"],
                 c["trees"], c["field"], c["annotated"], c["missing"]))
    print("23 and above: %d trees, missing %+.3f" % (r["heavy"]["trees"], r["heavy"]["missing"]))


if __name__ == "__main__":
    main()
