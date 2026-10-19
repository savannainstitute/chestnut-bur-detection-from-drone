"""Draw the campaign's figures from analyze_sweep.py's cells.csv.

THE FIGURE RULES HERE ARE NOT PREFERENCES AND SHOULD NOT BE RELAXED. No footnotes, no source lines,
no methodology notes, no text set smaller than the body. A fact that matters earns a readable place
in the figure; a fact that does not is cut. Small print is not rigor, it is a way of avoiding the
decision about what belongs in the figure, by putting the undecided material somewhere nobody reads.
So every annotation here is at body size or larger, and anything that could not be made legible was
left out rather than shrunk.

Run with an interpreter that has matplotlib: ~/.venvs/chestnut-yield/bin/python3.
"""
import argparse
import csv
import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_PER_TREE = 3.26          # measured from Ren's own annotation_stats.json
LOW_R = 0.85
BODY = 13                    # nothing in any figure is smaller than this
ALL_REN = [0, 1, 3, 6, 10, 15, 22, 32, 45, 63, 78, 92, 103]   # the design's own levels
N_FOLDS = 5                  # the design's folds; a value pooling fewer is on a different test set
SCHEME = "flight"            # folds block whole drone missions. The `tree` scheme splits trees at
                             # random, which is the OPTIMISTIC side of the leakage contrast that
                             # tranche 2 exists to measure, so the two must never share a figure.

# ONLY DRAW A VALUE POOLED OVER ALL FIVE FOLDS. This is a validity rule, not a presentation choice.
# The metric pools out-of-fold predictions for all 154 annotated trees, each scored by a model that
# never saw it. A cell with only folds 0 and 1 is scored on 61 of those trees, and not a random 61:
# fold 3 alone holds 21 of the campaign's 46 failed folds, and fold 1 is one whole drone mission. So
# a two-fold value differs from a five-fold neighbor in WHICH TREES WERE SCORED as well as in the
# training data, and the difference cannot be attributed to the axis the figure is about.
#
# MEASURED 2026-10-05 and this is why the rule exists: 14 of 102 grid positions had no five-fold
# cell, 13 of them in the Ohio 25 and 36 columns, because Ohio 36 ran on VT ARC and VT finished fold
# 0 and part of fold 1 before the campaign moved. That is the facility-and-fold collinearity
# recorded on 2026-10-03 reappearing inside a figure, where it reads as an effect of Ohio frames.

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": BODY,
    "axes.titlesize": BODY + 4,
    "axes.labelsize": BODY + 1,
    "xtick.labelsize": BODY,
    "ytick.labelsize": BODY,
    "legend.fontsize": BODY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
})


def save(fig, out):
    """Save a figure, refusing one with anything drawn outside the canvas.

    A title one line too long for the figure is cut off at the edge of the image with no warning, and
    tight_layout cannot shrink text to fit. Figure 6's title was cut off that way on 2026-10-08, so every
    figure now goes through this check rather than relying on someone reading each image at full size.
    """
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())      # inches, over every artist actually drawn
    w, h = fig.get_size_inches()
    tol = 0.01
    if tb.x0 < -tol or tb.y0 < -tol or tb.x1 > w + tol or tb.y1 > h + tol:
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        plt.close(fig)
        raise RuntimeError("%s: drawn content spans %.2f to %.2f by %.2f to %.2f inches on a %.2f by %.2f "
                           "canvas, so something is cut off; titles %s" % (out, tb.x0, tb.x1, tb.y0, tb.y1, w, h, titles))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fnum(r, k):
    try:
        v = float(r[k])
        return v if np.isfinite(v) else float("nan")
    except (ValueError, KeyError, TypeError):
        return float("nan")


def load(path):
    rows = list(csv.DictReader(open(path)))
    pool = [r for r in rows if r["omit_fold"] == "" and r["arm"] == "grid"]
    jack = [r for r in rows if r["omit_fold"] != "" and r["arm"] == "grid"]
    return rows, pool, jack


def assert_comparable(points, what):
    """Every value in one figure must differ ONLY along the axes the figure is about.

    TWO SEPARATE MIXTURES GOT INTO THE FIRST DRAFTS, and the second would have survived a check
    built only for the first, which is why this tests several keys rather than one.

    1. DIFFERENT TREE SETS. The Ohio 36 column pooled two folds and was scored on 61 of the 154
       annotated trees while its neighbors used all 154, and the missing 93 were not random: fold 3
       alone holds 21 of the campaign's 46 failed folds.
    2. DIFFERENT FOLD SCHEMES, which `n_trees_scored` cannot see because both schemes score all 154
       trees and merely partition them differently. Three squares at the top of the Ohio 36 column
       were tranche 2, whose folds split trees at random, sitting in a grid otherwise built from
       flight-blocked folds. Flight against tree is the single contrast tranche 2 exists to measure
       and the random split is its optimistic side, so those squares read as "the full Ohio set
       gives the lowest error" when they say nothing of the kind.

    Both facts were columns in `cells.csv` the whole time. This is an assertion rather than a habit
    because reading them was already possible and did not happen twice.
    """
    for key, why in (("n_trees_scored", "scored on DIFFERENT tree sets"),
                     ("scheme", "built from DIFFERENT fold schemes"),
                     ("rule", "built with DIFFERENT tree-selection rules"),
                     ("nathan_rule", "built with DIFFERENT Ohio-frame rules"),
                     ("tile", "built at DIFFERENT tile sizes")):
        vals = {r[key] for r in points if key in r}
        if len(vals) > 1:
            raise SystemExit(
                "%s would compare values %s: %s. A figure may only vary along the axes it is "
                "about, so filter on %s before plotting."
                % (what, why, sorted(vals), key))
    sizes = {int(r["n_trees_scored"]) for r in points}
    return sizes.pop() if sizes else 0


def jackknife_se(jack, key, sel):
    """Leave-one-fold-out standard error over the fold, which is the independent block.

    Two seeds at one fold score the same held-out trees, so a bootstrap over replicates would treat
    correlated observations as independent. The replicates here are the per-fold deletions that
    emit() already wrote.
    """
    vals = defaultdict(list)
    for r in jack:
        if sel(r):
            v = fnum(r, key)
            if np.isfinite(v):
                vals[r["omit_fold"]].append(v)
    means = [np.mean(v) for v in vals.values() if v]
    if len(means) < 3:
        return float("nan")
    m = float(np.mean(means))
    k = len(means)
    return math.sqrt((k - 1) / k * float(np.sum((np.array(means) - m) ** 2)))


# ----------------------------------------------------------------------------------------------
def dense_cells(rows):
    """The no-Ohio curve at full resolution: tranche 6 plus tranche 1's no-Ohio cells.

    Tranche 6 (`grid_dense`) runs 44 Clarks Run levels with six seeds and deliberately omits seeds
    1 to 3 at the nine levels it shares with the main grid, because tranche 1's no-Ohio cells are
    those runs already. So the curve is the union of the two arms, restricted by the same validity
    rules as every other figure: one fold scheme and only values pooled over all five folds.
    """
    d = [r for r in rows if r["omit_fold"] == "" and r["arm"] in ("grid", "grid_dense")
         and int(r["n_nathan"]) == 0 and r["scheme"] == SCHEME and int(r["folds"]) >= N_FOLDS]
    assert_comparable(d, "the dense no-Ohio figures")
    return d


BANDS = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 15), (16, 22), (23, 31), (32, 44), (45, 63), (64, 103)]


def fig_failure_rate(dense, out):
    """How often a run fails, by annotated trees, on the no-Ohio data where failures are commonest.

    REDRAWN 2026-10-06 FROM THE DENSE ARM, and the first version's headline did not survive it. That
    version pooled the main grid over every Ohio level and both fold schemes and reported no failure
    in 322 folds from 32 trees up. Ohio frames suppress the failure, so pooling over them mixed a
    rate that depends on the Ohio axis into a figure about the Clarks Run axis, and the no-Ohio data,
    with six runs at 47 levels, shows failures continuing at about 1 percent rather than stopping.
    """
    lvl = defaultdict(lambda: [0, 0])
    for r in dense:
        n = int(r["n_ren"])
        lvl[n][0] += int(fnum(r, "gate_folds_lowr")) if np.isfinite(fnum(r, "gate_folds_lowr")) else 0
        lvl[n][1] += int(r["folds"])
    fails = [sum(lvl[n][0] for n in lvl if a <= n <= b) for a, b in BANDS]
    folds = [sum(lvl[n][1] for n in lvl if a <= n <= b) for a, b in BANDS]
    rate = [100.0 * f / max(1, d) for f, d in zip(fails, folds)]
    lab = [("%d" % a) if a == b else ("%d to %d" % (a, b)) for a, b in BANDS]
    hrs = ["%.1f h" % (b * MIN_PER_TREE / 60) for a, b in BANDS]
    tail = next(i for i, (a, b) in enumerate(BANDS) if a >= 23)
    tf, td = sum(fails[tail:]), sum(folds[tail:])

    fig, ax = plt.subplots(figsize=(12, 6.8))
    x = np.arange(len(BANDS))
    bars = ax.bar(x, rate, color="#b2182b", width=0.68)
    for b, f, d in zip(bars, fails, folds):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, "%d of %d" % (f, d),
                ha="center", va="bottom", fontsize=BODY)
    ax.axvspan(tail - 0.5, len(BANDS) - 0.5, color="#4393c3", alpha=0.13)
    ax.text((tail + len(BANDS) - 1) / 2, max(rate) * 0.62,
            "%d failed in %d folds\nfrom %d trees on" % (tf, td, BANDS[tail][0]),
            ha="center", va="center", fontsize=BODY + 2, color="#2166ac", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (l, h) for l, h in zip(lab, hrs)])
    ax.set_xlabel("Clarks Run trees annotated, with no Ohio frames, and the hours of annotation\n"
                  "Up to six runs at each level; a fold fails when its per-tree correlation is under 0.85")
    ax.set_ylabel("percent of folds whose detector\nranks trees badly")
    ax.set_title("%d percent of runs fail with one annotated tree, and about %d percent from %d trees on"
                 % (round(rate[0]), max(1, round(100.0 * tf / td)), BANDS[tail][0]))
    ax.set_ylim(0, max(rate) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, out)
    return dict(bands=BANDS, folds=folds, fails=fails, tail=(tf, td))


def _medians(dense):
    by = defaultdict(list)
    for r in dense:
        by[int(r["n_ren"])].append(fnum(r, "gate_resid_sd"))
    lv = sorted(by)
    return lv, by, np.array([float(np.median(by[n])) for n in lv])


def _hours_axis(ax):
    top = ax.secondary_xaxis("top", functions=(lambda n: n * MIN_PER_TREE / 60,
                                               lambda h: h * 60 / MIN_PER_TREE))
    top.set_xlabel("hours of Ren's annotation")
    top.tick_params(labelsize=BODY)


def fig_dense_curve(dense, out):
    lv, by, med = _medians(dense)
    fig, ax = plt.subplots(figsize=(12, 7))
    xs = [n for n in lv for _ in by[n]]
    ys = [v for n in lv for v in by[n]]
    ax.scatter(xs, ys, s=26, color="#7f7f7f", alpha=0.5, label="one run")
    ax.plot(lv, med, color="#b2182b", lw=2.6, label="median of the runs at each level")
    _hours_axis(ax)
    ax.set_xlim(0, 106)
    ax.set_ylim(0, max(ys) * 1.06)
    ax.set_xlabel("Clarks Run trees annotated, with no Ohio frames\n"
                  "Each dot is one run pooled over all five folds")
    ax.set_ylabel("bur count error per tree, burs\n(the part no calibration removes)")
    i20 = lv.index(20)
    ax.set_title("Count error falls fast over the first 20 annotated trees and slowly after that")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out)
    return dict(levels=len(lv), runs=len(ys), med1=float(med[0]), med20=float(med[i20]),
                med_last=float(med[-1]))


def _power(n, A, B, c):
    return A + B * np.power(n, -c)


def _fit_power(x, y):
    from scipy.optimize import curve_fit
    hi = float(np.min(y))
    p, _ = curve_fit(_power, x, y, p0=(0.9 * hi, 4.0, 0.7),
                     bounds=([0.0, 1e-6, 1e-3], [hi, 100.0, 5.0]), maxfev=100000)
    return p


def fig_marginal(dense, out, n_boot=300, seed=20261006):
    """What the next ten trees buy, with an interval from resampling the runs at each level.

    ONE SHAPE, AND WHY, because the campaign's earlier rule was to read the largest of three. That
    rule exists for EXTRAPOLATION past the data, where the shapes disagree by orders of magnitude.
    Every gain drawn here starts at 93 trees or fewer and ends at 103 or fewer, so it is
    interpolation, which the 2026-10-02 simulation showed the best-fitting shape recovers to about
    0.0002 burs. Measured on these medians: the shifted power law collapses onto the plain one (its
    shift fits to zero), and the exponential fits more than twice as badly (0.38 burs a tree of
    error on the medians against 0.16), while reading LARGER at 10 trees because it falls too fast
    early and then claims the tail is flat. Its 0.63 at 10 trees became the first draft's headline,
    which is why it is not drawn. The interval resamples runs within each level, so it carries the
    run-to-run spread, including the occasional failed run, into the fitted gain.
    """
    lv, by, med = _medians(dense)
    x = np.array(lv, float)
    p = _fit_power(x, med)
    rmse = float(np.sqrt(np.mean((med - _power(x, *p)) ** 2)))
    grid = np.arange(5, 94)
    gain = _power(grid, *p) - _power(grid + 10, *p)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        mb = np.array([np.median(rng.choice(by[n], size=len(by[n]), replace=True)) for n in lv])
        try:
            pb = _fit_power(x, mb)
        except Exception:
            continue
        boots.append(_power(grid, *pb) - _power(grid + 10, *pb))
    B = np.vstack(boots)
    lo, hi = np.percentile(B, 5, axis=0), np.percentile(B, 95, axis=0)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(grid, lo, hi, color="#b2182b", alpha=0.18,
                    label="90 percent of refits after resampling the runs")
    ax.plot(grid, gain, lw=2.8, color="#b2182b", label="power law fitted to the medians")
    ax.set_yscale("log")
    ax.set_ylim(0.003, 2.0)
    ax.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    ax.set_yticklabels(["0.003", "0.01", "0.03", "0.1", "0.3", "1"])
    _hours_axis(ax)
    ax.set_xlim(0, 96)
    idx = {n: list(grid).index(n) for n in (10, 32, 93)}
    at = {n: float(gain[i]) for n, i in idx.items()}
    up = {n: float(hi[i]) for n, i in idx.items()}
    ax.set_xlabel("Clarks Run trees already annotated, with no Ohio frames\n"
                  "Fitted to the median count error at each of %d levels, up to six runs a level" % len(lv))
    ax.set_ylabel("fall in count error from ten more\nannotated trees, burs a tree")
    ax.set_title("With ten more annotated trees, count error falls by %.2f burs a tree starting from\n"
                 "10 trees, by %.2f starting from 32 and by %.2f starting from 93" % (at[10], at[32], at[93]))
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.3, which="major")
    fig.tight_layout()
    save(fig, out)
    return dict(at=at, upper=up, params=[float(v) for v in p], rmse=rmse, n_boot=len(boots))


def fig_count_scatter(pool, jack, out):
    """The decision metric: per-tree count scatter that survives the best affine correction."""
    ohio = sorted({int(r["n_nathan"]) for r in pool})
    WANT = (0, 4, 11, 17, 25, 36)
    BAND_AT = 17        # a column complete at every level, so the band means what it says
    show = [o for o in ohio if o in WANT
            and len({int(r["n_ren"]) for r in pool if int(r["n_nathan"]) == o}) >= 7]
    pool = [r for r in pool if r["scheme"] == SCHEME]
    full_all = [r for r in pool if int(r["folds"]) >= N_FOLDS]
    n_trees = assert_comparable(full_all, "the count-error figure")
    print("    every drawn value is scored on %d trees" % n_trees)
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(11, 6.8))
    for i, o in enumerate(show):
        pts = []
        for n in sorted({int(r["n_ren"]) for r in pool if int(r["n_nathan"]) == o}):
            g = [r for r in pool if int(r["n_nathan"]) == o and int(r["n_ren"]) == n]
            v = [fnum(r, "gate_resid_sd") for r in g]
            v = [x for x in v if np.isfinite(x)]
            full = [r for r in g if int(r["folds"]) >= N_FOLDS]
            v = [fnum(r, "gate_resid_sd") for r in full]
            v = [x for x in v if np.isfinite(x)]
            if v:
                pts.append((n, float(np.mean(v)), len(v)))
        if len(pts) < 5:
            continue
        xs = [ALL_REN.index(p[0]) for p in pts]
        ys = [p[1] for p in pts]
        c = cmap(i / max(1, len(show) - 1))
        lab = ("%d Ohio frames, with its leave-one-fold-out band" % o
               if o == BAND_AT else "%d Ohio frames" % o)
        ax.plot(xs, ys, "-o", color=c, lw=2.2, ms=6, label=lab)
        if o == BAND_AT:
            se = [jackknife_se(jack, "gate_resid_sd",
                               lambda r, O=o, N=p[0]: int(r["n_nathan"]) == O
                               and int(r["n_ren"]) == N) for p in pts]
            lo = [y - (s if np.isfinite(s) else 0) for y, s in zip(ys, se)]
            hi = [y + (s if np.isfinite(s) else 0) for y, s in zip(ys, se)]
            ax.fill_between(xs, lo, hi, color=c, alpha=0.2)
    ax.set_xticks(range(len(ALL_REN)))
    ax.set_xticklabels(ALL_REN)
    ax.set_xlabel("Clarks Run trees annotated\n\n"
                  "Every point pools all five folds, and every fold blocks a whole drone mission")
    ax.set_ylabel("bur count error per tree, burs\n(the part no calibration removes)")
    ax.set_title("Ohio frames matter below about 32 annotated trees and little above")

    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out)


def fig_gate_method(pool, out):
    """Why the size gate is in: it turns an alternating mess into a curve."""
    g0 = [r for r in pool if int(r["n_nathan"]) == 0 and r["scheme"] == "flight"
          and r["rule"] == "random"]
    levels = sorted({int(r["n_ren"]) for r in g0 if int(r["n_ren"]) > 0})
    un, ga, ns = [], [], []
    for n in levels:
        g = [r for r in g0 if int(r["n_ren"]) == n]
        u = [fnum(r, "resid_sd") for r in g]
        q = [fnum(r, "gate_resid_sd") for r in g]
        un.append(np.nanmean(u))
        ga.append(np.nanmean(q))
        ns.append(len(g))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xi = range(len(levels))
    ax.plot(xi, un, "-o", color="#b2182b", lw=2.2, ms=7,
            label="every predicted box kept")
    ax.plot(xi, ga, "-s", color="#2166ac", lw=2.6, ms=7,
            label="boxes larger than any possible bur discarded")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_xlabel("Clarks Run trees annotated, with no Ohio frames")
    ax.set_ylabel("bur count error per tree, burs")
    ax.set_title("With boxes larger than any bur removed, count error falls steadily")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.3)
    ax.text(0.37, 0.97,
            "No hand-annotated bur reaches 110 pixels across\n"
            "41,212 of them. Predictions that match nothing\n"
            "reach 640, the width of a whole image tile.",
            transform=ax.transAxes, fontsize=BODY, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", fc="#eef3f8", ec="#b8cfe0"))
    fig.tight_layout()
    save(fig, out)


def fig_surface(pool, out, design_path):
    """Both design axes at once, distinguishing a square the design never had from one still running.

    Those are different facts and the first version printed "not run" for both. The design omits
    exactly one position, zero annotated trees with zero Ohio frames, because it has nothing to
    train on. Everything else that is blank is simply not finished.
    """
    import json as _json
    dz = _json.load(open(design_path))
    in_design = {(r["n_ren"], r["n_nathan"]) for r in dz["runs"] if r["arm"] == "grid"}
    pool = [r for r in pool if r["scheme"] == SCHEME]
    assert_comparable([r for r in pool if int(r["folds"]) >= N_FOLDS], "the surface figure")
    ren = sorted({int(r["n_ren"]) for r in pool})
    ohio = sorted({int(r["n_nathan"]) for r in pool})
    M = np.full((len(ren), len(ohio)), np.nan)
    N = np.zeros_like(M)
    for i, n in enumerate(ren):
        for j, o in enumerate(ohio):
            g = [r for r in pool if int(r["n_ren"]) == n and int(r["n_nathan"]) == o
                 and int(r["folds"]) >= N_FOLDS]
            v = [fnum(r, "gate_resid_sd") for r in g]
            v = [x for x in v if np.isfinite(x)]
            if v:
                M[i, j] = float(np.mean(v))
                N[i, j] = len(v)
            else:
                any_g = [r for r in pool if int(r["n_ren"]) == n
                         and int(r["n_nathan"]) == o]
                N[i, j] = -max((int(r["folds"]) for r in any_g), default=0)
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis_r",
                   vmin=np.nanmin(M), vmax=min(np.nanmax(M), 4.0))
    for i in range(len(ren)):
        for j in range(len(ohio)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, "%.1f" % M[i, j], ha="center", va="center",
                        fontsize=BODY, color="white" if M[i, j] > 2.2 else "black")
            else:
                if (ren[i], ohio[j]) not in in_design:
                    lab = "nothing\nto train"
                elif N[i, j] < 0:
                    lab = "%d of 5\nfolds" % (-N[i, j])
                else:
                    lab = "still\nrunning"
                ax.text(j, i, lab, ha="center", va="center", fontsize=BODY,
                        color="#777777", style="italic")
    ax.set_xticks(range(len(ohio)))
    ax.set_xticklabels(ohio)
    ax.set_yticks(range(len(ren)))
    ax.set_yticklabels(ren)
    ax.set_ylabel("Clarks Run trees annotated")
    ax.set_title("Bur count error per tree across both kinds of training data")
    ax.set_xlabel("Ohio frames from Nathan's site\n\n"
                  "A square is filled only where all five mission-blocked folds are finished")
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("burs of error per tree", fontsize=BODY + 1)
    fig.tight_layout()
    save(fig, out)
    return int(np.isfinite(M).sum()), M.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import os
    os.makedirs(a.out, exist_ok=True)
    rows, pool, jack = load(a.cells)
    print("cells.csv: %d rows, %d pooled grid cells, %d jackknife rows" % (len(rows), len(pool), len(jack)))

    dense = dense_cells(rows)
    r = fig_failure_rate(dense, os.path.join(a.out, "1_failure_rate.png"))
    print("  1_failure_rate.png        %d bands, %d folds, %d failed; %d of %d from 23 trees on"
          % (len(r["bands"]), sum(r["folds"]), sum(r["fails"]), r["tail"][0], r["tail"][1]))
    fig_count_scatter(pool, jack, os.path.join(a.out, "2_count_error.png"))
    print("  2_count_error.png")
    fig_gate_method(pool, os.path.join(a.out, "3_size_gate.png"))
    print("  3_size_gate.png")
    got, tot = fig_surface(pool, os.path.join(a.out, "4_surface.png"), a.design)
    print("  4_surface.png             %d of %d grid positions have a cell" % (got, tot))
    d = fig_dense_curve(dense, os.path.join(a.out, "5_dense_curve.png"))
    print("  5_dense_curve.png         %d levels, %d runs; median %.2f at 1 tree, %.2f at 20, %.2f at 103"
          % (d["levels"], d["runs"], d["med1"], d["med20"], d["med_last"]))
    m = fig_marginal(dense, os.path.join(a.out, "6_marginal_gain.png"))
    print("  6_marginal_gain.png       gain from ten more trees %s; upper 95th percentile %s"
          % (", ".join("%.3f at %d" % (v, k) for k, v in m["at"].items()),
             ", ".join("%.3f at %d" % (v, k) for k, v in m["upper"].items())))
    print("      power law A %.3f B %.3f c %.3f, rmse on medians %.3f, %d refits"
          % (*m["params"], m["rmse"], m["n_boot"]))


if __name__ == "__main__":
    main()
