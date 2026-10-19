#!/usr/bin/env python3
"""Pool each model's re-scored new test trees with its own original trees, and compare the curves.

Step 5 of work/2026-10-07_new-annotation-plan/PLAN.md. Trees annotated after the design was frozen are
scored by the models already trained, with no retraining (`sweep_one.py --score-only`, run on CHTC). This
script joins those scores to each model's original result and writes two result sets for the SAME
models: BEFORE, the original trees only, and AFTER, the new trees added. It runs tools/analyze_sweep.py
on each and compares every cell that pools all five folds in both.

THE RULES. Each one stops the script rather than guessing, except the last, which leaves a model out.

1. A re-scored tree joins a model only if design_extension.json lists it for that model's scheme and
   fold. Under the flight scheme a new tree is a test tree only for the fold that holds its drone
   mission, because every other fold trained on other trees from that mission.
2. The model is identified by the md5 of the weights that were scored, as the re-score records it,
   looked up in the S3 weights manifests, and its original trees come from the result of THAT model.
   The 763 tranche 1 cells whose first CHTC runs kept no weights are re-scored from their second runs,
   which are different trained models, so such a model takes its original trees, and with them its
   threshold, from its second-run result. Taking them from the first run would pool two models in
   one fold.
3. The re-score must carry the training record of the result it is joined to (epochs, best epoch,
   training time, tiles, best validation metrics, validation composition, design row) and the
   campaign's scoring settings, from scoring code that passed the reproduction test. A mismatch
   means the job paired weights with the wrong result.
4. Each new tree's truth must equal the crown burs the extension records for it, and every model of
   a fold must have scored the same truth boxes. No tree may join a model twice or join a model that
   already scored it.
5. A model whose fold has new trees and which has no complete re-score is left out of BOTH sets, with
   its reason: tranche 4's tile sizes, which the re-scoring job does not cut; no weights kept; or
   weights kept and not yet re-scored. The last are listed in owed_rescores.tsv for the next job.

WHY BEFORE IS REBUILT RATHER THAN TAKEN FROM THE CAMPAIGN'S ANALYSIS. The comparison has to hold the
models fixed, so that it measures the trees added and nothing else. BEFORE is each model's own original
result, which is its filed result unless it was re-scored from a second run. For a cell whose five runs
are all filed results, BEFORE must equal the campaign's cells.csv row exactly; --check-campaign asserts
that against a campaign analysis run with the same interpreter.

The validation trees, and so each run's own threshold, are the original result's in both sets. A
re-score adds test trees and nothing else, and the comparison asserts that every matched cell's
thresholds are unchanged and that it gained exactly the new trees and burs its folds were owed.

    ~/.venvs/chestnut-yield/bin/python3 tools/pool_rescores.py \\
        --rescore work/2026-10-08_chtc_rescore/results \\
        --rescore work/2026-10-10_chtc_rescore/results \\
        --scratch ~/scratch/bur-detection/2026-10-11_rescore_pooled \\
        --out work/2026-10-11_rescore_pooled [--check-campaign <campaign cells.csv>]

`--rescore` is repeatable, one directory per re-scoring delivery, so a later extension's trees join
the same models. `--through-extension k` pools extensions 1 to k only; trees of later extensions are
set aside and counted.

KEPT, because it encodes this campaign's design and pooling rules and is the provenance of every number
computed on the extended test set.
"""
import argparse
import contextlib
import csv
import gzip
import io
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path.insert(0, str(HERE))
import analyze_sweep as an  # noqa: E402

DESIGN = PROJ / "work/2026-10-02_arc_sweep/design/design.json"
EXTENSION = PROJ / "work/2026-10-02_arc_sweep/design/design_extension.json"
SECOND_RUNS = PROJ / "work/2026-10-05_chtc_second_runs_t1/results"
NO_WEIGHTS = PROJ / "work/2026-10-05_chtc_second_runs_t1/expected_second_runs.txt"
# Which trained model a weights md5 is. A first run's weights belong with its filed result, a second
# run's with its second-run result.
MANIFESTS = (("first", PROJ / "work/2026-10-07_new-annotation-plan/s3_weights_arc.tsv"),
             ("first", PROJ / "work/2026-10-04_s3_weights_chtc.tsv"),
             ("second", PROJ / "work/2026-10-05_s3_weights_chtc_second_runs.tsv"))
# score_frames in tools/sweep_one.py, which produced every filed result
SCORING = dict(conf=0.15, infer_batch=64, iou_tile=0.45, iou_frame=0.3)
# Scoring code whose re-scores reproduced the filed results (counts equal on 1,018 of 1,019 frames,
# work/2026-10-08_chtc_rescore/repro_compare.tsv). A new version must pass that test before it joins.
VALIDATED_CODE = {"ff89d1d36642f5059725fced96a5cf0b"}
# What a re-score replaces or records about its own machine; everything else it copies from the
# result it was paired with, so everything else must be equal.
OWN_FIELDS = {"trees", "infer_seconds", "rescore", "device", "host", "host_env", "slurm_job"}
KEY = ("tranche", "arm", "scheme", "seed", "rule", "nathan_rule", "n_ren", "n_nathan", "tile")
COMPARE = ("gate_resid_sd", "gate_mae", "gate_bias", "gate_slope", "gate_pearson", "gate_within_pearson",
           "gate_folds_lowr", "gate_fn_pooled", "gate_fn_gap", "spearman", "top10_capture",
           "thr_gate_median")


def load(p):
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


def refuse(msg):
    raise SystemExit("REFUSED: " + msg)


def rel(p):
    """A path relative to the project when it is inside it, so the records read on either machine."""
    p = Path(p).resolve()
    return str(p.relative_to(PROJ.resolve())) if p.is_relative_to(PROJ.resolve()) else str(p)


def cellkey(r):
    return tuple(str(r.get(k, "random") if k == "nathan_rule" else r[k]) for k in KEY)


def weights_index():
    idx = {}
    for kind, path in MANIFESTS:
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                v = (kind, row["run_id"])
                if idx.get(row["md5"], v) != v:
                    refuse("weights md5 %s is listed for %s and for %s" % (row["md5"], idx[row["md5"]], v))
                idx[row["md5"]] = v
    return idx


def filed_results():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dirs = an.discover_results_trees([str(PROJ / "work")])
    where = {}
    for d in dirs:
        for p in Path(d).rglob("*.json.gz"):
            rid = p.name[:-8]
            if rid in where:
                refuse("%s is filed twice, %s and %s" % (rid, where[rid], p))
            where[rid] = p
    return where, buf.getvalue()


def extension_index(ext, through):
    """listed[(scheme, fold)][stem] = extension id, over EVERY extension, so that a tree beyond the
    cutoff can be told from one that was never listed."""
    listed, crown, truth_classes = defaultdict(dict), {}, sorted(ext["classes"])
    for e in ext["extensions"]:
        for t in e["trees"]:
            if t["stem"] in crown:
                refuse("%s is in two extensions" % t["stem"])
            crown[t["stem"]] = t["crown_burs"]
        for scheme, byfold in e["folds"].items():
            for fold, stems in byfold.items():
                for s in stems:
                    if s in listed[(scheme, int(fold))]:
                        refuse("%s is listed twice for %s fold %s" % (s, scheme, fold))
                    listed[(scheme, int(fold))][s] = e["id"]
    ids = [e["id"] for e in ext["extensions"]]
    through = max(ids) if through is None else through
    if through not in ids:
        refuse("no extension %s; the file has %s" % (through, ids))
    return listed, crown, truth_classes, through


def build(a):
    design = json.load(open(DESIGN))
    runs = {r["run_id"]: r for r in design["runs"]}
    ext = json.load(open(EXTENSION))
    listed, crown, classes, through = extension_index(ext, a.through_extension)
    widx = weights_index()
    byrun = {v: m for m, v in widx.items()}
    filed, discover_log = filed_results()
    second = {p.name[:-8]: p for p in SECOND_RUNS.glob("*.json.gz")}
    no_weights = set(NO_WEIGHTS.read_text().split())

    rescores = defaultdict(list)
    for d in a.rescore:
        for p in sorted(Path(d).glob("*.json.gz")):
            rid = p.name[:-8]
            if any(Path(q).parent == p.parent for q in rescores[rid]):
                refuse("%s appears twice in %s" % (rid, d))
            rescores[rid].append(p)
    stray = sorted(set(rescores) - set(filed))
    if stray:
        refuse("re-scores for runs with no filed result: %s" % stray[:8])

    truth_seen = {}           # frame -> the truth boxes the first model to score it saw
    models, owed = [], []
    set_aside = Counter()
    for rid in sorted(filed):
        if rid not in runs:
            continue
        r = runs[rid]
        scheme, fold = r["scheme"], int(r["fold"])
        expect = {s for s, eid in listed[(scheme, fold)].items() if eid <= through}
        row = dict(run_id=rid, tranche=r["tranche"], arm=r["arm"], scheme=scheme, fold=fold, seed=r["seed"],
                   rule=r["rule"], nathan_rule=r.get("nathan_rule", "random"), n_ren=r["n_ren"],
                   n_nathan=r["n_nathan"], tile=r["tile"], status="", reason="",
                   source="", weights_kind="", weights_md5="", old_test=0, old_val=0, expected_new=len(expect),
                   new_trees=0, new_burs=0, code_md5="")
        if not rescores.get(rid):
            if expect:
                row["status"] = "excluded"
                if r["tile"] != 640:
                    row["reason"] = "tile %d: the re-scoring job cuts 640 px tiles only" % r["tile"]
                elif rid in no_weights and rid not in second:
                    row["reason"] = "no weights kept; waits for its second run"
                else:
                    kind = "second" if rid in no_weights else "first"
                    have = byrun.get((kind, rid))
                    row["reason"] = ("weights kept, not yet re-scored" if have
                                     else "no weights in the manifests")
                    if have:
                        owed.append(dict(run_id=rid, scheme=scheme, fold=fold, tranche=r["tranche"],
                                         weights_kind=kind, weights_md5=have,
                                         expected_new=len(expect)))
                models.append(row)
                continue
            # nothing to add for this fold; the model joins both sets as it is
            use_second = rid in no_weights and rid in second
            src_path = second[rid] if use_second else filed[rid]
            src = load(src_path)
            row.update(status="unchanged", source=rel(src_path),
                       weights_kind="second" if use_second else "first",
                       old_test=sum(t.get("split", "test") == "test" for t in src["trees"]),
                       old_val=sum(t.get("split", "test") != "test" for t in src["trees"]))
            models.append(dict(row, _src=src_path, _after=None))
            continue

        kinds, added, rblocks, src_path, src = set(), [], [], None, None
        for p in rescores[rid]:
            d = load(p)
            rb = d["rescore"]
            hit = widx.get(rb["weights_md5"])
            if hit is None:
                refuse("%s was scored with weights %s, which no manifest lists" % (p, rb["weights_md5"]))
            kind, mrid = hit
            if mrid != rid:
                refuse("%s was scored with the weights of %s" % (p, mrid))
            if kind == "first" and rid in no_weights:
                refuse("%s is one of the 763 with no first-run weights, yet %s names first-run weights"
                       % (rid, p))
            kinds.add((kind, rb["weights_md5"]))
            if len(kinds) > 1:
                refuse("%s was re-scored with two different weights: %s" % (rid, sorted(kinds)))
            path = second.get(rid) if kind == "second" else filed[rid]
            if path is None:
                refuse("%s was re-scored from second-run weights and its second-run result is not filed" % rid)
            if src is None:
                src_path, src = path, load(path)
            diff = sorted(k for k in (set(d) | set(src)) - OWN_FIELDS if d.get(k) != src.get(k))
            if diff:
                refuse("%s does not carry the training record of %s: %s differ" % (p, path, diff))
            for k, v in SCORING.items():
                if rb.get(k) != v:
                    refuse("%s scored with %s = %s, the campaign uses %s" % (p, k, rb.get(k), v))
            if rb.get("tile") != r["tile"] or sorted(rb.get("classes", [])) != classes:
                refuse("%s scored at tile %s classes %s" % (p, rb.get("tile"), rb.get("classes")))
            if rb.get("code_md5") not in VALIDATED_CODE and rb.get("code_md5") not in a.accept_code_md5:
                refuse("%s was scored by code %s, which has not passed the reproduction test"
                       % (p, rb.get("code_md5")))
            if rb.get("n_val", 0) != 0 or rb.get("n_test") != len(d["trees"]):
                refuse("%s records %s test and %s validation frames for %d trees"
                       % (p, rb.get("n_test"), rb.get("n_val"), len(d["trees"])))
            old_frames = {t["frame"] for t in src["trees"]}
            for t in d["trees"]:
                f = t["frame"]
                if t.get("split") != "test" or t["tree"] != f.split("__")[0]:
                    refuse("%s holds %s as split %s, tree %s" % (p, f, t.get("split"), t["tree"]))
                eid = listed[(scheme, fold)].get(f)
                if eid is None:
                    where = [k for k, v in listed.items() if f in v]
                    refuse("%s scored %s, which the extension does not list for %s fold %d (listed for %s)"
                           % (p, f, scheme, fold, where or "no fold"))
                if f in old_frames:
                    refuse("%s scored %s, which %s already scored" % (p, f, src_path))
                if any(x["frame"] == f for x in added):
                    refuse("%s scored %s a second time" % (p, f))
                if eid > through:
                    set_aside[eid] += 1
                    continue
                if len(t["truth"]) != crown[f]:
                    refuse("%s gives %s %d truth boxes, the extension records %d crown burs"
                           % (p, f, len(t["truth"]), crown[f]))
                tb = sorted(tuple(round(v, 3) for v in b) for b in t["truth"])
                if truth_seen.setdefault(f, tb) != tb:
                    refuse("%s gives %s different truth boxes from another model of its fold" % (p, f))
                added.append(dict(t, ext=eid))
            rblocks.append(dict(path=rel(p), weights_md5=rb["weights_md5"],
                                code_md5=rb.get("code_md5"), when=rb.get("when"), n_trees=len(d["trees"])))
        got = {t["frame"] for t in added}
        kind, md5 = next(iter(kinds))
        row.update(source=rel(src_path), weights_kind=kind, weights_md5=md5,
                   old_test=sum(t.get("split", "test") == "test" for t in src["trees"]),
                   old_val=sum(t.get("split", "test") != "test" for t in src["trees"]),
                   new_trees=len(added), new_burs=sum(len(t["truth"]) for t in added),
                   code_md5="|".join(sorted({b["code_md5"] for b in rblocks})))
        if got != expect:
            row.update(status="excluded", reason="re-scored %d of the %d trees its fold is owed"
                       % (len(got & expect), len(expect)))
            print("WARNING: %s %s" % (rid, row["reason"]))
            models.append(row)
            continue
        merged = dict(src, trees=src["trees"] + added,
                      pooled=dict(source=row["source"], weights_kind=kind, weights_md5=md5,
                                  through_extension=through, rescores=rblocks))
        row["status"] = "pooled"
        models.append(dict(row, _src=src_path, _after=merged))
    return dict(models=models, owed=owed, set_aside=set_aside, through=through,
                discover_log=discover_log, crown=crown, listed=listed)


def write_sets(b, scratch):
    scratch = Path(scratch).expanduser()
    stamp = scratch / ".pool_rescores"
    if scratch.exists() and any(scratch.iterdir()) and not stamp.exists():
        refuse("%s exists and was not written by this script" % scratch)
    for sub in ("before", "after"):
        if (scratch / sub).exists():
            shutil.rmtree(scratch / sub)
        (scratch / sub).mkdir(parents=True)
    stamp.write_text("written by tools/pool_rescores.py; regenerable, delete freely\n")
    n = 0
    for m in b["models"]:
        if "_src" not in m:
            continue
        (scratch / "before" / (m["run_id"] + ".json.gz")).symlink_to(m["_src"].resolve())
        dst = scratch / "after" / (m["run_id"] + ".json.gz")
        if m["_after"] is None:
            dst.symlink_to(m["_src"].resolve())
        else:
            with gzip.open(dst, "wt") as fh:
                json.dump(m["_after"], fh)
        n += 1
    return scratch, n


def run_analysis(results, out, expect):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "analyze_sweep.py"), "--design", str(DESIGN), "--results", str(results),
           "--out", str(out), "--expect-runs", str(expect)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (out / "analysis_log.txt").write_text(" ".join(cmd) + "\n\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        refuse("analyze_sweep.py failed on %s; see %s" % (results, out / "analysis_log.txt"))


def read_cells(path):
    with open(path, newline="") as fh:
        return {cellkey(r): r for r in csv.DictReader(fh) if r["omit_fold"] == ""}


def fnum(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def compare(b, out, check_campaign):
    before = read_cells(out / "analysis_before" / "cells.csv")
    after = read_cells(out / "analysis_after" / "cells.csv")
    new_trees = {s: sum(len([1 for st, eid in b["listed"][(s, f)].items() if eid <= b["through"]])
                        for f in range(5)) for s in ("flight", "tree")}
    new_burs = {s: sum(b["crown"][st] for f in range(5) for st, eid in b["listed"][(s, f)].items()
                       if eid <= b["through"]) for s in ("flight", "tree")}
    matched = sorted(k for k in before if k in after and before[k]["folds"] == "5" and after[k]["folds"] == "5")
    rows = []
    for k in matched:
        x, y = before[k], after[k]
        s = x["scheme"]
        if (x["thr_gate_median"], x["thr_own_median"]) != (y["thr_gate_median"], y["thr_own_median"]):
            refuse("cell %s changed its thresholds when trees were added" % (k,))
        dt = int(y["n_trees_scored"]) - int(x["n_trees_scored"])
        db = int(y["n_burs_scored"]) - int(x["n_burs_scored"])
        if (dt, db) != (new_trees[s], new_burs[s]):
            refuse("cell %s gained %d trees and %d burs, its folds are owed %d and %d"
                   % (k, dt, db, new_trees[s], new_burs[s]))
        row = {kk: x[kk] for kk in KEY}
        row.update(trees_before=x["n_trees_scored"], trees_after=y["n_trees_scored"],
                   burs_before=x["n_burs_scored"], burs_after=y["n_burs_scored"])
        for m in COMPARE:
            row[m + "_before"], row[m + "_after"] = x[m], y[m]
        rows.append(row)
    with open(out / "compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else list(KEY))
        w.writeheader()
        w.writerows(rows)

    checked = None
    if check_campaign:
        camp = read_cells(check_campaign)
        src = defaultdict(set)
        for m in b["models"]:
            if "_src" in m:
                src[cellkey(m)].add(m["weights_kind"])
        same = [k for k in matched if src[k] == {"first"} and k in camp]
        bad = [k for k in same if camp[k] != before[k]]
        if bad:
            cols = [c for c in camp[bad[0]] if camp[bad[0]][c] != before[bad[0]].get(c)]
            refuse("%d of %d cells differ from the campaign analysis, e.g. %s in %s"
                   % (len(bad), len(same), bad[0], cols[:6]))
        checked = len(same)
    return before, after, matched, rows, checked, new_trees, new_burs


def curves(before, after, matched, out):
    """The dense no-Ohio curve and what the next ten trees buy, before and after, on the matched cells,
    computed by the figure code's own functions."""
    import make_figures as mf
    lines = []
    sel = {"before": mf.dense_cells([before[k] for k in matched]),
           "after": mf.dense_cells([after[k] for k in matched])}
    lv_b, by_b, med_b = mf._medians(sel["before"])
    lv_a, by_a, med_a = mf._medians(sel["after"])
    assert lv_a == lv_b and all(len(by_a[n]) == len(by_b[n]) for n in lv_a)
    lines.append("DENSE NO-OHIO CURVE, flight scheme, cells pooling all five folds in both sets: %d levels, "
                 "%d runs" % (len(lv_a), sum(len(by_a[n]) for n in lv_a)))
    lines.append("  median count error a tree, burs (gate_resid_sd), before -> after, runs:")
    for n, mb, ma in zip(lv_a, med_b, med_a):
        lines.append("    %3d trees  %.3f -> %.3f   (%d)" % (n, mb, ma, len(by_a[n])))
    for name, (lv, by, med) in (("before", (lv_b, by_b, med_b)), ("after", (lv_a, by_a, med_a))):
        x = np.array(lv, float)
        p = mf._fit_power(x, med)
        rmse = float(np.sqrt(np.mean((med - mf._power(x, *p)) ** 2)))
        rng = np.random.default_rng(20261006)
        at = {}
        boots = []
        for _ in range(300):
            mb = np.array([np.median(rng.choice(by[n], size=len(by[n]), replace=True)) for n in lv])
            try:
                pb = mf._fit_power(x, mb)
            except Exception:
                continue
            boots.append([mf._power(g, *pb) - mf._power(g + 10, *pb) for g in (10, 32, 93)])
        hi = np.percentile(np.array(boots), 95, axis=0)
        for g in (10, 32, 93):
            at[g] = float(mf._power(g, *p) - mf._power(g + 10, *p))
        lines.append("  %s: power law %.3f + %.3f n^-%.3f, error on the medians %.3f; the next ten trees "
                     "remove %.3f burs a tree at 10, %.3f at 32, %.3f at 93 (95th percentile of %d refits "
                     "%.3f, %.3f, %.3f)" % (name, p[0], p[1], p[2], rmse, at[10], at[32], at[93], len(boots),
                                            hi[0], hi[1], hi[2]))
    for name, d in sel.items():
        lvl = defaultdict(lambda: [0, 0])
        for r in d:
            n = int(r["n_ren"])
            v = fnum(r, "gate_folds_lowr")
            lvl[n][0] += int(v) if np.isfinite(v) else 0
            lvl[n][1] += int(r["folds"])
        parts = []
        for lo, hi_ in mf.BANDS:
            f = sum(lvl[n][0] for n in lvl if lo <= n <= hi_)
            t = sum(lvl[n][1] for n in lvl if lo <= n <= hi_)
            parts.append("%s %d/%d" % (("%d" % lo) if lo == hi_ else ("%d-%d" % (lo, hi_)), f, t))
        lines.append("  failed folds (within-fold r under 0.85), %s: %s" % (name, ", ".join(parts)))
    return lines


SHIFTS = ("gate_resid_sd", "gate_resid_frac", "gate_pearson", "gate_within_pearson", "gate_mae", "gate_nmae",
          "gate_bias", "gate_slope", "gate_fn_pooled", "gate_fn_gap", "spearman")


def shifts(before, after, matched):
    """How each reading moved when the trees were added, over the matched cells. Adding trees changes the
    spread of the true counts, and the scatter in burs moves with that spread; the correlation and the
    scatter as a fraction of the spread (resid_frac) are the readings free of it."""
    lines = ["HOW EACH READING MOVED, median over the %d matched cells, before -> after:" % len(matched)]
    for s in sorted({before[k]["scheme"] for k in matched}):
        ks = [k for k in matched if before[k]["scheme"] == s]
        lines.append("  %s scheme, true counts: mean %s -> %s burs a tree, sd %s -> %s"
                     % (s, before[ks[0]]["mean_true"], after[ks[0]]["mean_true"],
                        before[ks[0]]["true_sd"], after[ks[0]]["true_sd"]))
    for m in SHIFTS:
        b = np.array([fnum(before[k], m) for k in matched])
        a = np.array([fnum(after[k], m) for k in matched])
        ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]
        lines.append("  %-20s %.4f -> %.4f; change %+.4f (middle half %+.4f to %+.4f), up in %d of %d cells"
                     % (m, np.median(b[ok]), np.median(a[ok]), np.median(d), np.percentile(d, 25),
                        np.percentile(d, 75), int((d > 0).sum()), int(ok.sum())))
    return lines


def grid_table(before, after, matched):
    lines = ["MAIN GRID, tranche 1, flight scheme, random rule, cells pooling all five folds in both sets:",
             "  Ohio frames: Clarks Run trees, count error before -> after (cells)"]
    by = defaultdict(lambda: defaultdict(list))
    for k in matched:
        x, y = before[k], after[k]
        if x["tranche"] == "1" and x["arm"] == "grid" and x["scheme"] == "flight" and x["rule"] == "random":
            by[int(x["n_nathan"])][int(x["n_ren"])].append((fnum(x, "gate_resid_sd"), fnum(y, "gate_resid_sd")))
    for lv in sorted(by):
        cells = ["%d: %.2f -> %.2f (%d)" % (n, np.mean([v[0] for v in by[lv][n]]),
                                             np.mean([v[1] for v in by[lv][n]]), len(by[lv][n]))
                 for n in sorted(by[lv])]
        lines.append("  %2d  %s" % (lv, "; ".join(cells)))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rescore", action="append", required=True,
                    help="a directory of new-frame re-score results; repeatable, one per delivery")
    ap.add_argument("--scratch", required=True, help="where the two result sets are written; regenerable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--through-extension", type=int, default=None)
    ap.add_argument("--accept-code-md5", action="append", default=[],
                    help="scoring code that has since passed the reproduction test")
    ap.add_argument("--check-campaign", help="the campaign's cells.csv from the same interpreter")
    a = ap.parse_args()
    a.rescore = [str((PROJ / d) if not Path(d).is_absolute() else Path(d)) for d in a.rescore]

    b = build(a)
    out = Path(a.out)
    out = out if out.is_absolute() else PROJ / out
    out.mkdir(parents=True, exist_ok=True)
    scratch, n = write_sets(b, a.scratch)
    pop = sorted(m["run_id"] for m in b["models"] if "_src" in m)
    (out / "runs_pooled.txt").write_text("".join(r + "\n" for r in pop))
    cols = [c for c in b["models"][0] if not c.startswith("_")]
    with open(out / "models.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(b["models"])
    with open(out / "owed_rescores.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id", "tranche", "scheme", "fold", "weights_kind",
                                           "weights_md5", "expected_new"], delimiter="\t")
        w.writeheader()
        w.writerows(b["owed"])

    st = Counter((m["status"], m["reason"]) for m in b["models"])
    summary = ["POOLING through extension %d: %d models in both sets (%d with new trees, %d unchanged), "
               "%d left out" % (b["through"], n, st[("pooled", "")], st[("unchanged", "")],
                                sum(v for (s, _), v in st.items() if s == "excluded"))]
    for (s, why), v in sorted(st.items()):
        if s == "excluded":
            summary.append("  left out, %s: %d" % (why, v))
    if b["set_aside"]:
        summary.append("  trees of later extensions set aside: %s" % dict(b["set_aside"]))
    pooled = [m for m in b["models"] if m["status"] == "pooled"]
    summary.append("  new trees joined: %d, burs %d; models re-scored from second-run weights: %d"
                   % (sum(m["new_trees"] for m in pooled), sum(m["new_burs"] for m in pooled),
                      sum(m["weights_kind"] == "second" for m in pooled)))

    run_analysis(scratch / "before", out / "analysis_before", out / "runs_pooled.txt")
    run_analysis(scratch / "after", out / "analysis_after", out / "runs_pooled.txt")
    before, after, matched, rows, checked, nt, nb = compare(b, out, a.check_campaign)
    bytr = Counter((before[k]["tranche"], before[k]["scheme"]) for k in matched)
    summary.append("CELLS pooling all five folds in both sets: %d (%s); each gained %d trees and %d burs "
                   "(flight) or %d and %d (tree), thresholds unchanged"
                   % (len(matched), ", ".join("tranche %s %s %d" % (t, s, v) for (t, s), v in sorted(bytr.items())),
                      nt["flight"], nb["flight"], nt["tree"], nb["tree"]))
    if checked is not None:
        summary.append("  BEFORE equals the campaign analysis on all %d matched cells whose runs are all filed "
                       "results" % checked)
    summary += shifts(before, after, matched)
    summary += curves(before, after, matched, out)
    summary += grid_table(before, after, matched)
    text = "\n".join(summary) + "\n"
    (out / "SUMMARY.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
