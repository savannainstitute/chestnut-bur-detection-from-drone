#!/usr/bin/env python3
"""Re-derive every pooled result of a tools/pool_rescores.py run from its sources, with separate code.

pool_rescores.py refuses a bad join by its own rules, but a rule implemented once is checked only by the code
that implements it. This script imports neither pool_rescores.py nor analyze_sweep.py. For each model in the
run's models.tsv it reads the original result, the re-score files named in the AFTER file's provenance and
the design extension, and requires that:

- a pooled model's AFTER file holds the original trees unchanged and in order, followed by exactly the
  re-scores' trees, all as test trees, with every other field of the original unchanged;
- the added frames are exactly the frames the extension lists for the model's scheme and fold, no tree is
  added twice or added to a model that already scored it, and each added tree's truth equals the
  extension's crown burs;
- each re-score is of this run and records the weights md5 that models.tsv gives;
- BEFORE is a link to the original for every model in the sets, and AFTER is too for an unchanged one;
- no excluded model appears in either set, and runs_pooled.txt lists exactly the pooled and unchanged ones.

It exits 1 on any defect. On 2026-10-11 it reproduced the first pooling's counts (1,834 pooled, 19,405
trees, 179,309 burs) and caught both defects planted in a copy of it, a dropped tree and a removed truth box.

    python3 tools/check_pooled_rescores.py --out work/2026-10-11_rescore_pooled \\
        --scratch ~/scratch/bur-detection/2026-10-11_rescore_pooled [--through-extension k]
"""
import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
EXTENSION = PROJ / "work/2026-10-02_arc_sweep/design/design_extension.json"


def load(p):
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the pooling run's --out folder")
    ap.add_argument("--scratch", required=True, help="the pooling run's --scratch folder")
    ap.add_argument("--through-extension", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else PROJ / a.out
    scratch = Path(a.scratch).expanduser()

    ext = json.load(open(EXTENSION))
    through = max(e["id"] for e in ext["extensions"]) if a.through_extension is None else a.through_extension
    listed, crown = defaultdict(set), {}
    for e in ext["extensions"]:
        if e["id"] > through:
            continue
        for scheme, byfold in e["folds"].items():
            for f, stems in byfold.items():
                listed[(scheme, int(f))] |= set(stems)
        crown.update({t["stem"]: t["crown_burs"] for t in e["trees"]})

    rows = list(csv.DictReader(open(out / "models.tsv"), delimiter="\t"))
    in_sets = set((out / "runs_pooled.txt").read_text().split())
    status = Counter(r["status"] for r in rows)
    bad, example = Counter(), {}

    def flag(what, rid):
        bad[what] += 1
        example.setdefault(what, rid)

    if in_sets != {r["run_id"] for r in rows if r["status"] in ("pooled", "unchanged")}:
        flag("runs_pooled.txt is not the pooled and unchanged runs", "")
    trees = burs = 0
    deliveries = Counter()
    for r in rows:
        rid, s = r["run_id"], r["status"]
        before, after = scratch / "before" / (rid + ".json.gz"), scratch / "after" / (rid + ".json.gz")
        if s == "excluded":
            if before.exists() or after.exists() or before.is_symlink() or after.is_symlink():
                flag("an excluded model is in a set", rid)
            continue
        src = (PROJ / r["source"]).resolve()
        if not (before.is_symlink() and Path(os.readlink(before)) == src):
            flag("BEFORE is not a link to the original", rid)
        if s == "unchanged":
            if not (after.is_symlink() and Path(os.readlink(after)) == src):
                flag("an unchanged model's AFTER is not a link to the original", rid)
            continue
        orig, pooled = load(src), load(after)
        n = len(orig["trees"])
        if pooled["trees"][:n] != orig["trees"]:
            flag("original trees altered", rid)
        if {k: v for k, v in pooled.items() if k not in ("trees", "pooled")} != \
                {k: v for k, v in orig.items() if k != "trees"}:
            flag("a field of the original altered", rid)
        added = pooled["trees"][n:]
        expected = []
        for q in pooled["pooled"]["rescores"]:
            rs = load(PROJ / q["path"])
            deliveries[str(Path(q["path"]).parent)] += 1
            if rs["run"]["run_id"] != rid:
                flag("a re-score of another run", rid)
            if rs["rescore"]["weights_md5"] != r["weights_md5"]:
                flag("re-score weights md5 differs from models.tsv", rid)
            expected += rs["trees"]
        bare = lambda t: {k: v for k, v in t.items() if k not in ("ext", "split")}
        if [bare(t) for t in added] != [bare(t) for t in expected]:
            flag("added trees differ from the re-scores' trees", rid)
        if any(t.get("split") != "test" for t in added):
            flag("an added tree is not a test tree", rid)
        names = [t["tree"] for t in added]
        if len(set(names)) != len(names) or set(names) & {t["tree"] for t in orig["trees"]}:
            flag("a tree added twice or already scored", rid)
        frames = [t["frame"] for t in added]
        if len(frames) != len(set(frames)) or set(frames) != listed[(r["scheme"], int(r["fold"]))]:
            flag("added frames are not the frames listed for the fold", rid)
        if any(len(t["truth"]) != crown.get(t["frame"]) for t in added):
            flag("truth differs from the extension's crown burs", rid)
        nb = sum(len(t["truth"]) for t in added)
        trees, burs = trees + len(added), burs + nb
        if (len(added), nb) != (int(r["new_trees"]), int(r["new_burs"])):
            flag("models.tsv new tree or bur count wrong", rid)

    print("models.tsv %d rows: %s; %d in both sets" % (len(rows), dict(status), len(in_sets)))
    print("re-score files read, by delivery: %s" % dict(deliveries))
    print("new trees joined %d, burs %d, through extension %d" % (trees, burs, through))
    if bad:
        for what, v in bad.items():
            print("DEFECT: %s: %d, e.g. %s" % (what, v, example[what]))
        sys.exit(1)
    print("no defects")


if __name__ == "__main__":
    main()
