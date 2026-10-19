#!/usr/bin/env python3
"""Add newly annotated Clarks Run trees to the learning-curve campaign as TEST trees.

A tree annotated after the design was frozen is in no training run, so every trained model can score
it with no retraining (`sweep_one.py --score-only`). This script decides which models may.

  flight scheme  A new tree is a test tree only for the fold that holds its drone mission, because
                 every other fold trained on other trees from that mission. The mission comes from
                 the raw-imagery folder of the tree's frame, through frame_missions.csv, and the
                 mission-to-fold table is derived from the design's own flight-scheme TEST trees.
  tree scheme    A new tree goes to exactly one fold by the design's own rule, sorted by crown burs
                 and dealt round-robin (make_sweep_design.stratified_folds), continuing the deal
                 from where earlier extensions left off, so that pooled over the five folds each
                 tree is scored once.

It APPENDS one entry to the design extension file and never edits an earlier one, then writes the
per-fold stem lists and the Drive file list that the re-scoring jobs need into a new folder. Run it
after tools/pull_ren_annotations.sh has filed a delivery. With nothing new it writes nothing, so
running it twice is harmless.

    python3 tools/extend_test_trees.py \
        --design work/2026-10-02_arc_sweep/design/design.json \
        --extension work/2026-10-02_arc_sweep/design/design_extension.json \
        --labels inputs/ren-annotations/ClarksRun_set_A/labels/detect [--labels <other set> ...] \
        --frame-missions work/2026-10-07_new-annotation-plan/frame_missions.csv \
        --date YYYY-MM-DD --source "<where the delivery came from>" \
        --out work/<date>_new-test-trees \
        [--drive-md5 <rclone md5sum output, paths relative to the KY-2026 Drive folder>] \
        [--raw-root ~/Data/chestnut-drone/ClarksRun_Maysville-KY/2026/raw-imagery] [--dry-run]

--drive-md5 and --raw-root are checks: every new frame must be on Drive, its raw flight copy must sit
in the mission folder that frame_missions.csv names, and the two copies must be byte-identical.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

N_FOLDS = 5
# Read from the design's flight-scheme test trees and asserted, never assumed. CRM0k in the design
# is M0k on disk, and M03 is spelled m03 on disk.
EXPECTED_MISSION_FOLD = {"M01": 1, "M02": 0, "M03": 2, "M04": 4, "M05": 3, "M06": 4, "M07": 0, "M08": 3}


def norm_mission(m):
    m = m.strip().upper()
    return "M" + m[3:] if m.startswith("CRM") else m


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def crown_burs(path, keep):
    n = 0
    for line in open(path):
        f = line.split()
        if f and int(f[0]) in keep:
            n += 1
    return n


def refuse(msg, items=()):
    print("REFUSED: " + msg, file=sys.stderr)
    for it in list(items)[:20]:
        print("   " + str(it), file=sys.stderr)
    if len(items) > 20:
        print("   ... and %d more" % (len(items) - 20), file=sys.stderr)
    sys.exit(1)


def folds_fingerprint(design):
    """The design's fold membership, which every assignment here depends on."""
    s = {sch: [sorted(f["test"]) for f in design["schemes"][sch]["folds"]] for sch in ("flight", "tree")}
    return hashlib.md5(json.dumps(s, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--extension", required=True)
    ap.add_argument("--labels", required=True, action="append")
    ap.add_argument("--frame-missions", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drive-md5")
    ap.add_argument("--raw-root")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    design_path = Path(a.design)
    design = json.load(open(design_path))
    manifest = design_path.parent / "train640_MANIFEST.json"
    keep = tuple(json.load(open(manifest))["classes"])     # the truth classes training used
    design_trees = set(design["ren_boxes"])

    # ---- the mission-to-fold table, from flight-scheme TEST trees only ----------------------------
    table = {}
    for fi, f in enumerate(design["schemes"]["flight"]["folds"]):
        for s in f["test"]:
            m = norm_mission(design["mission"][s])
            if table.setdefault(m, fi) != fi:
                refuse("mission %s holds test trees in two flight folds" % m)
    if table != EXPECTED_MISSION_FOLD:
        refuse("the design's mission-to-fold table is not the recorded one", [table])

    # ---- frame_missions.csv must agree with the design on every design tree -----------------------
    fm = {}
    for r in csv.DictReader(open(a.frame_missions)):
        m = norm_mission(r["mission"])
        prev = fm.setdefault(r["frame"], {"mission": m, "sets": []})
        if prev["mission"] != m:
            refuse("frame %s is listed under two missions" % r["frame"])
        prev["sets"].append(r["set"])
    bad = [(s, norm_mission(design["mission"][s]), fm.get(s, {}).get("mission"))
           for s in sorted(design_trees) if fm.get(s, {}).get("mission") != norm_mission(design["mission"][s])]
    if bad:
        refuse("frame_missions.csv disagrees with the design's own missions on %d design trees" % len(bad), bad)

    # ---- the extension file, append-only ----------------------------------------------------------
    ext_path = Path(a.extension)
    if ext_path.exists():
        ext = json.load(open(ext_path))
        if ext["base_design"]["folds_md5"] != folds_fingerprint(design):
            refuse("the design's fold membership has changed since the extension file was started")
    else:
        ext = dict(
            about=("Append-only extension of the learning-curve design: trees annotated after the design "
                   "was frozen, each a TEST tree for existing models and never a training or validation "
                   "tree. Written only by tools/extend_test_trees.py, which appends one entry per "
                   "delivery and never edits an earlier one."),
            base_design=dict(file=design_path.name, md5=md5_file(design_path), n_runs=len(design["runs"]),
                             n_ren_trees=len(design_trees), folds_md5=folds_fingerprint(design)),
            mission_fold=table, classes=list(keep), extensions=[])
    before = json.dumps(ext["extensions"], sort_keys=True)
    already = {t["stem"] for e in ext["extensions"] for t in e["trees"]}

    # ---- the candidates: held labels in neither the design nor an earlier extension ---------------
    held = {}
    for d in a.labels:
        for f in sorted(os.listdir(d)):
            if f.endswith(".txt"):
                s = f[:-4]
                if s in held:
                    refuse("label %s is held in two of the --labels folders" % s, [held[s], d])
                held[s] = os.path.join(d, f)
    new = sorted(s for s in held if s not in design_trees and s not in already)
    print("labels held %d; design trees %d; in earlier extensions %d; new %d"
          % (len(held), len(design_trees), len(already), len(new)))
    if not new:
        print("nothing new: no extension written")
        return

    missing = [s for s in new if s not in fm]
    if missing:
        refuse("%d new frames are not in frame_missions.csv, so their mission is unknown" % len(missing), missing)
    unmapped = [(s, fm[s]["mission"]) for s in new if fm[s]["mission"] not in table]
    if unmapped:
        refuse("%d new frames come from a mission no fold holds; that needs a decision" % len(unmapped), unmapped)

    rows = []
    for s in new:
        sets = [x for x in fm[s]["sets"] if x.startswith("ClarksRun_set_")]
        if len(sets) != 1:
            refuse("frame %s is not in exactly one ClarksRun_set_ folder" % s, fm[s]["sets"])
        rows.append(dict(stem=s, tree=s.split("__")[0], set=sets[0], mission=fm[s]["mission"],
                         flight_fold=table[fm[s]["mission"]], crown_burs=crown_burs(held[s], keep),
                         label_md5=md5_file(held[s])))

    # tree scheme: the design's own rule, continuing the deal across extensions
    offset = sum(len(e["trees"]) for e in ext["extensions"])
    for i, r in enumerate(sorted(rows, key=lambda r: (r["crown_burs"], r["stem"]))):
        r["tree_fold"] = (offset + i) % N_FOLDS

    # ---- the frame checks -------------------------------------------------------------------------
    if a.drive_md5:
        drive = {}
        for line in open(a.drive_md5):
            if line.strip():
                h, p = line.rstrip("\n").split("  ", 1)
                drive[p] = h
        absent = [r["stem"] for r in rows if "%s/%s.JPG" % (r["set"], r["stem"]) not in drive]
        if absent:
            refuse("%d new frames are not in the Drive md5 listing" % len(absent), absent)
        for r in rows:
            r["jpg_md5_drive"] = drive["%s/%s.JPG" % (r["set"], r["stem"])]
    if a.raw_root:
        from PIL import Image
        index = {}
        for dp, _, fns in os.walk(os.path.expanduser(a.raw_root)):
            for fn in fns:
                if fn.endswith(".JPG"):
                    if fn in index:
                        refuse("raw file name %s occurs under two folders" % fn, [index[fn], dp])
                    index[fn] = dp
        for r in rows:
            fn = r["stem"].split("__")[1] + ".JPG"
            if fn not in index:
                refuse("no raw flight copy of %s" % r["stem"])
            path = os.path.join(index[fn], fn)
            mission_dir = norm_mission(os.path.relpath(index[fn], os.path.expanduser(a.raw_root)).split(os.sep)[0])
            if mission_dir != r["mission"]:
                refuse("%s sits in raw folder %s, not %s" % (r["stem"], mission_dir, r["mission"]))
            r["jpg_md5_raw"] = md5_file(path)
            r["jpg_bytes"] = os.path.getsize(path)
            with Image.open(path) as im:
                r["jpg_px"] = "%dx%d" % im.size
            if a.drive_md5 and r["jpg_md5_raw"] != r["jpg_md5_drive"]:
                refuse("the Drive copy and the raw flight copy of %s differ" % r["stem"])

    # ---- the entry, and what it writes ------------------------------------------------------------
    folds = {sch: {str(k): sorted(r["stem"] for r in rows if r[sch + "_fold"] == k) for k in range(N_FOLDS)}
             for sch in ("flight", "tree")}
    entry = dict(id=len(ext["extensions"]) + 1, date=a.date, source=a.source,
                 use="test only: never a training or validation tree",
                 labels_dirs=a.labels, trees=rows, folds=folds,
                 code_md5=md5_file(__file__))
    for sch in ("flight", "tree"):
        print("%-6s scheme, new test trees by fold: %s" % (sch, [len(folds[sch][str(k)]) for k in range(N_FOLDS)]))
    print("crown burs on the new trees: %d" % sum(r["crown_burs"] for r in rows))
    if a.dry_run:
        print("dry run: nothing written")
        return

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        refuse("%s exists and is not empty; this script never overwrites" % out)
    (out / "stems").mkdir(parents=True, exist_ok=True)
    for sch in ("flight", "tree"):
        for k in range(N_FOLDS):
            (out / "stems" / ("new_%s_fold%d.txt" % (sch, k))).write_text(
                "".join(s + "\n" for s in folds[sch][str(k)]))
    cols = ["stem", "tree", "set", "mission", "flight_fold", "tree_fold", "crown_burs", "label_md5",
            "jpg_md5_drive", "jpg_md5_raw", "jpg_bytes", "jpg_px"]
    with open(out / "new_trees.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[c for c in cols if any(c in r for r in rows)])
        w.writeheader()
        w.writerows(rows)
    (out / "drive_files.txt").write_text("".join("%s/%s.JPG\n" % (r["set"], r["stem"]) for r in rows))

    ext["extensions"].append(entry)
    assert json.dumps(ext["extensions"][:-1], sort_keys=True) == before   # earlier entries untouched
    tmp = ext_path.with_suffix(".json.part")
    with open(tmp, "w") as fh:
        json.dump(ext, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, ext_path)
    print("appended extension %d to %s and wrote %s" % (entry["id"], ext_path, out))


if __name__ == "__main__":
    main()
