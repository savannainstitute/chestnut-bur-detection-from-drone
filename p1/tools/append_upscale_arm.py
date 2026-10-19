"""Append the 2.35x upscale arm to the design: 20 runs that repeat 20 filed tranche 1 runs with Nathan's six
older-tree frames cut at 2.35x instead of 1.5x.

WHY. Those six frames hold 7,117 of his 9,407 boxes, and at 1.5x their burs measured 43.6 px against 68.4 px at
Clarks Run. Per thousand boxes they bought nothing measurable (-0.0020 burs of MAE, se 0.0073) while his native
frames bought -0.0801 (se 0.0214). But the six are also exactly his dense-canopy frames, so scale and canopy
regime could not be separated inside the campaign. Re-cut at 2.35x, their burs measure 60.8 to 72.5 px by frame.
Repeating runs that drew them, with everything else held fixed, attributes the difference to scale alone.

WHICH RUNS. Tranche 1, flight scheme, seed 1, Clarks Run 0 and 6 trees, Ohio 7 and 17 frames, all five folds.
Seed 1 is the seed whose Ohio draws hold the most of the six: 2 at 7 frames and 4 at 17 (seed 2 draws 0 and 2,
seed 3 draws 1 and 1). Each new run copies its source run's record exactly, including its training and
validation draws, and adds:
  tilestore      "train640_up235", the store that must be passed to sweep_one.py as --tilestore <dir>/..;
                 it keeps the train640 layout, so a task given the ordinary store would silently train at 1.5x.
                 Whoever runs these must check that the six frames' MANIFEST scale is 2.35 before training.
  paired_with    the source run id, for the paired comparison.

It APPENDS, like tools/append_tranche6.py: new ids past the current maximum, every existing run asserted
byte-identical, an atomic write, and a refusal to add a run twice. Since 2026-10-14 the design.json in
work/2026-10-02_arc_sweep/design/ is the only copy and is authoritative.

    python3 tools/append_upscale_arm.py --design work/2026-10-02_arc_sweep/design/design.json [--dry-run]
"""
import argparse
import json
import os

STORE = "train640_up235"
ARM = "upscale235"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--tranche", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    design = json.load(open(a.design))
    runs = design["runs"]
    before = json.dumps(runs, sort_keys=True)
    nextid = max(int(r["run_id"][1:]) for r in runs) + 1

    if any(r.get("arm") == ARM for r in runs):
        print(f"the {ARM} arm is already in the design; nothing to do")
        return

    src = [r for r in runs if r["tranche"] == 1 and r["scheme"] == "flight" and r["seed"] == 1
           and r["rule"] == "random" and r.get("nathan_rule", "random") == "random" and r["tile"] == 640
           and r["n_ren"] in (0, 6) and r["n_nathan"] in (7, 17)]
    src.sort(key=lambda r: (r["n_nathan"], r["n_ren"], r["fold"]))
    assert len(src) == 20, f"expected 20 source runs, found {len(src)}"
    assert len({(r["n_ren"], r["n_nathan"], r["fold"]) for r in src}) == 20, "duplicate source runs"

    upscaled = {s for s, g in design["nathan_group"].items() if g == "older_trees_2025-09-05"}
    assert len(upscaled) == 6, f"expected six older-tree frames, found {len(upscaled)}"

    added = []
    for r in src:
        n = dict(r)
        n.update(run_id=f"r{nextid + len(added):05d}", tranche=a.tranche, arm=ARM,
                 tilestore=STORE, paired_with=r["run_id"])
        added.append(n)

    print(f"{'new':7s} {'source':7s} fold  ren  ohio  upscaled frames drawn")
    for n in added:
        k = sum(1 for f in n["nathan_train"] if f in upscaled)
        print(f"{n['run_id']} {n['paired_with']}  {n['fold']}    {n['n_ren']:3d}  {n['n_nathan']:4d}  {k}")
    assert not ({r["run_id"] for r in runs} & {n["run_id"] for n in added}), "run id collision"

    if a.dry_run:
        print("\ndry run, nothing written")
        return

    design["runs"] = runs + added
    after = json.dumps(design["runs"][:len(json.loads(before))], sort_keys=True)
    assert after == before, "a pre-existing run changed; refusing to write"
    tmp = a.design + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(design, fh, indent=1)
    json.load(open(tmp))
    os.replace(tmp, a.design)
    print(f"\nappended {len(added)} runs ({added[0]['run_id']} to {added[-1]['run_id']}); the design now holds "
          f"{len(design['runs'])} runs and every pre-existing one is byte-identical")


if __name__ == "__main__":
    main()
