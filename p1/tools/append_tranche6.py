"""Append a DENSE Clarks-Run-only sweep to the design, without disturbing anything already running.

WHY IT EXISTS. The first results showed that the Ohio frames saturate at about 1,400 boxes and then
add nothing, while the Clarks Run annotations keep paying: 817 of her boxes beat 7,977 of his, 0.929
recall against 0.685. So the column with NO Ohio data at all is both the best-performing
configuration and the clean measurement of what an hour of her time buys, with nothing of his able to
mask or dilute it. It is also by far the cheapest part of the campaign, 17 minutes a cell against 97
for the median, because her 103 trees make 1,736 tiles where his full set makes 8,494.

The crude curve bends somewhere between 6 and 32 trees and the existing design has four points in
that region. The marginal-value rule lives on the slope there, so this puts every integer from 1 to
24 and then every fourth tree up to 103, at six seeds instead of three.

THE DANGER THIS FILE IS WRITTEN AROUND: RUN IDS ARE POSITIONAL. Regenerating the design would
renumber every run, and the done flags that make a requeue safe are keyed on run id, so two arrays
already running would start skipping the wrong cells and writing results under names belonging to
other cells. So this APPENDS, with ids continuing past the current maximum, and it asserts byte
equality on every pre-existing run before writing anything.

It also refuses to emit a cell that already exists. Duplicate cells would not crash the analysis:
`cellkey` would map both to one cell and one fold's result would silently overwrite the other's,
which is worse than a crash.
"""

import argparse
import json
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--scheme", default="flight")
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--tranche", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    design = json.load(open(a.design))
    before = json.dumps(design["runs"], sort_keys=True)
    existing = {r["run_id"] for r in design["runs"]}
    nextid = max(int(r["run_id"][1:]) for r in design["runs"]) + 1

    ren_boxes = design["ren_boxes"]
    mission = design["mission"]
    sch = design["schemes"][a.scheme]
    max_ren = design["max_ren_level"]

    # every integer through 24, where the curve bends, then every fourth tree to the top
    levels = [n for n in range(1, 25)] + [n for n in range(28, max_ren, 4)] + [max_ren]
    levels = sorted({n for n in levels if n <= max_ren})
    seeds = [int(s) for s in a.seeds.split(",")]

    # a cell is identified by everything except the fold, exactly as the analysis groups them
    def cellkey(r):
        return (r["scheme"], r["seed"], r["rule"], r.get("nathan_rule", "random"),
                r["n_ren"], r["n_nathan"], r["tile"])

    have = {(cellkey(r), r["fold"]) for r in design["runs"]}

    added, skipped = [], 0
    for seed in seeds:
        for fi, fold in enumerate(sch["folds"]):
            # the SAME permutation the existing cells use, so the new levels interleave into the
            # existing nested prefix sequence rather than forming a separate one
            pool = list(fold["ren_pool"])
            random.Random(f"{a.scheme}|{fi}|{seed}").shuffle(pool)
            for nr in levels:
                if nr > len(pool):
                    continue
                k = ((a.scheme, seed, "random", "random", nr, 0, 640), fi)
                if k in have:
                    skipped += 1
                    continue
                rtr = pool[:nr]
                added.append(dict(
                    run_id=f"r{nextid + len(added):05d}", arm="grid_dense", tranche=a.tranche,
                    scheme=a.scheme, fold=fi, seed=seed, rule="random", nathan_rule="random",
                    n_ren=nr, n_nathan=0, tile=640,
                    ren_train=rtr, nathan_train=[],
                    ren_train_boxes=sum(ren_boxes[s] for s in rtr),
                    nathan_train_boxes=0, nathan_groups=[]))
                have.add(k)

    print(f"{len(levels)} levels: {levels}")
    print(f"seeds {seeds}, {len(sch['folds'])} folds, scheme {a.scheme}")
    print(f"cells already in the design and therefore SKIPPED: {skipped}")
    print(f"new cells to append: {len(added)}")
    print(f"run ids {added[0]['run_id']} to {added[-1]['run_id']}" if added else "nothing to add")
    byl = defaultdict(int)
    for r in added:
        byl[r["n_ren"]] += 1
    print(f"replicates per level: {sorted(set(byl.values()))}")
    assert not (existing & {r["run_id"] for r in added}), "run id collision"

    if a.dry_run or not added:
        print("\ndry run, nothing written")
        return

    design["runs"] += added
    design.setdefault("dense_levels", levels)
    # NOTHING pre-existing may change, because two arrays are running against these ids right now
    after = json.dumps(design["runs"][:len(json.loads(before))], sort_keys=True)
    assert after == before, "a pre-existing run changed; refusing to write"
    # ATOMIC, because live tasks open this file every time they start a cell and a partial write
    # would hand one of them invalid JSON. Write beside it, then rename.
    tmp = a.design + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(design, fh, indent=1)
    json.load(open(tmp))                      # refuse to install a file that does not parse
    os.replace(tmp, a.design)
    print(f"\nappended; design now holds {len(design['runs'])} runs "
          f"and every pre-existing one is byte-identical")


if __name__ == "__main__":
    main()
