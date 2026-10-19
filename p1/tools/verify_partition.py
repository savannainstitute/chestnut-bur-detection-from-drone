"""Assert a tranche's task partition covers every cell exactly once, and that each task is coherent.

Run before letting an array go: a partition that overlaps duplicates GPU work and races two cells on
one result file, and a partition that omits leaves holes that only show up as gaps in a plot later.
"""
import json
import sys
from collections import Counter
from pathlib import Path

R, tranche = Path(sys.argv[1]), int(sys.argv[2])
T = R / "tasks" / f"t{tranche}"
idx = json.load(open(T / "INDEX.json"))
design = json.load(open(R / "design.json"))

print(f"{len(idx)} tasks")
tl = sorted(i["train_tiles"] for i in idx)
print(f"  tiles per task: min {tl[0]:,} median {tl[len(tl)//2]:,} max {tl[-1]:,} "
      f"(spread {tl[-1]/max(1,tl[0]):.2f}x)")
for k in ("tile", "scheme", "fold", "n_runs"):
    print(f"  {k}: {dict(Counter(i[k] for i in idx))}")

ids = []
for i in idx:
    ids += open(T / i["file"]).read().split()
want = {r["run_id"] for r in design["runs"] if r["tranche"] == tranche}
dups = [k for k, v in Counter(ids).items() if v > 1]
print(f"\n  run ids placed: {len(ids)}, distinct {len(set(ids))}, design wants {len(want)}")
print(f"  DUPLICATED across tasks: {len(dups)}  {dups[:5]}")
print(f"  MISSING from every task: {len(want - set(ids))}  {sorted(want - set(ids))[:5]}")
print(f"  placed but not in the tranche: {len(set(ids) - want)}")

# every run in a task must agree with that task's staged tile, scheme and fold
by = {r["run_id"]: r for r in design["runs"]}
bad = 0
for i in idx:
    for rid in open(T / i["file"]).read().split():
        r = by[rid]
        if (r["tile"], r["scheme"], r["fold"]) != (i["tile"], i["scheme"], i["fold"]):
            bad += 1
            if bad <= 3:
                print(f"  MISMATCH task {i['task']} stages tile {i['tile']} {i['scheme']} "
                      f"fold {i['fold']} but {rid} wants {r['tile']} {r['scheme']} {r['fold']}")
print(f"  cells whose tile/scheme/fold disagrees with their task: {bad}")

# and the store each task needs must exist
missing = sorted({i["tile"] for i in idx if not (R / "tilestore" / f"train{i['tile']}" /
                                                 "MANIFEST.json").exists()})
print(f"  tile sizes with NO store built: {missing}")

ok = not dups and not (want - set(ids)) and not (set(ids) - want) and not bad and not missing
print("\n" + ("PASSED: the partition covers every cell exactly once and every store exists"
              if ok else "FAILED: see above"))
sys.exit(0 if ok else 1)
