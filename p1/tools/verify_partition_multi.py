"""Assert a multi-tranche task partition covers every outstanding cell exactly once.

Same job as verify_partition.py but for a partition spanning several tranches, where the tile size
varies and a task that stages one store must contain only cells wanting that store. A partition that
overlaps duplicates GPU work and races two cells on one result file; one that omits leaves holes that
surface only as gaps in a plot weeks later.
"""
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(sys.argv[1])
T = Path(sys.argv[2])
# An EXPECTED-SET file may be given in place of a tranche list. A mop-up submission covers only the
# cells still outstanding, so checking it against every cell of the tranche reports the already
# finished ones as "missing" and refuses a correct partition. That happened on the first resubmission
# and the guard was right to refuse rather than to guess which it was looking at.
args = sys.argv[3:]
expect_file = None
tranches = set()
for x in args:
    if x.isdigit():
        tranches.add(int(x))
    else:
        expect_file = x
idx = json.load(open(T / "INDEX.json"))
design = json.load(open(R / "design.json"))
by = {r["run_id"]: r for r in design["runs"]}

by = {r["run_id"]: r for r in design["runs"]}

placed = []
for i in idx:
    placed += open(T / i["file"]).read().split()
if expect_file:
    want = {L.strip() for L in open(expect_file) if L.strip()}
    tranches = {by[r]["tranche"] for r in want if r in by}
    print("expected set taken from %s: %d cells across tranches %s"
          % (expect_file, len(want), sorted(tranches)))
else:
    want = {r["run_id"] for r in design["runs"] if r["tranche"] in tranches}
dups = [k for k, v in Counter(placed).items() if v > 1]
missing = want - set(placed)
extra = set(placed) - want

print("%d tasks, %d cells placed, %d distinct, design wants %d"
      % (len(idx), len(placed), len(set(placed)), len(want)))
print("  tasks by tile: %s" % dict(Counter(i["tile"] for i in idx)))
print("  tasks by tranche: %s" % dict(Counter(i["tranche"] for i in idx)))
tl = sorted(i["train_tiles"] for i in idx)
print("  task load: min %d median %d max %d (spread %.2fx)"
      % (tl[0], tl[len(tl) // 2], tl[-1], tl[-1] / max(1, tl[0])))

bad = 0
for i in idx:
    for rid in open(T / i["file"]).read().split():
        r = by[rid]
        if (r["tile"], r["scheme"], r["fold"], r["tranche"]) != (
                i["tile"], i["scheme"], i["fold"], i["tranche"]):
            bad += 1
            if bad <= 3:
                print("  MISMATCH task %d stages tile %s %s fold %s tranche %s; %s wants %s %s %s %s"
                      % (i["task"], i["tile"], i["scheme"], i["fold"], i["tranche"], rid,
                         r["tile"], r["scheme"], r["fold"], r["tranche"]))

stores = sorted({i["tile"] for i in idx
                 if not (R / "tilestore" / ("train%d" % i["tile"]) / "MANIFEST.json").exists()})
print("  duplicated: %d   missing: %d   foreign: %d   incoherent: %d   stores absent: %s"
      % (len(dups), len(missing), len(extra), bad, stores))
if missing:
    print("  first missing: %s" % sorted(missing)[:5])

ok = not dups and not missing and not extra and not bad and not stores
print("PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
