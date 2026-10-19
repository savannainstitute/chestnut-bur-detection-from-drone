"""Freeze the whole learning-curve sweep as one design file, before any job runs.

The question the sweep answers is what the next hour of annotation buys and when it stops being
worth buying, so the design has to make a marginal comparison possible. Three choices do that work
and each is deliberate.

NESTED SUBSAMPLING. Within one replicate, the trees used at level k are a strict subset of those
used at level k+1, taken as a prefix of one permutation. A curve built from independent draws at
each level mixes the marginal value of data with the luck of which trees were drawn, and at these
sample sizes the luck dominates. A prefix curve measures the value of the trees actually added.

TWO FOLD SCHEMES, RUN SIDE BY SIDE. `flight` folds put whole drone missions on one side of the
split, so a test tree shares no flight, illumination or hour with any training tree. `tree` folds
split trees at random, which is the matched estimand for scoring more trees in flights already
collected. Neither is the right answer alone, and the GAP between them measures how much apparent
accuracy comes from within-flight similarity, which is a number nobody has.

A FIXED VALIDATION SET PER FOLD. Early stopping needs a validation split, and drawing it from each
run's own training subset would make it vanish at the small levels and would change what "converged"
means along the curve. So each fold reserves its own validation trees and frames once; every level
and seed in that fold stops against the same set, and no test tree is ever in it.

THE TWO SOURCES ARE DIFFERENT REGIMES, NOT MORE AND LESS OF ONE THING, and the design has to treat
them that way. Measured: a Clarks Run frame holds a median of 6 burs and at most 45, one young tree
in a 9.1 by 6.1 meter footprint. An Ohio frame holds a median of 66 and at most 1,560, and the six
`older_trees` frames average 1,211 burs each in canopy dense enough that the median gap between
neighboring burs is 0.97 bur widths. Those six frames carry 76 percent of the Ohio boxes.

So a random prefix of two Ohio frames is a lottery over regimes rather than a quantity: it can
deliver 2,400 boxes of touching canopy or 130 boxes of isolated burs, a factor of fifteen. The main
grid keeps the random order anyway, because the informative axis is the realized box count and
fifteen replicates sample it densely. What the random order cannot answer is WHICH regime carries
the transfer, so a separate arm selects Ohio frames by flight group: all dense, all sparse, or
stratified across his four groups at a matched frame count.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

REN_LEVELS = [0, 1, 3, 6, 10, 15, 22, 32, 45, 63, 78, 92, 103]
NATHAN_LEVELS = [0, 2, 4, 7, 11, 17, 25, 36]
N_VAL_REN = 18
N_VAL_NATHAN = 7
SEEDS = [1, 2, 3]
N_FOLDS = 5


def crown_boxes(path, keep=(0, 2, 4)):
    n = 0
    for line in open(path):
        f = line.split()
        if f and int(f[0]) in keep:
            n += 1
    return n


def stratified_folds(items, key, n_folds, rng):
    """Deal items into folds after sorting by key, so every fold spans the key's range."""
    order = sorted(items, key=lambda s: (key[s], s))
    rng.shuffle_in_strata = None
    folds = [[] for _ in range(n_folds)]
    for i, s in enumerate(order):
        folds[i % n_folds].append(s)
    return folds


def flight_folds(trees, mission, count, n_folds):
    """Bin-pack whole missions into folds, largest mission into the emptiest fold."""
    by_mission = defaultdict(list)
    for t in trees:
        by_mission[mission[t]].append(t)
    folds = [[] for _ in range(n_folds)]
    for m in sorted(by_mission, key=lambda m: (-len(by_mission[m]), m)):
        folds.sort(key=len)
        folds[0].extend(sorted(by_mission[m]))
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ren-labels", required=True)
    ap.add_argument("--nathan-labels", required=True)
    ap.add_argument("--frame-index", required=True,
                    help="dir tree holding the Clarks Run frames, used to read the mission folder")
    ap.add_argument("--manifest", required=True, help="tile store MANIFEST.json")
    ap.add_argument("--nathan-manifest", required=True,
                    help="Nathan's manifest.csv, for the flight group of each Ohio frame")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    man = json.load(open(a.manifest))["frames"]
    ren = sorted(s for s, v in man.items() if v["source"] == "ren")
    nat = sorted(s for s, v in man.items() if v["source"] == "nathan")

    ren_boxes = {s: crown_boxes(os.path.join(a.ren_labels, s + ".txt")) for s in ren}
    nat_boxes = {s: crown_boxes(os.path.join(a.nathan_labels, s + ".txt"), keep=(0,)) for s in nat}

    import csv as _csv
    nat_group = {r["stem"]: r["flight_group"] for r in _csv.DictReader(open(a.nathan_manifest))}
    # DENSE is the regime the owner flagged: huge trees, overlapping canopies, hundreds of burs a
    # frame. Measured, the split is clean at 300 boxes a frame and does not need a judgment call.
    nat_dense = sorted(s for s in nat if nat_boxes[s] >= 300)
    nat_sparse = sorted(s for s in nat if nat_boxes[s] < 300)

    # the mission is the second-to-last path component of the frame on disk, e.g. .../M08/<flight>/
    index = {}
    for dp, _, fn in os.walk(os.path.expanduser(a.frame_index)):
        for f in fn:
            if f.endswith(".JPG"):
                index.setdefault(f, dp)
    mission = {}
    for s in ren:
        d = index[s.split("__")[1] + ".JPG"].split(os.sep)
        mission[s] = d[-2]

    design = dict(
        ren_levels=REN_LEVELS, nathan_levels=NATHAN_LEVELS, seeds=SEEDS, n_folds=N_FOLDS,
        n_val_ren=N_VAL_REN, n_val_nathan=N_VAL_NATHAN,
        ren_boxes=ren_boxes, nathan_boxes=nat_boxes, mission=mission,
        nathan_group=nat_group, nathan_dense=nat_dense, nathan_sparse=nat_sparse,
        schemes={}, runs=[],
    )

    for scheme in ("flight", "tree"):
        rng = random.Random(1000 + hash(scheme) % 1000)
        if scheme == "flight":
            folds = flight_folds(ren, mission, ren_boxes, N_FOLDS)
        else:
            folds = stratified_folds(ren, ren_boxes, N_FOLDS, random.Random(77))

        # the Nathan validation frames are fixed across everything: his set is not being split
        nv_rng = random.Random(9)
        nat_sorted = sorted(nat, key=lambda s: (nat_boxes[s], s))
        val_nat = sorted(nat_sorted[i] for i in
                         sorted(nv_rng.sample(range(len(nat_sorted)), N_VAL_NATHAN)))
        train_nat_pool = [s for s in nat if s not in set(val_nat)]

        sch = dict(folds=[], val_nathan=val_nat, nathan_pool=train_nat_pool)
        for fi, test in enumerate(folds):
            rest = [s for s in ren if s not in set(test)]
            rest_sorted = sorted(rest, key=lambda s: (ren_boxes[s], s))
            vr = random.Random(500 + fi)
            # take every k-th tree along the bur-count order, so validation spans the load range
            step = len(rest_sorted) / N_VAL_REN
            val = sorted({rest_sorted[min(len(rest_sorted) - 1, int(i * step + vr.random() * step))]
                          for i in range(N_VAL_REN)})
            pool = [s for s in rest if s not in set(val)]
            sch["folds"].append(dict(test=sorted(test), val_ren=val, ren_pool=sorted(pool),
                                     test_boxes=sum(ren_boxes[s] for s in test),
                                     missions=sorted({mission[s] for s in test})))
        design["schemes"][scheme] = sch

    # ---- the run table ----------------------------------------------------------------
    rid = 0
    max_ren = min(len(f["ren_pool"]) for s in design["schemes"].values() for f in s["folds"])
    ren_levels = [n for n in REN_LEVELS if n <= max_ren]
    if ren_levels[-1] != max_ren:
        ren_levels.append(max_ren)
    design["ren_levels_used"] = ren_levels
    design["max_ren_level"] = max_ren

    def round_robin(pool, group_of):
        """Deal a pool out one group at a time, so a prefix spans the groups evenly."""
        by = defaultdict(list)
        for s in pool:
            by[group_of[s]].append(s)
        keys = sorted(by, key=lambda m: (-len(by[m]), m))
        out = []
        while any(by[k] for k in keys):
            for k in keys:
                if by[k]:
                    out.append(by[k].pop(0))
        return out

    def orders(scheme, fi, seed, rule, nat_rule):
        """Return the (ren, nathan) selection orders. Prefixes of these are the training sets."""
        f = design["schemes"][scheme]["folds"][fi]
        pool = list(f["ren_pool"])
        r = random.Random(f"{scheme}|{fi}|{seed}")
        r.shuffle(pool)
        if rule == "heavy":
            pool.sort(key=lambda s: -ren_boxes[s])          # stable, so ties keep random order
        elif rule == "spread":
            pool = round_robin(pool, mission)

        npool = list(design["schemes"][scheme]["nathan_pool"])
        random.Random(f"nat|{seed}").shuffle(npool)
        if nat_rule == "stratified":
            npool = round_robin(npool, nat_group)
        elif nat_rule == "dense":
            npool = [s for s in npool if s in set(nat_dense)]
        elif nat_rule == "sparse":
            npool = [s for s in npool if s in set(nat_sparse)]
        return pool, npool

    def add(arm, tranche, scheme, fi, seed, rule, n_ren, n_nat, tile, nat_rule="random"):
        nonlocal rid
        pool, npool = orders(scheme, fi, seed, rule, nat_rule)
        if n_nat > len(npool):
            return                       # a rule with too few frames simply has no cell there
        rtr, ntr = pool[:n_ren], npool[:n_nat]
        design["runs"].append(dict(
            run_id=f"r{rid:05d}", arm=arm, tranche=tranche, scheme=scheme, fold=fi, seed=seed,
            rule=rule, nathan_rule=nat_rule, n_ren=n_ren, n_nathan=n_nat, tile=tile,
            ren_train=rtr, nathan_train=ntr,
            ren_train_boxes=sum(ren_boxes[s] for s in rtr),
            nathan_train_boxes=sum(nat_boxes[s] for s in ntr),
            nathan_groups=sorted({nat_group[s] for s in ntr})))
        rid += 1

    for tranche, scheme in ((1, "flight"), (2, "tree")):
        for fi in range(N_FOLDS):
            for seed in SEEDS:
                for nr in ren_levels:
                    for nn in NATHAN_LEVELS:
                        if nr == 0 and nn == 0:
                            continue
                        add("grid", tranche, scheme, fi, seed, "random", nr, nn, 640)

    for fi in range(N_FOLDS):                                    # composition
        for seed in SEEDS:
            for rule in ("heavy", "spread"):
                for nr in [n for n in ren_levels if n > 0]:
                    add("composition", 3, "flight", fi, seed, rule, nr, max(NATHAN_LEVELS), 640)

    for fi in range(N_FOLDS):                                    # tile size
        for seed in SEEDS:
            for tile in (448, 960, 1280):
                add("tilesize", 4, "flight", fi, seed, "random",
                    max_ren, max(NATHAN_LEVELS), tile)

    # Which Ohio regime carries the transfer: the dense mature canopy the owner flagged, or the
    # sparse frames? MATCHED ON FRAMES, not boxes, and that is forced rather than chosen: the six
    # dense frames in the pool hold 6,280 boxes against 1,697 in all thirty sparse ones, so no
    # box-matched comparison exists. Frames are also what a pilot actually decides to fly, so the
    # comparison that exists is the one that answers a collection question. Levels run past six so
    # "all the sparse frames there are" is a cell; a rule with too few frames skips its upper ones.
    for fi in range(N_FOLDS):
        for seed in SEEDS:
            for nat_rule in ("dense", "sparse", "stratified"):
                for nn in (2, 4, 6, 11, 20, 30):
                    for nr in (0, max_ren):
                        add("ohio_regime", 5, "flight", fi, seed, "random",
                            nr, nn, 640, nat_rule=nat_rule)

    Path(a.out).write_text(json.dumps(design, indent=1))

    print(f"{len(ren)} Clarks Run trees ({sum(ren_boxes.values())} crown burs), "
          f"{len(nat)} Ohio frames ({sum(nat_boxes.values())} boxes)")
    for scheme, s in design["schemes"].items():
        print(f"\n{scheme} folds:")
        for i, f in enumerate(s["folds"]):
            print(f"  fold {i}: test {len(f['test']):3d} trees / {f['test_boxes']:4d} burs, "
                  f"val {len(f['val_ren'])}, pool {len(f['ren_pool'])}, "
                  f"missions {','.join(f['missions'])}")
    print(f"\nren levels {ren_levels}\nnathan levels {NATHAN_LEVELS}")
    from collections import Counter
    for t, n in sorted(Counter(r["tranche"] for r in design["runs"]).items()):
        print(f"tranche {t}: {n} runs")
    print(f"TOTAL {len(design['runs'])} runs -> {a.out}")


if __name__ == "__main__":
    main()
