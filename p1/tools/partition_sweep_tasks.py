"""Split the sweep's runs into Slurm array tasks, grouped by fold and balanced by cost.

TWO PROPERTIES THE PARTITION HAS TO HAVE, and both come from how the runs differ.

Grouped by FOLD, because a task cuts the whole-frame inference grid for its fold's thirty-odd test
frames before it trains anything, and that work is shared by every run in the task. Mixing folds
inside a task would multiply it.

Balanced by COST, because the cells differ by nearly three orders of magnitude in training tiles,
from seventeen to ten thousand. Chunking a fold's runs in design order would put every cheap cell
in one task and every expensive one in another, so the array's finishing times would spread by
hours and the last task would decide the campaign's wall clock. Dealing cost-sorted runs round
robin makes every task roughly the same size.

Each task gets a DISJOINT run list, so nothing coordinates between tasks. Within a task the
workers claim cells from a directory; across tasks the done flags in the shared results directory
are what makes a resubmission skip finished work, which is also what makes a preempted task safe
to requeue.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tranche", type=int, action="append", required=True)
    ap.add_argument("--per-task", type=int, default=24)
    ap.add_argument("--max-task-tiles", type=int, default=0,
                    help="cap a task's total training tiles, which is what actually sets its wall "
                         "clock. A fixed cell count cannot do this: tranche 3's cells are ALL at "
                         "full Ohio data so 24 of them come to 180,000 tiles and about 18 hours, "
                         "while tranche 5's come to 6,000 and twenty minutes, a 28-fold spread "
                         "between tasks that the within-group balancing cannot touch because a "
                         "task may not span folds. 0 disables the cap.")
    ap.add_argument("--only-runs", default=None,
                    help="comma separated run ids, OR the path to a file of run ids one per line. "
                         "The file form is what a hand-off actually uses, since an outstanding list "
                         "runs to a thousand ids; passing the path to the comma form used to make "
                         "the path itself look like a run id, which the design check below caught "
                         "but only after the cut had been attempted.")
    # THE SECONDS-BASED CUT, added for HTCondor, where a task is lost WHOLE when a backfill machine
    # is reclaimed. Tiles are a proxy for cost; seconds are the cost, and the two stop agreeing once
    # a task holds one very large cell, because a task can never finish faster than its longest
    # single cell however many workers share the card.
    ap.add_argument("--max-task-seconds", type=int, default=0,
                    help="cap a task's PREDICTED WALL CLOCK in seconds, computed at --concurrent "
                         "workers from the cost law fitted on 452 finished cells. 0 disables it and "
                         "leaves the tile cap in charge.")
    ap.add_argument("--concurrent", type=int, default=6,
                    help="workers a task will run, which sets how its cell seconds turn into wall "
                         "seconds. Must match what the submission actually passes to the pool.")
    ap.add_argument("--setup-seconds", type=int, default=120,
                    help="fixed per-task cost before any cell starts: copy and unpack the store and "
                         "the frames, then cut the inference grid. MEASURED at 55 s on a CHTC GPU "
                         "node and about 115 s on a VT node that stages from shared scratch.")
    ap.add_argument("--epochs-assumed", type=int, default=200,
                    help="epochs to assume when CLASSIFYING a task's length. Defaults to the 200 "
                         "cap rather than the p90 of 130, because an under-sized wall clock loses "
                         "the whole task while an over-sized one only wastes a reservation.")
    ap.add_argument("--slowest-card-factor", type=float, default=1.0,
                    help="divide the wall target by this before packing, to leave margin for the "
                         "SLOWEST machine a task can land on. Packing to the fitted average "
                         "guarantees that the slow half of a heterogeneous pool overruns: the first "
                         "cut left this at 1.0, so it filled every bucket to exactly the 8 h target "
                         "and 59 of 75 short tasks could have overrun on the slow card. MEASURED "
                         "over 480 CHTC cells at 6 workers, seconds per epoch divided by this "
                         "law, by tile band (under 300 / 300-1,200 / 1,200-3,000 / over 3,000): "
                         "A100 40 GB 1.29 / 1.44 / - / 1.26; A100 80 GB - / - / 1.05 / 1.15; "
                         "H100 80 GB - / 0.94 / - / -; H200 0.86 / 0.85 / 0.79 / 0.79; "
                         "L40S 0.78 / 0.91 / 0.84 / 0.86. So the law is fitted to an A100-80 at 5 "
                         "workers and the pool spans 0.78 to 1.44 of it. USE 1.3 for a mixed CHTC "
                         "pool, which is the A100 40 GB's large-cell figure; use 1.0 only when you "
                         "choose the machine. Note the H200 and L40S run 15 to 21 percent FASTER "
                         "than the law, so a target set for the slow card is generous on most cards, "
                         "which is the correct direction for a wall clock.")
    ap.add_argument("--short-max-seconds", type=int, default=28800,
                    help="predicted wall at or under this goes in a short job (CHTC short is a 12 h "
                         "ceiling, so the default 8 h leaves a third of it as margin); above it the "
                         "task is marked medium.")
    # A TENSION WITH NO FREE ANSWER, found 2026-10-04 and pointing the OPPOSITE WAY from the
    # utilization fix above. Filling a bucket up to its straggler's length is right when a task runs
    # to completion: it keeps all C workers busy and took tranche 1 from 914 tasks to 114. It is
    # WRONG when the task can be PREEMPTED, because packing the long cells together makes every one
    # of them in flight at the same time, so the task's wall clock IS its longest cell and all C
    # cells finish within minutes of each other at the end. MEASURED on the 39 medium tasks: cells
    # run 4.0 h median on an H200 and 6.4 h on an A100 40 GB, only 14 percent of each task's cells
    # finish inside 2 hours, and those are 2 percent of the hours, so an interruption shorter than
    # about six hours banks 2 percent of the tranche's work however often the task resumes.
    #
    # So the packing target depends on whether the executor can take the machine back. Where it can,
    # size a task so its WHOLE duration fits comfortably inside the observed interruption window, and
    # accept worse utilization; where it cannot, pack to the straggler as this code does by default.
    # There is no setting that satisfies both, and the measurement that decides it is the window
    # length on the preemptible machines, not anything about the cells.
    ap.add_argument("--bigmem-tiles", type=int, default=5400,
                    help="a task holding any cell above this many training tiles is marked "
                         "`bigmem` in the index, so the scheduler can place it on a card large "
                         "enough to hold --concurrent cells of that size. The default of 5,400 "
                         "sits just above the largest cell measured to fit: six concurrent cells "
                         "of up to 5,346 tiles used 45.6 to 46.0 GB of a 45 GB L40S, so the real "
                         "per-cell need is about 7,700 MiB against the 6,350 MiB the pool budgets.")
    ap.add_argument("--resumable-max-seconds", type=int, default=7200,
                    help="predicted wall at or under this is marked resumable, which opens CHTC's "
                         "238 backfill GPUs at the price of being interruptible. Short tasks only, "
                         "so that an eviction loses at most this much work.")
    a = ap.parse_args()

    design = json.load(open(a.design))
    man = json.load(open(a.manifest))["frames"]
    tiles = {s: len(v["tiles"]) for s, v in man.items()}

    # MEASURED store ratios, not the inverse-square law. The grid of candidate windows does scale as
    # the inverse square of the tile edge, but the tiler keeps only windows CONTAINING a bur plus
    # background at a fixed ratio, and a larger tile makes a larger fraction of candidates positive
    # while also merging nearby burs into one tile. Measured store sizes are 15,161 at 448, 10,643 at
    # 640, 6,644 at 960 and 4,404 at 1280, so the real ratios are 1.42, 1.00, 0.62 and 0.41 against
    # the inverse-square law's 2.04, 1.00, 0.44 and 0.25. Sizing a walltime from the law would be
    # wrong by about 1.4x in both directions.
    STORE = {448: 15161, 640: 10643, 960: 6644, 1280: 4404}

    def cost(r):
        n = sum(tiles[s] for s in r["ren_train"] + r["nathan_train"])
        return int(round(n * STORE[r["tile"]] / STORE[640]))

    # THE COST LAW, FITTED ON 452 FINISHED CELLS on A100-SXM4-80GB, 2026-10-03:
    #     seconds per epoch = 0.025418 * training_tiles + 3.422     r2 0.9685, residual sd 10.6 s
    # The structure is mechanical rather than chosen: a cell trains for `epochs_run` epochs and each
    # epoch touches every training tile once, so the per-tile term is the data cost and the constant
    # is the per-epoch fixed cost (the validation pass and the loader warm-up, which do not scale
    # with the training set). Fitting seconds against tiles directly would confound the two and be
    # wrong at both ends of a range that spans twelve tiles to 8,630.
    #
    # Epochs are NOT known before a cell runs. Measured: median 102, p75 116, p90 130, p95 143, and
    # 4 of 452 reached the 200 cap. At 130 the law covers 408 of 452 finished cells.
    #
    # AND THE FIGURE THIS REPLACES WAS WRONG BY EIGHT TIMES. "About 17 minutes a cell" was quoted
    # for days; it is the cost of the two SMALLEST cells in the design. The measured mean over 452
    # cells is 2.27 HOURS, and the 1,025 GPU-hours they consumed is the number to reason from.
    SEC_PER_TILE_EPOCH, SEC_PER_EPOCH, CELL_OVERHEAD = 0.025418, 3.422, 198.0

    def cell_seconds(r, epochs):
        return epochs * (SEC_PER_TILE_EPOCH * cost(r) + SEC_PER_EPOCH) + CELL_OVERHEAD

    def task_wall(group, epochs):
        """Predicted wall clock for a set of cells sharing one card.

        A task with C workers cannot finish before max(longest single cell, total / C): the first
        term is the straggler and the second is the throughput bound. Taking only the second is the
        mistake the tile cap makes, and it is wrong exactly where it matters, on a task holding one
        8,600 tile cell that no amount of concurrency can shorten.
        """
        if not group:
            return 0.0
        secs = [cell_seconds(r, epochs) for r in group]
        return a.setup_seconds + max(max(secs), sum(secs) / max(1, a.concurrent))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("task_*.txt"):
        f.unlink()

    runs = [r for r in design["runs"] if r["tranche"] in set(a.tranche)]
    if a.only_runs:
        if Path(a.only_runs).is_file():
            want = set(Path(a.only_runs).read_text().split())
            print(f"--only-runs read {len(want)} run ids from {a.only_runs}")
        else:
            want = set(x for x in a.only_runs.split(",") if x)
        runs = [r for r in design["runs"] if r["run_id"] in want]
        missing = want - {r["run_id"] for r in runs}
        assert not missing, ("--only-runs names %d ids that are not in this design, for example %s. "
                             "A published list and a design that disagree is the failure that makes "
                             "run ids look valid while naming nothing."
                             % (len(missing), sorted(missing)[:5]))
        # The restriction must not silently reach outside the tranches asked for, which it would if
        # an outstanding list from one hand-off were passed to the cut for another.
        off = sorted({r["tranche"] for r in runs} - set(a.tranche))
        assert not off, ("--only-runs names cells from tranche(s) %s, which --tranche does not "
                         "list. Cutting them here would claim another hand-off's cells." % off)
    runs_by_id = {r["run_id"]: r for r in runs}

    # THE TILE IS PART OF THE GROUPING KEY, and leaving it out was a fatal defect. A task stages
    # exactly one `train<tile>` store and cuts exactly one inference grid at one tile size, so a
    # task holding runs at 448, 960 and 1280 can serve at most one of them. Without the tile in the
    # key, all five tile-size tasks held three sizes each and every one of those 45 runs would have
    # died at `open(train448/MANIFEST.json)`.
    groups = {}
    for r in runs:
        groups.setdefault((r["tranche"], r["scheme"], r["fold"], r["tile"]), []).append(r)

    index, n = [], 0
    for key in sorted(groups):
        g = sorted(groups[key], key=cost, reverse=True)
        if a.max_task_seconds:
            # PACK TO A WALL-CLOCK TARGET, NOT TO A FIXED TASK COUNT. Choosing a task count and then
            # dealing cells into it round robin is what the tile cap did, and on this tranche it was
            # badly wrong: a 4,000 tile cap produced 914 tasks holding ONE OR TWO cells each, so
            # four of every six worker slots sat idle and each task held a whole GPU to run two
            # cells. Utilization, not task count, is what a cap should control.
            #
            # A bucket accepts another cell while its predicted wall, max(longest cell, total over
            # C workers), stays under the target. That fills a task up to the length of its own
            # straggler, which is exactly the slack a long cell creates: a task carrying one 12 hour
            # cell can carry roughly C times that much work before the throughput bound overtakes
            # the straggler bound, and all of it is free in wall-clock terms.
            target = float(max(a.setup_seconds + 1, a.max_task_seconds))
            buckets, loads, longests = [], [], []
            for r in g:                              # g is already sorted longest cell first
                s = cell_seconds(r, a.epochs_assumed)
                placed = False
                # First fit over existing buckets, preferring the fullest that still has room, so
                # tasks finish full rather than leaving a tail of half-empty ones.
                order = sorted(range(len(buckets)), key=lambda j: -loads[j])
                for j in order:
                    lo = max(longests[j], s)
                    tot = loads[j] + s
                    # A BUCKET'S OWN STRAGGLER RAISES ITS CEILING, and comparing against the global
                    # target alone is what left 220 tasks holding a single 12 hour cell apiece, each
                    # occupying a whole GPU to run one cell while five of six worker slots idled. A
                    # task can never finish before its longest cell, so once that cell exceeds the
                    # target the task is already that long and every further cell added up to C
                    # times that length is FREE in wall-clock terms. The ceiling is therefore the
                    # larger of the target and what this bucket is already committed to.
                    ceiling = max(target, a.setup_seconds + lo)
                    if a.setup_seconds + max(lo, tot / max(1, a.concurrent)) <= ceiling:
                        buckets[j].append(r)
                        loads[j] = tot
                        longests[j] = lo
                        placed = True
                        break
                if not placed:
                    # A cell whose own time already exceeds the target cannot be split, so it opens
                    # a task and that task is declared over the cap rather than hidden.
                    buckets.append([r])
                    loads.append(s)
                    longests.append(s)
            if a.per_task:                           # still honor a hard cell-count ceiling
                split = []
                for b in buckets:
                    for k in range(0, len(b), a.per_task):
                        split.append(b[k:k + a.per_task])
                buckets = split
        else:
            ntask = max(1, -(-len(g) // a.per_task))
            if a.max_task_tiles:
                ntask = max(ntask, -(-sum(cost(r) for r in g) // a.max_task_tiles))
            buckets = [[] for _ in range(ntask)]
            load = [0.0] * ntask
            for r in g:                              # longest first into the lightest bucket
                i = min(range(ntask), key=lambda j: load[j])
                buckets[i].append(r)
                load[i] += cell_seconds(r, a.epochs_assumed)
        tr, scheme, fold, tsize = key
        for b in buckets:
            n += 1
            p = out / f"task_{n:04d}.txt"
            p.write_text("\n".join(r["run_id"] for r in b) + "\n")
            lo = sum(cost(r) for r in b)
            wall = task_wall(b, a.epochs_assumed)
            longest = max((cell_seconds(r, a.epochs_assumed) for r in b), default=0.0)
            # LENGTH AND RESUMABLE, so an HTCondor submit file needs no second pass and no judgement
            # of its own. Classified at the 200 epoch cap: a task that overruns a short job's ceiling
            # is lost whole, so the honest basis is the worst case rather than the p90.
            length = "short" if wall <= a.short_max_seconds else "medium"
            resumable = bool(length == "short" and wall <= a.resumable_max_seconds)
            # the tile goes into INDEX.json so the sbatch reads it from here rather than from the
            # environment, which is what made a mixed-tile task possible in the first place
            # DOES THIS TASK NEED A BIG CARD? Measured on CHTC from HTCondor's own GPU accounting,
            # six concurrent cells of up to 5,346 training tiles used 45.6 to 46.0 GB, which is the
            # whole of a 45 GB L40S, so the per-cell need is about 7,700 MiB rather than the
            # 6,350 MiB the pool budgets. A task holding a cell above that measured point cannot be
            # run at this concurrency on a 48 GB card, and the failure is SILENT: Ultralytics
            # catches the out-of-memory inside its label assignment and finishes on the CPU in a
            # different floating-point order, which contaminated 11 cells before the poison-line
            # gate existed. So the task is marked here and the scheduler is told to place it, rather
            # than left to discover the limit at run time.
            biggest = max((cost(r) for r in b), default=0)
            bigmem = bool(biggest > a.bigmem_tiles)
            index.append(dict(task=n, file=p.name, tranche=tr, scheme=scheme, fold=fold,
                              tile=tsize, n_runs=len(b), train_tiles=lo,
                              length=length, resumable=resumable, bigmem=bigmem,
                              largest_cell_tiles=int(biggest),
                              longest_cell_seconds=int(round(longest)),
                              predicted_wall_seconds=int(round(wall)),
                              epochs_assumed=a.epochs_assumed, concurrent=a.concurrent,
                              bigmem_tiles=a.bigmem_tiles,
                              test_frames=design["schemes"][scheme]["folds"][fold]["test"]))
            print(f"task {n:4d}  tranche {tr} {scheme} fold {fold} tile {tsize}  "
                  f"{len(b):3d} runs  {lo:7d} training tiles  "
                  f"wall {wall / 3600:5.2f} h  longest cell {longest / 3600:5.2f} h  "
                  f"largest {biggest:5d} tiles  "
                  f"{length}{' resumable' if resumable else ''}{' BIGMEM' if bigmem else ''}")

    (out / "INDEX.json").write_text(json.dumps(index, indent=1))
    # A one-line-per-task plain file, because an HTCondor submit file reads a table far more easily
    # than it reads JSON and ap2002 has no scientific Python. Columns: task, fold, length,
    # resumable, bigmem. `bigmem` is APPENDED rather than inserted, so a reader that splits on
    # whitespace and takes the first four fields keeps working on the tranche 6 and tranche 3 and 5
    # index files already in use.
    (out / "INDEX.txt").write_text(
        "".join("%d %d %s %s %s\n" % (i["task"], i["fold"], i["length"],
                                      "true" if i["resumable"] else "false",
                                      "true" if i["bigmem"] else "false") for i in index))
    # The frames each task cuts an inference grid for: its fold's test trees AND its fold's
    # validation trees. The validation trees are never trained on, so scoring them costs 58 percent
    # more inference and buys a second out-of-sample set for diagnosis. The operating point itself
    # is chosen leave-one-fold-out across test folds in the analysis, which needs no reserved trees
    # at all and leaves the top of the learning curve where it is.
    for scheme, s in design["schemes"].items():
        for fi, f in enumerate(s["folds"]):
            (out / f"testframes_{scheme}_{fi}.txt").write_text(
                "\n".join(f["test"] + f["val_ren"]) + "\n")

    print(f"\n{len(runs)} runs -> {n} tasks  (array 1-{n})")
    print(f"total training tiles across tasks: {sum(i['train_tiles'] for i in index):,}")

    # EVERY CELL EXACTLY ONCE, asserted here rather than left to the separate verifier, because a
    # partition that silently drops or duplicates a cell is the one defect that cannot be caught by
    # looking at the submission afterwards.
    assigned = [rid for i in index for rid in (out / i["file"]).read_text().split()]
    want = [r["run_id"] for r in runs]
    assert len(assigned) == len(set(assigned)), "a cell was assigned to more than one task"
    assert set(assigned) == set(want), (
        "partition does not cover the input: %d assigned against %d wanted, %d missing, %d extra"
        % (len(assigned), len(want), len(set(want) - set(assigned)), len(set(assigned) - set(want))))
    print(f"every one of {len(want):,} cells appears in exactly one task")

    walls = sorted(i["predicted_wall_seconds"] for i in index)
    print("predicted wall clock at %d workers and %d epochs: min %.2f h, median %.2f h, max %.2f h"
          % (a.concurrent, a.epochs_assumed, walls[0] / 3600,
             walls[len(walls) // 2] / 3600, walls[-1] / 3600))
    print("  spread between the lightest and heaviest task: %.1fx" % (walls[-1] / max(1, walls[0])))
    # CELL WORK IS THE SUM OF THE CELLS, NOT THE WALL TIMES THE WORKER COUNT. The first version of
    # this line multiplied a task's wall clock by `--concurrent`, which counts IDLE worker slots as
    # work and overstated a single-cell task by a factor of six. Card-hours and cell-hours are
    # different quantities, and the ratio between them is the utilization this cut exists to fix.
    cellsec = {}
    for i in index:
        cellsec[i["task"]] = sum(cell_seconds(runs_by_id[rid], a.epochs_assumed)
                                 for rid in (out / i["file"]).read_text().split())
    for lab, sel in (("short and resumable", lambda i: i["resumable"]),
                     ("short, not resumable", lambda i: i["length"] == "short" and not i["resumable"]),
                     ("medium", lambda i: i["length"] == "medium")):
        s = [i for i in index if sel(i)]
        if s:
            cells_h = sum(cellsec[i["task"]] for i in s) / 3600
            cards_h = sum(i["predicted_wall_seconds"] for i in s) / 3600
            print("  %-22s %4d tasks, %5d cells, %7.0f cell-hours on %7.0f card-hours (%.0f%% of %d slots)"
                  % (lab, len(s), sum(i["n_runs"] for i in s), cells_h, cards_h,
                     100 * cells_h / max(1e-9, cards_h * a.concurrent), a.concurrent))
    tot_cells_h = sum(cellsec.values()) / 3600
    tot_cards_h = sum(i["predicted_wall_seconds"] for i in index) / 3600
    print("  %-22s %4d tasks, %5d cells, %7.0f cell-hours on %7.0f card-hours (%.0f%% of %d slots)"
          % ("ALL", len(index), sum(i["n_runs"] for i in index), tot_cells_h, tot_cards_h,
             100 * tot_cells_h / max(1e-9, tot_cards_h * a.concurrent), a.concurrent))
    over = [i for i in index if a.max_task_seconds and i["predicted_wall_seconds"] > a.max_task_seconds]
    if over:
        print("  %d task(s) EXCEED the cap because a single cell does; they cannot be split further:"
              % len(over))
        for i in over[:8]:
            print("    task %d: %d cells, longest cell %.2f h, wall %.2f h, marked %s"
                  % (i["task"], i["n_runs"], i["longest_cell_seconds"] / 3600,
                     i["predicted_wall_seconds"] / 3600, i["length"]))


if __name__ == "__main__":
    main()
