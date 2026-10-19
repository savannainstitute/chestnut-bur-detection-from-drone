"""Run several sweep cells concurrently on ONE allocated GPU, pulling from a work list.

Why pack rather than ask for one GPU per cell: yolov8n-p2 at 640 with a batch of 16 uses a few
gigabytes and does not saturate an A30, let alone an A100, so a GPU-per-cell campaign would ask the
scheduler for thousands of slots on partitions that are currently fully allocated and would get
them slowly. Packing turns the same science into a few dozen allocations.

Why a claim directory rather than a fixed split of the work: the cells differ in cost by more than
an order of magnitude, since one trains on seventeen tiles and another on ten thousand. A fixed
split would leave most workers idle behind the slowest one. `mkdir` is atomic, so a worker claims
the next unclaimed cell and nothing coordinates.

Claims live in the job's own scratch and done flags live in the shared results directory, which is
the distinction that makes a preempted job safe to requeue: the done flag is written LAST and only
after the result file is in place, so a requeued task skips what finished and redoes what did not.
A claim proves only that something started, which is why it is not consulted across jobs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def gpu_total_mib():
    """Memory of the GPU this task was given, taken as the MINIMUM of two independent readings.

    `nvidia-smi --query-gpu=memory.total` is the obvious source and is not sufficient on its own,
    because a MIG slice can report its PARENT card. CHTC's pool includes a machine whose eight
    advertised GPUs are 3g.40gb slices of 80 GB A100s, 40,100 MiB each, and if nvidia-smi there
    reports 81,920 then sizing from it would pack twelve cells into forty gigabytes and every task
    on that machine would die out of memory.

    CUDA is the authority, because `total_memory` is what the allocator will actually enforce: a
    process inside a MIG slice sees the slice. So read both and take the smaller, which fails safe
    whichever one is lying, and print both so a disagreement is visible in the log rather than
    inferred later from a crash.
    """
    smi = cuda = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        vals = [int(x.strip()) for x in out.stdout.split("\n") if x.strip().isdigit()]
        smi = min(vals) if vals else None
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            cuda = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
    except Exception:
        pass
    have = [v for v in (smi, cuda) if v]
    if not have:
        return None
    if smi and cuda and abs(smi - cuda) > 2048:
        print(f"[pool] WARNING nvidia-smi says {smi} MiB and CUDA says {cuda} MiB; taking the "
              f"smaller. A gap this size is the MIG-slice case.", flush=True)
    print(f"[pool] GPU memory: nvidia-smi {smi}, CUDA {cuda}, using {min(have)} MiB", flush=True)
    return min(have)


def gpu_free_mib():
    """Memory actually AVAILABLE right now, which is not the same as the card's size.

    THE POOL SIZED ITSELF FROM TOTAL MEMORY AND THAT ASSUMES THE CARD IS OURS. It is not always.
    Measured on CHTC 2026-10-06: two tranche 6 tasks each lost cells to out-of-memory errors on a
    MIG 3g.40gb slice of gpu2011, and both error messages name the same two FOREIGN processes
    holding 11.63 and 13.02 GiB, three hours apart. Our own cells never outlive their task, so about
    25 GiB of that 39.25 GiB slice belonged to another user while our six cells ran. A MIG slice is
    advertised as a GPU and is not an exclusive one.

    Sizing from free memory would have given 2 cells there instead of 6. WHAT IT DOES NOT DO is
    protect against a neighbor that arrives after the pool has sized itself, which no start-time
    reading can. That case stays with the poison gate, which catches the result, and with excluding
    the machine. So this narrows the window rather than closing it, and the log line says which
    number was used so a later surprise is traceable.
    """
    smi = cuda = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        vals = [int(x.strip()) for x in out.stdout.split("\n") if x.strip().isdigit()]
        smi = min(vals) if vals else None
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            cuda = int(torch.cuda.mem_get_info()[0] // (1024 * 1024))
    except Exception:
        pass
    have = [v for v in (smi, cuda) if v]
    return min(have) if have else None


def condor_alloc():
    """Cores and memory from `$_CONDOR_MACHINE_AD`, which is the only honest source under HTCondor.

    MEASURED ON A REAL CHTC GPU NODE, and every other source lied. A job given 16 cores and 96 GB
    saw `len(os.sched_getaffinity(0))` = 64 and `os.cpu_count()` = 64, because HTCondor does not pin
    to a cpuset; `/sys/fs/cgroup/memory.max` did not exist inside the container in either the v1 or
    v2 spelling; and there is no `_CONDOR_MEMORY` variable. So the cgroup-then-sysconf path fell
    through to the machine's PHYSICAL memory, 376 GB on that node and up to 1.7 TB on the H200 ones,
    and the cores path read the whole machine. Sizing from either would have started sixteen cells
    against a 96 GB request and been killed.

    The allocation is in the machine ad, as `Cpus = 16` and `Memory = 98304`, so read that first and
    treat affinity and sysconf as the fallback they are.
    """
    p = os.environ.get("_CONDOR_MACHINE_AD")
    if not p or not os.path.exists(p):
        return None, None
    cpus = mem = None
    try:
        for line in open(p):
            k, _, v = line.partition("=")
            k, v = k.strip().lower(), v.strip()
            if k == "cpus" and v.replace(".", "").isdigit():
                cpus = int(float(v))
            elif k == "memory" and v.replace(".", "").isdigit():
                mem = int(float(v))                      # HTCondor reports Memory in MiB
    except Exception:
        return None, None
    return cpus, mem


def host_mem_mib():
    """Host memory this task may actually use, preferring whatever the scheduler will enforce."""
    _, cmem = condor_alloc()
    if cmem:
        return cmem
    for p in ("/sys/fs/cgroup/memory.max",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit():
                n = int(v) // (1024 * 1024)
                if 0 < n < 64 * 1024 * 1024:          # ignore the "unlimited" sentinel
                    return n
        except Exception:
            pass
    for k, mult in (("SLURM_MEM_PER_NODE", 1), ("_CONDOR_MEMORY", 1)):
        v = os.environ.get(k)
        if v and v.isdigit():
            return int(v) * mult
    v, c = os.environ.get("SLURM_MEM_PER_CPU"), os.environ.get("SLURM_CPUS_PER_TASK")
    if v and c and v.isdigit() and c.isdigit():
        return int(v) * int(c)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
    except Exception:
        return None


# Lines that mean a cell finished but did not compute what every other cell computed. Each one is a
# WARNING in Ultralytics rather than an error, so the cell exits 0 and looks finished.
# EVERY WAY A CELL CAN SILENTLY STOP RUNNING THE RECIPE. Each of these lets the cell exit 0 and
# write a normal-looking result, so the done flag is a lie and nothing downstream can tell.
#
# MATCHING ON A SUBSTRING RATHER THAN A FULL MESSAGE IS DELIBERATE. The campaign's first version of
# this listed only the assigner line, and a SECOND out-of-memory path was then found that shares
# none of its words: Ultralytics halves the batch and carries on. A gate built from the one failure
# anybody had seen did not generalize to the next one, so these are now as broad as they can be
# without matching ordinary progress output.
POISON_LINES = (
    # 1. THE LOSS MOVED TO THE CPU. Ultralytics catches an out-of-memory error inside the label
    # assignment and finishes on the CPU, changing the floating-point reduction order. Measured: 13
    # falcon cells, all at Ohio level 36. Card-pressure dependent, so it appears only when a card is
    # oversubscribed.
    "OutOfMemoryError in TaskAlignedAssigner",
    # 2. THE BATCH WAS HALVED, which is a RECIPE CHANGE and not a reordering. Ultralytics 8.4.76
    # catches an out-of-memory error in the first epoch and retries at half the batch, so a cell
    # trains at batch 8 where the recipe says 16, changing the steps per epoch and the batch-norm
    # statistics. First seen on CHTC cell r03511. Its message contains neither
    # "OutOfMemoryError" nor "using CPU", which is why the first version of this gate missed it:
    #   "WARNING CUDA out of memory with batch=16. Reducing to batch=8 and retrying (1/3)."
    "Reducing to batch",
    # 3. ANY out-of-memory text at all, as the backstop. A cell that mentions running out of GPU
    # memory and still claims success has, by construction, done something the recipe did not ask
    # for, whatever route Ultralytics took to survive it. This is the pattern the CHTC result collector
    # has used from the start, and it is the one that caught r03511 when the pool did not.
    "CUDA out of memory",
    # 4. Non-maximum suppression gives up after a time limit that scales with the batch, truncating
    # the detections it returns, which lowers a validation score. It is load dependent in the same
    # way as the out-of-memory paths and just as silent. None of the 930 VT cell logs scanned on
    # 2026-10-05 had it. It first appeared on 2026-10-11 in CHTC cell r02924, one
    # of six cells sharing an A100 80 GB that was bound by its 16 CPU cores. That log shows the
    # line below twice during per-epoch validation; the cell still ended OK, and the pool revoked
    # its done flag on this match:
    #   "WARNING NMS time limit 3.600s exceeded"
    "NMS time limit",
)


def scan_log_for_poison(path):
    """Return the offending text if a cell's log shows it silently changed its own arithmetic.

    Reads the whole log, which is up to a megabyte of progress bars, once per finished cell. That is
    cheap against a cell that took hours, and the alternative, scanning afterwards, is what let 11
    contaminated results sit in the results tree for a day.
    """
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    low = text.lower()
    for marker in POISON_LINES:
        # case-insensitive, because the batch-reduction warning has appeared both as "CUDA out of
        # memory" and "CUDA Out Of Memory" across Ultralytics versions and a gate that depends on
        # capitalization is a gate that will be missed once
        if marker.lower() in low:
            return marker
    return None


def task_cores():
    """Cores this task may actually use, and WHERE that number came from.

    The machine ad comes first for the reason given in `condor_alloc`: HTCondor does not pin the job
    to a cpuset, so affinity and `cpu_count` both report the whole machine and a 16-core job reads
    64. Under Slurm the cgroup really does restrict affinity, so the second rung is right there and
    the environment variable is only a fallback for a node that reports neither.
    """
    ccpus, _ = condor_alloc()
    if ccpus:
        return ccpus, "machine ad"
    try:
        n = len(os.sched_getaffinity(0))
        if n:
            return n, "affinity"
    except Exception:
        pass
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v and v.isdigit():
        return int(v), "SLURM_CPUS_PER_TASK"
    return (os.cpu_count() or 1), "cpu_count"


def auto_concurrent(per_cell_mib, reserve_mib, cap, fallback,
                    host_per_cell_mib=14400, cores_per_cell=2.5):
    """Set the cell count from the CARD, not from whichever card was measured first.

    This exists because copying a working number between partitions is a measured near-miss rather
    than a hypothetical. Five concurrent cells peaked at 30,862 MiB on an 81,920 MiB A100, so 6.2 GB
    a cell. Carrying that 5 onto falcon's A30 would have hit out-of-memory on every task, because an
    A30 has 24 GB. The campaign has now run on cards from 24 GB to 141 GB, and the CHTC pool admits
    machines from an 11 GB RTX 2080 Ti to a 141 GB H200, so a fixed number is wrong somewhere by
    construction. Reading the card at run time makes one submission correct everywhere.
    """
    tot = gpu_total_mib()
    if tot is None:
        print(f"[pool] could not read GPU memory; falling back to {fallback} cells", flush=True)
        return fallback, None

    # THREE CEILINGS, NOT ONE, and the card is only the first of them. A job asks for its host
    # memory and its cores at SUBMISSION and learns its card at RUN TIME, so sizing from the card
    # alone over-commits whatever was requested. Measured on tinkercliffs: 5 cells held 70.4 GB
    # resident, so about 14 GB of HOST memory a cell. A job landing on a 141 GB H200 would start 16
    # cells and want roughly 225 GB, and a 112 GB request would be killed by its own memory limit,
    # taking the whole task with it. Cores bind too: each cell is one trainer plus its loader
    # workers, and falcon's 7-cell L40S tasks run 21 loader processes on 16 cores.
    # THE CARD'S SIZE IS NOT THE MEMORY WE HAVE, when the card is not exclusively ours. A MIG slice
    # is advertised as a GPU and can carry another user's processes; on CHTC's gpu2011 about 25 GiB
    # of a 39.25 GiB slice was held by two foreign processes while six of our cells ran, and four
    # cells were lost. So size from whichever of total-less-reserve and free-now is SMALLER, which
    # is the card's size when we have it to ourselves and the real headroom when we do not.
    free = gpu_free_mib()
    usable = tot - reserve_mib
    if free is not None:
        shortfall = tot - free
        if free < usable:
            print(f"[pool] WARNING only {free} MiB of this {tot} MiB GPU is free, so {shortfall} "
                  f"MiB is held by processes that are not ours. Sizing from the free figure. If "
                  f"this card is shared the cells may still be squeezed later, which the poison "
                  f"gate catches but does not prevent.", flush=True)
            usable = free - reserve_mib
    by_gpu = usable // per_cell_mib
    hm = host_mem_mib()
    by_host = (hm - 4096) // host_per_cell_mib if hm else None
    cores, csrc = task_cores()
    by_cores = int(cores / cores_per_cell)

    lims = {"card": by_gpu, "cores": by_cores, "cap": cap}
    if by_host is not None:
        lims["host"] = by_host
    n = max(1, min(v for v in lims.values() if v is not None))
    binding = min(lims, key=lambda k: lims[k] if lims[k] is not None else 10 ** 9)
    print(f"[pool] sizing: card {tot} MiB (free {free}) -> {by_gpu}; host "
          f"{hm if hm else 'unknown'} MiB -> {by_host}; {cores} cores ({csrc}) -> {by_cores}; "
          f"cap {cap}. Using {n} concurrent cells, bound by {binding}.", flush=True)
    return int(n), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="file of run ids, one per line")
    ap.add_argument("--concurrent", type=int, required=True,
                    help="cells per GPU, or 0 to size it from the card at run time")
    ap.add_argument("--per-cell-mib", type=int, default=6350,
                    help="measured GPU memory per concurrent cell: 30,862 MiB for 5 cells")
    ap.add_argument("--reserve-mib", type=int, default=2048)
    ap.add_argument("--host-per-cell-mib", type=int, default=14400,
                    help="measured HOST memory per cell: 70.4 GB resident for 5 cells")
    ap.add_argument("--cores-per-cell", type=float, default=2.5,
                    help="one trainer plus its loader workers; falcon runs 7 cells on 16 cores")
    ap.add_argument("--max-concurrent", type=int, default=16,
                    help="ceiling regardless of card size, since beyond about 5 cells the GPU is "
                         "already at 100 percent utilization and more cells only split it")
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--one", required=True, help="path to sweep_one.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--deadline-seconds", type=int, default=0,
                    help="stop claiming new cells once this much of the walltime is gone")
    # THE BAD-CARD BAIL. A GPU can answer its name and its memory and then fail every cell at the
    # first `.to(device)`. Measured on CHTC 2026-10-04: one L40S (GPU-fa14651d on jcaicedogpu0004)
    # raised `torch.AcceleratorError: CUDA error: uncorrectable ECC error encountered` for every
    # cell of four consecutive tasks, 90 cells, each in about ten seconds. Nothing was corrupted,
    # because no done flag is written on failure, but the task left the queue looking exactly like a
    # task that had finished, and the freed card immediately took the next one. A scheduler cannot
    # tell those apart from the outside, so the pool has to say which it was.
    ap.add_argument("--bad-card-seconds", type=int, default=180,
                    help="a cell that fails in less than this has not failed for a reason of its "
                         "own: the smallest cell in this campaign still costs about 17 minutes, so "
                         "a ten-second exit is the machine and not the data")
    ap.add_argument("--bad-card-failures", type=int, default=3,
                    help="how many DISTINCT cells must fail that fast, with none ever having "
                         "finished and none still running, before the pool reports the card rather "
                         "than the cells. Exit 85, which a submit file turns into a retry "
                         "elsewhere; a Slurm array would behave the same way on a bad node.")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER,
                    help="-- then every argument sweep_one.py needs")
    a = ap.parse_args()

    extra = [x for x in a.passthrough if x != "--"]
    want = [L.strip() for L in open(a.runs) if L.strip()]
    claims, out, logs = Path(a.claims), Path(a.out), Path(a.log_dir)
    for p in (claims, out / "done", logs):
        p.mkdir(parents=True, exist_ok=True)

    conc = a.concurrent
    gpu_mib = None
    if conc <= 0:
        conc, gpu_mib = auto_concurrent(a.per_cell_mib, a.reserve_mib,
                                        a.max_concurrent, fallback=3,
                                        host_per_cell_mib=a.host_per_cell_mib,
                                        cores_per_cell=a.cores_per_cell)

    # TELL EACH CELL HOW MANY CELLS SHARE ITS CARD, so the result records it. `sweep_one.py` has
    # always READ `SWEEP_CONCURRENT` into its `host_env` block and nothing ever SET it, so the field
    # was silently absent from every result: a read with no writer, which looks like a working
    # feature and records nothing.
    #
    # It matters more than most of that block. Cells per card is the leading suspect for why two
    # runs of one cell on two facilities differ, because it is the only thing that differed on BOTH
    # sides between two comparisons that came out with opposite signs, and it is also what caused
    # the one measured contamination in this campaign, an out-of-memory fallback in the loss on an
    # oversubscribed card. A factor nobody records cannot be controlled for afterwards.
    os.environ["SWEEP_CONCURRENT"] = str(conc)
    if gpu_mib is not None:
        os.environ["SWEEP_GPU_MIB"] = str(gpu_mib)

    start = time.time()
    pending = list(want)
    running = {}
    done = skipped = failed = 0
    fast_fails = []
    # at least two distinct cells, so a one-cell task can never diagnose its own machine
    need_fast = max(2, min(a.bad_card_failures, len(want)))
    bad_card = False

    def release(rid):
        """Give the claim back when a cell did not finish.

        The claim exists so two workers cannot run one cell at the same time; the DONE FLAG is what
        records completion. So a failed cell's claim is stale the moment it fails. Today's caller
        passes `$TMPDIR/bur_$TID/claims`, which is node-local and vanishes with the task, so nothing
        is currently lost by leaving it: that is a property of one caller and not of this pool, and
        a shared claims directory would silently make every failed cell unrecoverable.
        """
        try:
            os.rmdir(claims / rid)
        except OSError:
            pass

    def claim_next():
        while pending:
            rid = pending.pop(0)
            if (out / "done" / rid).exists():
                nonlocal skipped
                skipped += 1
                continue
            try:
                os.mkdir(claims / rid)
            except FileExistsError:
                continue
            return rid
        return None

    while pending or running:
        expired = a.deadline_seconds and (time.time() - start) > a.deadline_seconds
        while not expired and len(running) < conc:
            rid = claim_next()
            if rid is None:
                break
            lf = open(logs / f"{rid}.log", "w")
            p = subprocess.Popen(
                [a.python, a.one, "--run-id", rid] + extra,
                stdout=lf, stderr=subprocess.STDOUT)
            running[rid] = (p, lf, time.time())
            print(f"[pool] start {rid} ({len(running)} running, {len(pending)} left)", flush=True)

        for rid, (p, lf, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            del running[rid]
            el = time.time() - t0
            # A CELL THAT SILENTLY CHANGED ITS OWN ARITHMETIC IS NOT A DONE CELL. Ultralytics catches
            # a CUDA out-of-memory error inside the loss's label assignment and falls back to the
            # CPU with only a warning, so the cell finishes, exits 0 and writes a normal-looking
            # result whose assignment was computed in a different floating-point order from every
            # other cell in the campaign. MEASURED 2026-10-04: 11 of the 111 finished L40S cells
            # above 7,000 training tiles, 10 percent of that band, and zero below it, because the
            # card was oversubscribed. Nobody noticed for a day, because nothing fails.
            #
            # So the done flag is REVOKED when that line appears, which puts the cell back in the
            # outstanding set for a resubmission at a safer cell count. Revoking beats refusing to
            # write it, because sweep_one.py writes the flag itself as its last act and the pool
            # must not have to reach inside that.
            # AND THE WEIGHTS MUST GO WITH THE FLAG, which is the half this was missing. `sweep_one.py`
            # copies the best weights BEFORE the pool can revoke anything, so a poisoned cell left a
            # valid-looking `<run_id>.pt` beside the good ones. That is exactly how eleven models with
            # no surviving measurement reached S3 under `weights/falcon/`: every consumer downstream
            # had to know to exclude them, and the first one that did not, did not. Renaming rather
            # than deleting keeps the bytes for diagnosis while dropping the file out of every
            # `*.pt` glob, so the default action downstream becomes the correct one.
            bad = scan_log_for_poison(logs / f"{rid}.log")
            if rc == 0 and bad and (out / "done" / rid).exists():
                try:
                    (out / "done" / rid).unlink()
                except OSError:
                    pass
                wt, moved = out / f"{rid}.pt", ""
                if wt.exists():
                    try:
                        wt.rename(out / f"{rid}.pt.poisoned")
                        moved = f"; weights set aside as {rid}.pt.poisoned"
                    except OSError as exc:
                        moved = f"; COULD NOT set the weights aside ({exc}), exclude {rid}.pt by hand"
                failed += 1
                release(rid)
                print(f"[pool] POISONED {rid} rc=0 {el:.0f}s: {bad}; done flag revoked so it is "
                      f"redone{moved}. see {logs/f'{rid}.log'}", flush=True)
            elif rc == 0 and (out / "done" / rid).exists():
                done += 1
                print(f"[pool] done  {rid} rc=0 {el:.0f}s", flush=True)
            else:
                failed += 1
                release(rid)
                if el <= a.bad_card_seconds and rid not in fast_fails:
                    fast_fails.append(rid)
                print(f"[pool] FAIL  {rid} rc={rc} {el:.0f}s see {logs/f'{rid}.log'}", flush=True)

        # IS IT THE CARD? Three conditions, and the third is what makes this safe to act on.
        # Nothing has ever finished here, so the card has never been shown to work; enough distinct
        # cells have failed fast that the cells are not the common factor; and NOTHING IS STILL
        # RUNNING, which is the one that matters, because a card with a live training process on it
        # is not a dead card. Requiring an empty `running` set means the pool never kills work in
        # progress to make this diagnosis, so a few cells failing fast for reasons of their own
        # while others train normally cannot trigger it.
        if done == 0 and not running and len(fast_fails) >= need_fast:
            bad_card = True
            break
        if running or pending:
            time.sleep(5)
        if expired and not running:
            break

    left = len([r for r in want if not (out / "done" / r).exists()])
    print(f"[pool] finished: {done} done, {skipped} already done, {failed} failed, "
          f"{left} of {len(want)} still outstanding, {time.time()-start:.0f}s", flush=True)
    # ANY failed cell makes the task fail, which is a deliberate change from "only if nothing
    # succeeded". The old rule meant a task that lost 23 of 24 cells still exited 0, the Slurm
    # state read COMPLETED, and --mail-type=FAIL stayed silent, so a campaign with holes in it
    # would have reported as a clean success and the holes would have shown up only as gaps in a
    # plot weeks later. Running OUT OF TIME is different and is not a failure: the done flags
    # record what finished and a resubmission picks up the rest, which is the designed behavior.
    # THE CARD, NOT THE CELLS, and it is reported with a DIFFERENT exit code because the right
    # response is different. Exit 2 means this task's work failed and wants investigating; exit 85
    # means this task never got a working GPU and should simply run somewhere else. Returning 2
    # here would send the task back to the queue looking like bad data, and returning 0 would send
    # it away looking finished, which is what let one faulty card consume four tasks in nine
    # minutes.
    if bad_card:
        print(f"[pool] BAD CARD: {len(fast_fails)} distinct cells each failed in under "
              f"{a.bad_card_seconds}s, none ever finished here, and nothing is still running. "
              f"Cells: {' '.join(fast_fails)}. Exiting 85 so this task is retried on another "
              f"machine. No done flag was written, so every cell of this task is simply owed. "
              f"Logs are in {logs}", flush=True)
        return 85
    if failed:
        print(f"[pool] FAILING the task because {failed} cell(s) failed; their logs are in "
              f"{logs}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
