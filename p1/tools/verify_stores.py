"""Assert the four tile stores are mutually consistent before any of them is trained on.

MY FIRST VERSION OF THIS CHECK WAS WRONG AND FAILED THE STORES, which is worth recording because the
wrong expectation is the natural one. I asserted that tile count scales as the inverse square of the
tile edge, since the candidate grid does. It does not, because the tiler keeps only tiles that CONTAIN
a bur plus background at a fixed ratio, and a larger tile makes a larger fraction of candidates
positive. Measured: positives run 10,110 at 448, 7,098 at 640, 4,511 at 960 and 3,089 at 1280, so the
ratio against 640 is 1.42x where the grid law predicts 2.04x, and 0.41x where it predicts 0.25x. A
bigger tile also merges nearby burs into one tile, which is the rest of the gap.

THE REAL INVARIANTS, all of them properties the tiling cannot change:

  source boxes      identical in every store, because it is a property of the label files. A
                    disagreement means a store read a different label set.
  boxes per source  about 3.4 at every tile size. At a 50 percent overlap each bur falls inside
                    roughly four tiles whatever the tile size, reduced a little by frame edges and by
                    the 75 percent visibility rule that gray-fills a clipped bur instead of labeling
                    it. This is the number that would move if a stride were wrong.
  background ratio  the configured 0.5, drifting down at large tiles only because a 1280 tile over a
                    frame holding one tree is almost always positive, so there are fewer background
                    candidates to sample.
  frames            197 in every store.
  stride            half the tile, which is what keeps the overlap at 50 percent across the arm.
"""
import json
import os
import sys

R = sys.argv[1]
rows = []
for T in (448, 640, 960, 1280):
    p = os.path.join(R, "tilestore", "train%d" % T, "MANIFEST.json")
    if not os.path.exists(p):
        continue
    m = json.load(open(p))
    fr = m["frames"]
    rows.append(dict(
        tile=T, stride=m["stride"], frames=len(fr),
        pos=sum(v["positive_tiles"] for v in fr.values()),
        bg=sum(v["background_tiles"] for v in fr.values()),
        boxes=sum(v["boxes"] for v in fr.values()),
        src=sum(v["boxes_in_source"] for v in fr.values()),
        masked=sum(v["partials_masked"] for v in fr.values())))

print("%5s %6s %6s %8s %10s %7s %11s %11s %9s" % (
    "tile", "stride", "frames", "positive", "background", "bg/pos", "tile boxes", "source", "per src"))
ok = True
for r in rows:
    bgr = r["bg"] / max(r["pos"], 1)
    per = r["boxes"] / max(r["src"], 1)
    print("%5d %6d %6d %8d %10d %7.2f %11d %11d %9.2f" % (
        r["tile"], r["stride"], r["frames"], r["pos"], r["bg"], bgr, r["boxes"], r["src"], per))
    for cond, msg in (
            (r["stride"] == r["tile"] // 2, "stride is not half the tile"),
            (r["frames"] == 197, "frame count is not 197"),
            (0.35 <= bgr <= 0.55, "background ratio %.2f is outside 0.35 to 0.55" % bgr),
            (3.0 <= per <= 4.0, "boxes per source box %.2f is outside 3.0 to 4.0" % per),
            (r["masked"] > 0, "no partially visible bur was gray-filled, so the seam rule did not run")):
        if not cond:
            print("    FAIL at tile %d: %s" % (r["tile"], msg))
            ok = False

tots = sorted({r["src"] for r in rows})
print("\nsource-box totals across the %d stores: %s" % (len(rows), tots))
if len(tots) != 1:
    print("  FAIL: the stores did not read the same label set")
    ok = False

print("\n" + ("PASSED: one source-box total, about 3.4 tile boxes per source box at every tile size, "
              "the configured background ratio, 197 frames and a half-tile stride throughout"
              if ok else "FAILED: see above"))
sys.exit(0 if ok else 1)
