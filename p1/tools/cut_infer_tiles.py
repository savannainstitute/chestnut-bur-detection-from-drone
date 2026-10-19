"""Cut the whole-frame inference grid for one fold's test frames, once per Slurm task.

Shipping these tiles pre-cut was measured at 6.1 GB against 4.4 GB for the frames they come from,
because a 20 percent overlap means 1.37 times the pixels and re-encoding already-compressed JPEG
inflates rather than shrinks. So the frames travel and the grid is cut on the node, into the node's
own scratch, where the several training runs packed into one allocation all read it.

The grid matches `eval_arms.py`: a 20 percent overlap, so a 512 pixel stride at 640, and the tiles
come from the NATIVE frame, which is what a deployed run sees.
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir tree holding the frames")
    ap.add_argument("--stems", required=True, help="file of <tree>__<DJI stem> lines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    a = ap.parse_args()

    index = {}
    for dp, _, fn in os.walk(os.path.expanduser(a.frames)):
        for f in fn:
            # skip `._*` AppleDouble forks: hidden from ls, visible to os.walk, and they would be
            # opened as images
            if f.lower().endswith((".jpg", ".png")) and not f.startswith("._"):
                index.setdefault(f, os.path.join(dp, f))

    out = Path(a.out) / str(a.tile)
    out.mkdir(parents=True, exist_ok=True)
    stride = int(round(a.tile * (1 - a.overlap)))
    frames = {}
    for stem in [s.strip() for s in open(a.stems) if s.strip()]:
        # the frame is named <tree>__<DJI stem> in the annotation bundles and <DJI stem> in the
        # raw flight folders, so accept either
        cands = [stem] + ([stem.split("__")[1]] if "__" in stem else [])
        src = next((index[c + e] for c in cands for e in (".JPG", ".jpg", ".png")
                    if c + e in index), None)
        if src is None:
            raise SystemExit(f"no frame on disk for {stem}")
        d = out / stem
        done = d / ".complete"
        im = Image.open(src)
        W, H = im.size
        # THE GRID MUST REACH THE EDGES. A plain range leaves whatever does not divide by the
        # stride uncovered: at 8192 by 5460 with a 640 tile and a 512 stride that is the right 384
        # and bottom 212 pixels, and 34 of the 1,232 hand-annotated crown burs (2.76 percent, on 10
        # of the 154 trees) have their centres in that margin. Those burs are unfindable, which is
        # a hard ceiling of 0.972 on recall. Worse for the tile-size arm, the margin depends on the
        # tile size, so comparing 448 against 1280 would compare two different reachable fractions
        # of the frame. Appending a final flush offset on each axis costs one row and one column of
        # tiles and removes both problems.
        def offsets(span):
            if span <= a.tile:
                return [0]
            o = list(range(0, span - a.tile + 1, stride))
            if o[-1] != span - a.tile:
                o.append(span - a.tile)
            return o

        origins = [[x, y] for y in offsets(H) for x in offsets(W)]
        frames[stem] = dict(W=W, H=H, tile=a.tile, stride=stride, origins=origins)
        if done.exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        im = im.convert("RGB")
        for x, y in origins:
            im.crop((x, y, x + a.tile, y + a.tile)).save(d / f"{x}_{y}.jpg", quality=92)
        done.write_text(f"{len(origins)}\n")

    (out / "FRAMES.json").write_text(json.dumps(
        dict(tile=a.tile, overlap=a.overlap, frames=frames), indent=1))
    print(f"{len(frames)} frames, {sum(len(v['origins']) for v in frames.values())} tiles -> {out}")


if __name__ == "__main__":
    main()
