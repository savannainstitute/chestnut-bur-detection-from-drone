"""Cut every frame ONCE into a reusable tile store, so a training sweep never re-tiles.

A sweep of about 1,900 training runs that each re-tiled its own subset would decode a 45-megapixel
JPEG tens of thousands of times, and on a shared cluster filesystem that is the expensive part of
the job rather than the training. So the tiles are cut once here and shipped as a single tar.

TWO STORES, because training and evaluation want different grids.

`train<tile>/` is a FLAT pair of directories holding every training tile from every frame, named
`<frame stem>_t<x>_<y>`. Ultralytics accepts a text file of image paths as its `train:` key and
finds each label by swapping `/images/` for `/labels/`, so a run selects its own subset by writing
a list of paths. No symlinks, no copying, and selecting by source frame is a prefix match on the
name. Tiling follows `build_tiles_vblue.py` exactly, gray-fill included, since that recipe is the
one being held fixed while the data varies.

`infer<tile>/` holds the whole-frame inference grid for the Clarks Run frames, one directory per
frame plus the frame size and tile origins, so scoring maps boxes back to original-frame pixels
without opening the frame at all. The grid matches `eval_arms.py` (20 percent overlap, so a 512
stride at 640) because the earlier results were measured on it.
"""

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_boxes(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path).read().split("\n"):
        f = line.split()
        if len(f) == 5:
            out.append((int(f[0]), *map(float, f[1:5])))
        elif len(f) >= 6:
            out.append((int(f[0]), *map(float, f[2:6])))
    return out


def to_pixels(boxes, W, H, keep):
    return [(b[1] * W - b[3] * W / 2, b[2] * H - b[4] * H / 2,
             b[1] * W + b[3] * W / 2, b[2] * H + b[4] * H / 2)
            for b in boxes if b[0] in keep]


def cut_train(img, bx, stem, tile, stride, minvis, bg_ratio, out_img, out_lbl, rng):
    W, H = img.size
    cand = []
    for ty in range(0, max(1, H - tile) + 1, stride):
        for tx in range(0, max(1, W - tile) + 1, stride):
            tx1, ty1 = min(tx + tile, W), min(ty + tile, H)
            lines, masked = [], []
            for (b0, b1, b2, b3) in bx:
                ix0, iy0 = max(b0, tx), max(b1, ty)
                ix1, iy1 = min(b2, tx1), min(b3, ty1)
                iw, ih = ix1 - ix0, iy1 - iy0
                if iw <= 0 or ih <= 0:
                    continue
                area = (b2 - b0) * (b3 - b1)
                if area > 0 and (iw * ih) / area >= minvis:
                    lines.append(f"0 {((ix0+ix1)/2-tx)/tile:.6f} {((iy0+iy1)/2-ty)/tile:.6f} "
                                 f"{iw/tile:.6f} {ih/tile:.6f}")
                else:
                    masked.append((ix0 - tx, iy0 - ty, ix1 - tx, iy1 - ty))
            cand.append((tx, ty, tx1, ty1, lines, masked))

    pos = [c for c in cand if c[4]]
    neg = [c for c in cand if not c[4]]
    rng.shuffle(neg)
    chosen = pos + neg[:int(round(bg_ratio * len(pos)))]

    names, nbox, nmask = [], 0, 0
    for (tx, ty, tx1, ty1, lines, masked) in chosen:
        t = img.crop((tx, ty, tx1, ty1))
        if t.size != (tile, tile):
            p = Image.new("RGB", (tile, tile), (114, 114, 114))
            p.paste(t, (0, 0))
            t = p
        if masked:
            d = ImageDraw.Draw(t)
            for m in masked:
                d.rectangle(list(m), fill=(114, 114, 114))
            nmask += len(masked)
        name = f"{stem}_t{tx}_{ty}"
        t.save(out_img / f"{name}.jpg", quality=92)
        (out_lbl / f"{name}.txt").write_text(("\n".join(lines) + "\n") if lines else "")
        names.append(name)
        nbox += len(lines)
    return names, len(pos), len(chosen) - len(pos), nbox, nmask


def cut_infer(img, stem, tile, overlap, out_dir):
    W, H = img.size
    stride = int(round(tile * (1 - overlap)))
    d = out_dir / stem
    d.mkdir(parents=True, exist_ok=True)
    origins = []
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            img.crop((x, y, x + tile, y + tile)).save(d / f"{x}_{y}.jpg", quality=92)
            origins.append([x, y])
    return dict(W=W, H=H, tile=tile, stride=stride, origins=origins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="name:images_dir_or_index:labels_dir:scale")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=320)
    ap.add_argument("--minvis", type=float, default=0.75)
    ap.add_argument("--bg-ratio", type=float, default=0.5)
    ap.add_argument("--classes", default="0,2,4")
    ap.add_argument("--scales", default=None, help="json frame stem -> scale override")
    ap.add_argument("--infer-source", default=None,
                    help="name of the source whose frames also get an inference grid")
    ap.add_argument("--infer-overlap", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    ti = out / f"train{a.tile}" / "images"
    tl = out / f"train{a.tile}" / "labels"
    for p in (ti, tl):
        p.mkdir(parents=True, exist_ok=True)
    inf = out / f"infer{a.tile}"
    keep = {int(c) for c in a.classes.split(",")}
    per_frame = json.load(open(a.scales)) if a.scales else {}
    rng = random.Random(a.seed)

    manifest, frames = {}, {}
    for spec in a.source:
        name, idir, ldir, scale = spec.split(":")
        scale = float(scale)
        idir, ldir = Path(idir).expanduser(), Path(ldir).expanduser()
        # the image directory may be a tree, so index it by basename once
        # `._*` AppleDouble resource forks ride along in any tar written on a Mac, they are hidden
        # from `ls` and not from os.listdir or glob, and `._x.txt` ends in `.txt`. One of them in a
        # label directory doubles the apparent frame count and tiles garbage.
        index = {}
        for dp, _, fn in os.walk(idir):
            for f in fn:
                if f.lower().endswith((".jpg", ".png")) and not f.startswith("._"):
                    index.setdefault(f, os.path.join(dp, f))

        for lp in sorted(p for p in ldir.glob("*.txt") if not p.name.startswith("._")):
            stem = lp.stem
            # A Clarks Run label is named <tree>__<DJI stem>. The frame carries the same name in
            # the bundles shipped to Ren and the bare DJI name in the raw flight folders, so try
            # both rather than assuming which copy this run is reading.
            cands = [stem] + ([stem.split("__")[1]] if "__" in stem else [])
            src = next((index[c + e] for c in cands for e in (".JPG", ".jpg", ".png")
                        if c + e in index), None)
            if src is None:
                raise SystemExit(f"no frame on disk for {stem}")

            img = Image.open(src).convert("RGB")
            # the per-frame scale map is keyed on whichever name the depositor used, so try the
            # full stem and the bare DJI stem before falling back to the source's own scale
            fs = next((per_frame[c] for c in cands if c in per_frame), scale)
            if fs != 1.0:
                img = img.resize((round(img.width * fs), round(img.height * fs)), Image.LANCZOS)
            bx = to_pixels(read_boxes(lp), img.width, img.height, keep)

            names, npos, nbg, nbox, nmask = cut_train(
                img, bx, stem, a.tile, a.stride, a.minvis, a.bg_ratio, ti, tl, rng)
            manifest[stem] = dict(source=name, scale=fs, tiles=names, positive_tiles=npos,
                                  background_tiles=nbg, boxes=nbox, partials_masked=nmask,
                                  boxes_in_source=len(bx))

            if a.infer_source and name == a.infer_source:
                # the inference grid is cut from the NATIVE frame, which is what a deployed run sees
                native = Image.open(src).convert("RGB")
                frames[stem] = cut_infer(native, stem, a.tile, a.infer_overlap, inf)
                frames[stem]["boxes_in_source"] = len(
                    to_pixels(read_boxes(lp), native.width, native.height, keep))
            print(f"{stem}: {npos}+{nbg} tiles, {nbox} boxes", flush=True)

    (out / f"train{a.tile}" / "MANIFEST.json").write_text(json.dumps(
        dict(tile=a.tile, stride=a.stride, minvis=a.minvis, bg_ratio=a.bg_ratio,
             classes=sorted(keep), seed=a.seed, frames=manifest), indent=1))
    if frames:
        (inf / "FRAMES.json").write_text(json.dumps(
            dict(tile=a.tile, overlap=a.infer_overlap, frames=frames), indent=1))

    print(f"\n{len(manifest)} frames tiled, "
          f"{sum(len(v['tiles']) for v in manifest.values())} training tiles, "
          f"{sum(v['boxes'] for v in manifest.values())} boxes, "
          f"{len(frames)} inference grids")


if __name__ == "__main__":
    main()
