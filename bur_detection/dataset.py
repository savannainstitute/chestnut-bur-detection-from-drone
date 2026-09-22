"""Preprocessing utilities for YOLO bur detection.

Handles dataset preparation, image tiling, and aggregating tree-level
detections.
"""

import csv
import numpy as np
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from typing import List, Tuple, Dict, Callable, Optional, Sequence
import yaml

from bur_detection.utils import set_seed


_BUR_TILE_GROUP_RE = re.compile(r"^(.*)_\d+_\d+$")


def bur_tile_group_key(path) -> str:
    """Group key for tiles named '<source>_<x>_<y>'.

    Strips the trailing two offset fields so sibling tiles from one
    source canopy share a key. Falls back to the full stem.
    """
    stem = Path(path).stem
    m = _BUR_TILE_GROUP_RE.match(stem)
    return m.group(1) if m else stem


def _count_label_lines(label_path) -> int:
    """Count non-empty lines (annotations) in a YOLO label file."""
    label_path = Path(label_path)
    if not label_path.exists():
        return 0
    with open(label_path) as f:
        return sum(1 for line in f if line.strip())


def _group_balanced_split(
    image_files, labels_dir, group_key_fn, splits=(0.7, 0.2, 0.1), seed=666
):
    """Split images into train/val/test by source group.

    Never splits a group across sets, balancing annotation and tile
    counts to prevent tree-level leakage. Returns lists of Paths.
    """
    labels_dir = Path(labels_dir)
    split_names = ["train", "val", "test"]
    fracs = dict(zip(split_names, splits))

    # Build groups: {key: {"files": [...], "annotations": int, "tiles": int}}
    grouped: Dict[str, Dict] = {}
    for img in image_files:
        key = group_key_fn(img)
        g = grouped.setdefault(
            key, {"files": [], "annotations": 0, "tiles": 0}
        )
        g["files"].append(img)
        g["annotations"] += _count_label_lines(
            labels_dir / f"{Path(img).stem}.txt"
        )
        g["tiles"] += 1

    fg_groups, bg_groups = [], []
    for name, g in grouped.items():
        item = {
            "name": name,
            "annotations": g["annotations"],
            "tiles": g["tiles"],
        }
        (fg_groups if item["annotations"] > 0 else bg_groups).append(item)

    if not fg_groups:
        raise ValueError(
            "No foreground groups (all labels empty); cannot build a "
            "meaningful split."
        )

    total_ann = sum(g["annotations"] for g in fg_groups)
    total_fg_tiles = sum(g["tiles"] for g in fg_groups)
    target_ann = {k: fracs[k] * total_ann for k in split_names}
    target_fg_tiles = {k: fracs[k] * total_fg_tiles for k in split_names}

    rng = random.Random(seed)
    rng.shuffle(fg_groups)
    fg_groups.sort(key=lambda g: g["annotations"], reverse=True)

    state = {
        k: {"groups": [], "ann": 0, "tiles": 0, "fg_groups": 0}
        for k in split_names
    }

    def assign(group, split_key):
        state[split_key]["groups"].append(group["name"])
        state[split_key]["ann"] += group["annotations"]
        state[split_key]["tiles"] += group["tiles"]
        if group["annotations"] > 0:
            state[split_key]["fg_groups"] += 1

    # Guarantee minimum foreground-group presence in val/test when
    # enough exist.
    min_fg = {"train": 1, "val": 0, "test": 0}
    if len(fg_groups) >= 2:
        min_fg["val"] = 1
    if len(fg_groups) >= 3:
        min_fg["test"] = 1

    # Meet the minimum with the smallest groups, so bur-dense trees
    # stay for the main pass.
    for split_key in ["test", "val", "train"]:
        while state[split_key]["fg_groups"] < min_fg[split_key] and fg_groups:
            smallest_i = min(
                range(len(fg_groups)),
                key=lambda i: fg_groups[i]["annotations"],
            )
            assign(fg_groups.pop(smallest_i), split_key)

    # Assign remaining foreground groups (largest first) to the split with the
    # lowest projected load ratio vs target, converging to target fractions.
    for g in fg_groups:
        best_split, best_score = None, None
        for split_key in split_names:
            ann_ratio = (state[split_key]["ann"] + g["annotations"]) / max(
                1.0, target_ann[split_key]
            )
            tile_ratio = (state[split_key]["tiles"] + g["tiles"]) / max(
                1.0, target_fg_tiles[split_key]
            )
            score = 0.7 * ann_ratio + 0.3 * tile_ratio
            if best_score is None or score < best_score:
                best_split, best_score = split_key, score
        assign(g, best_split)

    # Assign pure-background groups to the current largest tile-count deficit.
    total_tiles = sum(g["tiles"] for g in grouped.values())
    target_total_tiles = {k: fracs[k] * total_tiles for k in split_names}
    rng.shuffle(bg_groups)
    for g in bg_groups:
        best_split = max(
            split_names,
            key=lambda k: target_total_tiles[k] - state[k]["tiles"],
        )
        assign(g, best_split)

    files = {k: [] for k in split_names}
    for k in split_names:
        for gname in state[k]["groups"]:
            files[k].extend(grouped[gname]["files"])

    def pct(x):
        return 100.0 * x / total_ann if total_ann > 0 else 0.0

    print("\nGroup-aware split (no source group spans splits):")
    for k in split_names:
        print(
            f"  {k.capitalize():5s}: {len(files[k]):4d} tiles, "
            f"{len(state[k]['groups']):3d} groups, "
            f"{state[k]['ann']:5d} annotations ({pct(state[k]['ann']):.1f}%)"
        )
    print(
        f"  Total: {total_tiles} tiles, {len(grouped)} groups, "
        f"{total_ann} foreground annotations"
    )

    return files["train"], files["val"], files["test"]


def prepare_dataset_splits(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    splits: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 666,
    group_key_fn: Optional[Callable[[Path], str]] = None,
) -> Dict[str, int]:
    """Create train/val/test splits and write dataset.yml.

    Writes one .txt of image paths per split. When group_key_fn is
    given, splits by group (no group spans splits) with
    annotation/tile balancing to prevent tree-level leakage;
    otherwise does a plain per-image shuffle. Returns split counts.
    """
    set_seed(seed)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_image_files = list(images_dir.glob("*.jpg")) + list(
        images_dir.glob("*.png")
    )

    if len(all_image_files) == 0:
        raise FileNotFoundError(f"No images found in {images_dir}")

    missing_labels = []
    for img_file in all_image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            missing_labels.append(img_file.name)

    if missing_labels:
        print(
            f"Warning: {len(missing_labels)} images have no "
            f"corresponding labels"
        )
        all_image_files = [
            f for f in all_image_files if f.name not in missing_labels
        ]

    if group_key_fn is not None:
        train_files, val_files, test_files = _group_balanced_split(
            all_image_files, labels_dir, group_key_fn, splits=splits, seed=seed
        )
    else:
        random.shuffle(all_image_files)
        train_ratio, val_ratio, _ = splits

        train_split = int(train_ratio * len(all_image_files))
        val_split = int((train_ratio + val_ratio) * len(all_image_files))

        train_files = all_image_files[:train_split]
        val_files = all_image_files[train_split:val_split]
        test_files = all_image_files[val_split:]

    split_files = {
        "train.txt": train_files,
        "val.txt": val_files,
        "test.txt": test_files,
    }

    for filename, files in split_files.items():
        with open(output_dir / filename, "w") as f:
            for file_path in files:
                rel_path = file_path  # relative to bur_detection/
                f.write(f"{rel_path.as_posix()}\n")

    dataset_yaml = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {0: "Chestnut-bur"},
    }
    with open(output_dir / "dataset.yml", "w") as f:
        yaml.dump(dataset_yaml, f)

    counts = {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }

    print("\nDataset splits created:")
    print(f"  Train: {counts['train']} images")
    print(f"  Val:   {counts['val']} images")
    print(f"  Test:  {counts['test']} images")
    print(f"\nSplit files saved to: {output_dir}")

    return counts


class CanopyTiler:
    """Crop, tile, and reconstruct detections for canopy images.

    Used for both labeled training tiles and unlabeled inference
    canopy images.
    """

    def __init__(self, tile_size: int = 224, overlap: float = 0.2):
        """Set the tile size, overlap fraction, and derived stride."""
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))

    def crop_canopy_from_polygon(
        self, image_path: Path, polygon_coords: Sequence[Sequence[float]]
    ) -> np.ndarray:
        """Crop the canopy region to its polygon bbox.

        Zeroes out pixels outside the polygon.
        """
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

        coords_array = np.array(polygon_coords)
        x_min = int(np.floor(coords_array[:, 0].min()))
        y_min = int(np.floor(coords_array[:, 1].min()))
        x_max = int(np.ceil(coords_array[:, 0].max()))
        y_max = int(np.ceil(coords_array[:, 1].max()))

        cropped_img = image.crop((x_min, y_min, x_max, y_max))

        mask = Image.new("L", (x_max - x_min, y_max - y_min), 0)
        draw = ImageDraw.Draw(mask)

        adjusted_coords = [(x - x_min, y - y_min) for x, y in polygon_coords]
        draw.polygon(adjusted_coords, fill=255)

        cropped_array = np.array(cropped_img)
        mask_array = np.array(mask)
        cropped_array[mask_array == 0] = 0

        return cropped_array

    def tile_image(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Sliding-window tile a (padded) image.

        Returns tiles plus per-tile (x, y) metadata.
        """
        img_height, img_width = image.shape[:2]

        padded_height = (
            (img_height + self.tile_size - 1) // self.tile_size
        ) * self.tile_size
        padded_width = (
            (img_width + self.tile_size - 1) // self.tile_size
        ) * self.tile_size

        padded_image = np.zeros(
            (padded_height, padded_width, 3), dtype=np.uint8
        )
        padded_image[:img_height, :img_width, :] = image

        tiles = []
        tile_info = []

        for tile_y in range(
            0, padded_height - self.tile_size + 1, self.stride
        ):
            for tile_x in range(
                0, padded_width - self.tile_size + 1, self.stride
            ):
                tile = padded_image[
                    tile_y : tile_y + self.tile_size,
                    tile_x : tile_x + self.tile_size,
                    :,
                ]

                if np.all(tile == 0):
                    continue

                tiles.append(tile)
                tile_info.append(
                    {
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "original_width": img_width,
                        "original_height": img_height,
                    }
                )

        return tiles, tile_info

    def reconstruct_detections(
        self, tile_detections: List[Dict], tile_info: List[Dict]
    ) -> List[Dict]:
        """Shift per-tile detections to full-image coords, clipped to image."""
        all_detections = []

        for tile_det, info in zip(tile_detections, tile_info):
            if len(tile_det["boxes"]) == 0:
                continue

            boxes = tile_det["boxes"].copy()
            boxes[:, [0, 2]] += info["tile_x"]
            boxes[:, [1, 3]] += info["tile_y"]

            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]], 0, info["original_width"]
            )
            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]], 0, info["original_height"]
            )

            for i, box in enumerate(boxes):
                all_detections.append(
                    {
                        "box": box,
                        "confidence": tile_det["confidences"][i],
                        "label": tile_det["labels"][i]
                        if "labels" in tile_det
                        else 0,
                    }
                )

        return all_detections

    def reconstruct_detections_core(
        self, tile_detections: List[Dict], tile_info: List[Dict]
    ) -> List[Dict]:
        """Combine tile predictions to full-image coords via each tile's core.

        Keeps only detections whose box center falls in each tile's
        non-overlapping core, so seam-straddling burs aren't
        double-counted. The core insets by half the overlap, except
        on image-boundary sides.
        """
        margin = (self.tile_size - self.stride) / 2.0
        all_detections = []

        for tile_det, info in zip(tile_detections, tile_info):
            if len(tile_det["boxes"]) == 0:
                continue

            tile_x, tile_y = info["tile_x"], info["tile_y"]
            img_w, img_h = info["original_width"], info["original_height"]

            core_x0 = tile_x + (margin if tile_x > 0 else 0)
            core_y0 = tile_y + (margin if tile_y > 0 else 0)
            core_x1 = (tile_x + self.tile_size) - (
                margin if (tile_x + self.tile_size) < img_w else 0
            )
            core_y1 = (tile_y + self.tile_size) - (
                margin if (tile_y + self.tile_size) < img_h else 0
            )

            boxes = tile_det["boxes"].copy()
            boxes[:, [0, 2]] += tile_x
            boxes[:, [1, 3]] += tile_y

            for i, box in enumerate(boxes):
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                if not (core_x0 <= cx <= core_x1 and core_y0 <= cy <= core_y1):
                    continue
                clipped = box.copy()
                clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, img_w)
                clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, img_h)
                all_detections.append(
                    {
                        "box": clipped,
                        "confidence": tile_det["confidences"][i],
                        "label": tile_det["labels"][i]
                        if "labels" in tile_det
                        else 0,
                    }
                )

        return all_detections


def _polygon_label_to_bbox(parts, w: int, h: int):
    """Convert a YOLO-segment label line to a pixel bbox on a (w,h) image.

    The line is `cls x1 y1 x2 y2 ...`.
    """
    xs = [float(parts[i]) * w for i in range(1, len(parts) - 1, 2)]
    ys = [float(parts[i]) * h for i in range(2, len(parts), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def _largest_polygon_px(label_path: Path, w: int, h: int):
    """Read the largest polygon from a YOLO-seg file as pixel coords.

    Largest is by vertex count.
    """
    if not label_path.exists():
        return None
    best = None
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 7:
            continue
        pts = [
            (float(p[i]) * w, float(p[i + 1]) * h)
            for i in range(1, len(p) - 1, 2)
        ]
        if best is None or len(pts) > len(best):
            best = pts
    return best


def _dedup_boxes(boxes, iou_thresh):
    """Greedily drop boxes overlapping a kept (larger) box by >= iou_thresh.

    Conservative de-duplication of obvious double-annotations: at a
    high threshold (e.g. 0.8) only near-identical boxes are removed,
    never genuinely distinct (touching) burs. Returns the kept boxes.
    """
    if not iou_thresh or iou_thresh >= 1.0 or len(boxes) < 2:
        return boxes
    order = sorted(
        range(len(boxes)),
        key=lambda i: (
            -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        ),
    )
    kept = []
    for i in order:
        b = boxes[i]
        is_dup = False
        for k in kept:
            ix1, iy1 = max(b[0], k[0]), max(b[1], k[1])
            ix2, iy2 = min(b[2], k[2]), min(b[3], k[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                continue
            union = (
                (b[2] - b[0]) * (b[3] - b[1])
                + (k[2] - k[0]) * (k[3] - k[1])
                - inter
            )
            if union > 0 and inter / union >= iou_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(b)
    return kept


def create_tiled_dataset(
    images_dir,
    labels_dir,
    output_dir,
    canopy_dir=None,
    tile_size: int = 224,
    overlap: float = 0.2,
    min_canopy_frac: float = 0.15,
    min_edge_keep_frac: float = 0.35,
    bg_keep_ratio: float = 0.3,
    dedup_iou: float = 0.8,
    seed: int = 666,
) -> Dict:
    """Tile full canopy images and polygon bur labels into a dataset.

    Geometry matches the inference CanopyTiler; also writes a
    group-aware split. Returns a stats dict.

    Per image: optionally crop+mask to its canopy polygon, tile at
    tile_size/overlap, and clip bur bboxes to each tile. Filtering
    drops mostly-background tiles (< min_canopy_frac), tiny edge
    fragments (< min_edge_keep_frac of the bur), and all but
    bg_keep_ratio of bur-free tiles. canopy_dir (optional) supplies
    YOLO-segment canopy polygons for masking.
    """
    images_dir, labels_dir, output_dir = (
        Path(images_dir),
        Path(labels_dir),
        Path(output_dir),
    )
    canopy_dir = Path(canopy_dir) if canopy_dir else None
    rng = random.Random(seed)
    tiler = CanopyTiler(tile_size=tile_size, overlap=overlap)

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    # Clear any previous run so re-tiling is idempotent -- stale
    # tiles must not survive.
    for d in (out_images, out_labels):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    image_files = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    for img_path in image_files:
        stem = img_path.stem
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            full_w, full_h = im.size
            full_arr = np.array(im)

        # Bur bboxes in full-image pixel coords (from the polygon labels).
        burs = []
        lbl = labels_dir / f"{stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) >= 7:
                    burs.append(_polygon_label_to_bbox(p, full_w, full_h))

        # Crop + mask to the canopy polygon (offsets needed to
        # translate bur boxes).
        off_x = off_y = 0
        canopy_poly = (
            _largest_polygon_px(canopy_dir / f"{stem}.txt", full_w, full_h)
            if canopy_dir
            else None
        )
        if canopy_poly:
            xs = [c[0] for c in canopy_poly]
            ys = [c[1] for c in canopy_poly]
            off_x, off_y = int(np.floor(min(xs))), int(np.floor(min(ys)))
            canopy_arr = tiler.crop_canopy_from_polygon(img_path, canopy_poly)
        else:
            canopy_arr = full_arr
        burs = [
            (x1 - off_x, y1 - off_y, x2 - off_x, y2 - off_y)
            for (x1, y1, x2, y2) in burs
        ]

        tiles, info = tiler.tile_image(canopy_arr)
        for tile, meta in zip(tiles, info):
            tx, ty = meta["tile_x"], meta["tile_y"]
            tile_stem = f"{stem}_{ty}_{tx}"
            stats["generated"] += 1

            # Canopy coverage = fraction of non-masked (non-black) pixels.
            coverage = float(np.count_nonzero(tile.any(axis=2))) / float(
                tile_size * tile_size
            )
            if coverage < min_canopy_frac:
                stats["low_canopy"] += 1
                continue

            tile_boxes = []
            for bx1, by1, bx2, by2 in burs:
                ix1, iy1 = max(bx1, tx), max(by1, ty)
                ix2, iy2 = min(bx2, tx + tile_size), min(by2, ty + tile_size)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 0 or ih <= 0:
                    continue
                clipped = (
                    bx1 < tx
                    or by1 < ty
                    or bx2 > tx + tile_size
                    or by2 > ty + tile_size
                )
                if (
                    clipped
                    and (iw * ih) / max(1.0, (bx2 - bx1) * (by2 - by1))
                    < min_edge_keep_frac
                ):
                    continue  # tiny edge sliver
                tile_boxes.append(
                    (ix1 - tx, iy1 - ty, ix2 - tx, iy2 - ty)
                )  # tile-local px

            n_before = len(tile_boxes)
            tile_boxes = _dedup_boxes(tile_boxes, dedup_iou)
            stats["dedup_removed"] += n_before - len(tile_boxes)
            lines = [
                f"0 {((x1 + x2) / 2.0) / tile_size:.6f} "
                f"{((y1 + y2) / 2.0) / tile_size:.6f} "
                f"{(x2 - x1) / tile_size:.6f} {(y2 - y1) / tile_size:.6f}"
                for (x1, y1, x2, y2) in tile_boxes
            ]

            is_bg = len(lines) == 0
            if is_bg and rng.random() > bg_keep_ratio:
                stats["bg_dropped"] += 1
                continue

            Image.fromarray(tile).save(
                out_images / f"{tile_stem}.jpg", quality=95
            )
            (out_labels / f"{tile_stem}.txt").write_text("\n".join(lines))
            stats["kept"] += 1
            stats["fg" if not is_bg else "bg"] += 1
            stats["boxes"] += len(lines)

    counts = prepare_dataset_splits(
        out_images,
        out_labels,
        output_dir,
        seed=seed,
        group_key_fn=bur_tile_group_key,
    )
    summary: dict[str, object] = dict(stats)
    summary["split"] = counts
    print(
        f"\nTiler: generated {stats['generated']}, kept {stats['kept']} "
        f"(fg {stats['fg']}, bg {stats['bg']}), {stats['boxes']} boxes | "
        f"low-canopy {stats['low_canopy']}, bg-dropped {stats['bg_dropped']}, "
        f"dup-boxes removed {stats['dedup_removed']}"
    )
    return summary


def tile_for_annotation(
    img_dir,
    canopy_dir,
    out_dir,
    tile_size=896,
    overlap=0.15,
    min_canopy_frac=0.15,
    jpeg_quality=95,
):
    """Tile full-res images for manual annotation, with a manifest.

    Maps each tile back to full-image pixel coords for aggregation:
        full_x = crop_x0 + tile_x + box_x ; full_y = crop_y0 + tile_y + box_y
    img_dir holds the source images; canopy_dir holds one YOLO-seg
    canopy polygon (.txt) per image stem. Tiles below
    min_canopy_frac canopy (non-masked) pixels are dropped.
    """
    img_dir, canopy_dir, out_dir = (
        Path(img_dir),
        Path(canopy_dir),
        Path(out_dir),
    )
    out_img = out_dir / "images"
    shutil.rmtree(out_dir, ignore_errors=True)  # clean re-run
    out_img.mkdir(parents=True, exist_ok=True)
    tiler = CanopyTiler(tile_size=tile_size, overlap=overlap)

    rows = []
    images = sorted(
        p
        for p in img_dir.glob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    for p in images:
        with Image.open(p) as im:
            full_w, full_h = im.size
        poly = _largest_polygon_px(
            canopy_dir / f"{p.stem}.txt", full_w, full_h
        )
        if not poly:
            print(f"[tile] no canopy polygon: {p.stem}")
            continue
        crop_x0 = int(min(x for x, _ in poly))
        crop_y0 = int(min(y for _, y in poly))
        cropped = tiler.crop_canopy_from_polygon(
            p, poly
        )  # bbox crop + mask outside polygon
        crop_h, crop_w = cropped.shape[:2]
        tiles, info = tiler.tile_image(cropped)
        kept = 0
        for tile, meta in zip(tiles, info):
            canopy_frac = float(np.any(tile != 0, axis=2).mean())
            if canopy_frac < min_canopy_frac:
                continue
            tx, ty = int(meta["tile_x"]), int(meta["tile_y"])
            fname = f"{p.stem}__x{tx}_y{ty}.jpg"
            Image.fromarray(tile).save(out_img / fname, quality=jpeg_quality)
            rows.append(
                {
                    "tile_file": fname,
                    "source_image": p.name,
                    "full_w": full_w,
                    "full_h": full_h,
                    "crop_x0": crop_x0,
                    "crop_y0": crop_y0,
                    "crop_w": crop_w,
                    "crop_h": crop_h,
                    "tile_x": tx,
                    "tile_y": ty,
                    "tile_size": tile_size,
                    "canopy_frac": round(canopy_frac, 3),
                }
            )
            kept += 1
        print(f"[tile] {p.stem}: kept {kept}/{len(tiles)}")
    with open(out_dir / "tile_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tile] {len(rows)} annotation tiles -> {out_img}")
    return rows
