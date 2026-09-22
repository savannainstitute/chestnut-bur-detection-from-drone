"""Segment tree canopies from a CHM raster via marker-controlled watershed."""

import os
import argparse

import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
from rasterio import mask as rio_mask
from rasterio.transform import rowcol, xy
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import gaussian
import geopandas as gpd
from shapely.geometry import shape, Point


class TreeCanopySegmentation:
    """
    Proximity-based segmentation of tree canopies from a CHM raster.

    Steps:
        1. Load CHM raster (optionally crop to extent)
        2. Refine tree top points to local maxima within a buffer
        3. Adaptive watershed segmentation using tree tops
        4. Save results as shapefiles (canopy polygons and refined tree tops)
    """

    def __init__(
        self, chm_path, min_height, min_area_m2=5.0, min_hole_area_m2=1.25
    ):
        """
        Initialize segmentation parameters and state.

        Args:
            chm_path (str): Path to input CHM raster file.
            min_height (float): Minimum CHM height (meters) for
                segmentation mask.
            min_area_m2 (float): Minimum canopy area (m^2); smaller
                segments are removed.
            min_hole_area_m2 (float): Holes within a segment smaller
                than this (m^2) are filled. Independent of min_area_m2.

        Returns:
            None
        """
        self.penalty_strength = (
            0.1  # lower if smaller trees take up too much space
        )
        self.gradient_weight = (
            0.4  # higher to follow steep slopes more closely
        )
        self.watershed_compactness = 1e-4  # higher for more regular shapes
        self.height_factor_scale = (
            0.2  # higher places boundaries further out for taller trees
        )
        self.min_height = (
            min_height  # minimum height in meters for segmentation mask
        )
        self.min_area_m2 = (
            min_area_m2  # minimum canopy area in m^2; smaller segments removed
        )
        self.min_hole_area_m2 = (
            # holes smaller than this (m^2) are filled; decoupled from
            # min_area
            min_hole_area_m2
        )
        self.surface_smooth_sigma = (
            0.1  # higher for smoother segmentation surface
        )

        self.chm_path = chm_path
        self.chm_data = None
        self.original_chm = None
        self.chm_profile = None
        self.tree_markers = None
        self.segments = None
        self.resolution_m_per_pixel = None
        self.extent_gdf = None

        # Maps marker label to original tree ID
        self.marker_id_to_original_id = {}

    def load_chm(self, extent_shapefile=None):
        """
        Load CHM raster, optionally cropping to a shapefile extent.

        Args:
            extent_shapefile (str, optional): Path to extent shapefile.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        try:
            with rasterio.open(self.chm_path) as src:
                if extent_shapefile is not None:
                    gdf = gpd.read_file(extent_shapefile)
                    if gdf.crs != src.crs:
                        gdf = gdf.to_crs(src.crs)
                    self.extent_gdf = gdf
                    geoms = [geom for geom in gdf.geometry]
                    arr, out_transform = rio_mask.mask(
                        src, geoms, crop=True, nodata=np.nan
                    )
                    profile = src.profile.copy()
                    profile.update(
                        {
                            "height": arr.shape[1],
                            "width": arr.shape[2],
                            "transform": out_transform,
                            "nodata": np.nan,
                        }
                    )
                    data = arr[0].astype(np.float64)
                    self.chm_profile = profile
                else:
                    data = src.read(1).astype(np.float64)
                    self.chm_profile = src.profile.copy()
                    self.extent_gdf = None

                nodata = self.chm_profile.get("nodata", None)
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)

                self.original_chm = data.copy()
                self.chm_data = data.copy()

                transform = self.chm_profile["transform"]
                px_w = abs(transform.a)
                px_h = abs(transform.e)
                self.resolution_m_per_pixel = float((px_w + px_h) / 2.0)

                print(
                    f"Loaded CHM: shape={self.chm_data.shape}, "
                    f"resolution={self.resolution_m_per_pixel:.4f} m/px"
                )
                return True

        except Exception as e:
            print(f"Error loading CHM: {e}")
            return False

    def meters_to_pixels(self, distance_meters):
        """
        Convert a distance in meters to pixels using raster resolution.

        Args:
            distance_meters (float): Distance in meters.

        Returns:
            int: Distance in pixels (rounded).
        """
        return int(round(distance_meters / self.resolution_m_per_pixel))

    def load_tree_markers_from_shapefile(
        self, shapefile_path, buffer_meters, id_column=None
    ):
        """
        Refine tree top points to local maxima in the CHM.

        Creates a marker image from the refined tree top locations.

        Args:
            shapefile_path (str): Path to input marker shapefile.
            buffer_meters (float): Buffer radius in meters for local
                maxima search.
            id_column (str, optional): Name of the column containing
                original tree IDs.

        Returns:
            tuple or None: (rows, cols) of refined points, or None if failed.
        """
        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs != self.chm_profile["crs"]:
                gdf = gdf.to_crs(self.chm_profile["crs"])

            # Determine which column to use for original IDs
            if id_column is None:
                id_column = next(
                    (col for col in gdf.columns if col != "geometry"), None
                )
            if id_column is None:
                raise ValueError(
                    "No suitable ID column found in marker shapefile."
                )

            transform = self.chm_profile["transform"]
            refined = []
            skipped = 0
            base_buf_px = max(1, self.meters_to_pixels(buffer_meters))
            original_ids = []
            refined_indices = []

            for idx, row in gdf.iterrows():
                pt = row.geometry
                r, c = rowcol(transform, pt.x, pt.y)
                r = int(r)
                c = int(c)

                if not (
                    0 <= r < self.chm_data.shape[0]
                    and 0 <= c < self.chm_data.shape[1]
                ):
                    skipped += 1
                    continue

                rmin = max(0, r - base_buf_px)
                rmax = min(self.original_chm.shape[0], r + base_buf_px + 1)
                cmin = max(0, c - base_buf_px)
                cmax = min(self.original_chm.shape[1], c + base_buf_px + 1)
                window = self.original_chm[rmin:rmax, cmin:cmax]

                finite_mask = np.isfinite(window)
                if not np.any(finite_mask):
                    skipped += 1
                    continue

                local = np.where(finite_mask, window, -np.inf)
                max_idx = np.argmax(local)
                max_local_idx = np.unravel_index(max_idx, local.shape)
                max_r = rmin + max_local_idx[0]
                max_c = cmin + max_local_idx[1]

                refined.append((max_r, max_c))
                original_ids.append(row[id_column])
                refined_indices.append(idx)

            if len(refined) == 0:
                print("No valid refined tree tops found.")
                return None

            markers = np.zeros(self.chm_data.shape, dtype=np.int32)
            self.marker_id_to_original_id = {}
            for i, (rr, cc) in enumerate(refined):
                markers[rr, cc] = i + 1
                self.marker_id_to_original_id[i + 1] = original_ids[i]

            self.tree_markers = markers
            rows = np.array([p[0] for p in refined], dtype=int)
            cols = np.array([p[1] for p in refined], dtype=int)
            print(f"Loaded {len(refined)} tree tops (skipped {skipped})")
            print(f"Minimum height threshold: {self.min_height:.3f} m")
            return (rows, cols)

        except Exception as e:
            print(f"Error loading tree tops shapefile: {e}")
            return None

    def watershed_segment(self):
        """
        Segment the CHM into tree crowns using adaptive watershed.

        Uses tree markers as seeds and computes a proximity-based
        penalty surface that adapts to local tree spacing and height.
        Efficiently caches pairwise distances and path checks to avoid
        redundant work. Cleans up small or spurious segments after
        segmentation.

        Returns:
            bool: True if segmentation succeeded, False otherwise.
        """
        if self.tree_markers is None:
            print("No markers available for watershed.")
            return False

        threshold = self.min_height if self.min_height is not None else 0
        mask = np.isfinite(self.chm_data) & (self.chm_data > threshold)

        if not np.any(mask):
            print(
                "No valid CHM pixels to segment (check minimum height "
                "threshold)."
            )
            return False

        marker_ids = np.unique(self.tree_markers)
        marker_ids = marker_ids[marker_ids > 0]

        marker_positions = [
            np.where(self.tree_markers == marker_id)
            for marker_id in marker_ids
        ]
        marker_positions = [
            (pos[0][0], pos[1][0])
            for pos in marker_positions
            if len(pos[0]) > 0
        ]
        marker_heights = [self.original_chm[r, c] for r, c in marker_positions]
        marker_positions = np.array(marker_positions)
        marker_heights = np.array(marker_heights)

        smoothed_chm = gaussian(
            self.chm_data, sigma=self.surface_smooth_sigma, preserve_range=True
        )

        grad_y, grad_x = np.gradient(smoothed_chm)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mag = np.where(np.isfinite(gradient_mag), gradient_mag, 0)
        height_range = np.nanmax(smoothed_chm) - np.nanmin(
            smoothed_chm[np.isfinite(smoothed_chm)]
        )
        proximity_penalty = np.full_like(smoothed_chm, np.inf)

        def path_crosses_low_height(start_r, start_c, end_r, end_c):
            from skimage.draw import line

            line_r, line_c = line(start_r, start_c, end_r, end_c)
            for lr, lc in zip(line_r, line_c):
                if (
                    0 <= lr < self.chm_data.shape[0]
                    and 0 <= lc < self.chm_data.shape[1]
                ):
                    val = self.chm_data[lr, lc]
                    if (
                        np.isfinite(val)
                        and self.min_height is not None
                        and val < self.min_height
                    ):
                        return True
            return False

        num_markers = len(marker_positions)
        max_search_distance = 25.0
        distance_matrix = np.full((num_markers, num_markers), np.inf)
        path_cross_matrix = np.zeros((num_markers, num_markers), dtype=bool)

        pos_r = marker_positions[:, 0][:, np.newaxis]
        pos_c = marker_positions[:, 1][:, np.newaxis]
        dists_px = np.sqrt((pos_r - pos_r.T) ** 2 + (pos_c - pos_c.T) ** 2)
        dists_m = dists_px * self.resolution_m_per_pixel
        distance_matrix = dists_m

        for i in range(num_markers):
            for j in range(i + 1, num_markers):
                if distance_matrix[i, j] <= max_search_distance:
                    crosses = path_crosses_low_height(
                        marker_positions[i][0],
                        marker_positions[i][1],
                        marker_positions[j][0],
                        marker_positions[j][1],
                    )
                    path_cross_matrix[i, j] = path_cross_matrix[j, i] = crosses

        for i, marker_id in enumerate(marker_ids):
            marker_pos = marker_positions[i]
            marker_height = marker_heights[i]

            valid = (
                (np.arange(num_markers) != i)
                & (distance_matrix[i] <= max_search_distance)
                & (~path_cross_matrix[i])
            )
            neighbor_indices = np.where(valid)[0]
            if neighbor_indices.size > 0:
                neighbor_distances = distance_matrix[i, neighbor_indices]
                nearest_idx_in_neighbors = np.argmin(neighbor_distances)
                nearest_idx = neighbor_indices[nearest_idx_in_neighbors]
                nearest_distance = neighbor_distances[nearest_idx_in_neighbors]
                nearest_neighbor_height = marker_heights[nearest_idx]

                if nearest_neighbor_height == 0:
                    height_ratio = 1.0
                else:
                    height_ratio = marker_height / nearest_neighbor_height

                height_factor = 0.8 + self.height_factor_scale * np.clip(
                    height_ratio, 0.5, 1.5
                )

                local_characteristic_distance = (
                    nearest_distance / 2.0
                ) * height_factor

                print(
                    f"Tree {
                        self.marker_id_to_original_id.get(marker_id, marker_id)
                    }: height={marker_height:.1f}m, "
                    f"neighbor_dist={nearest_distance:.1f}m, "
                    f"neighbor_height={nearest_neighbor_height:.1f}m, "
                    f"height_factor={height_factor:.2f}, "
                    f"boundary_dist={local_characteristic_distance:.1f}m"
                )

                marker_mask = self.tree_markers == marker_id
                buf_px = (
                    int(
                        np.ceil(
                            (nearest_distance / 2.0)
                            / self.resolution_m_per_pixel
                        )
                    )
                    + 10
                )
                r, c = marker_pos
                rmin = max(0, r - buf_px)
                rmax = min(self.chm_data.shape[0], r + buf_px + 1)
                cmin = max(0, c - buf_px)
                cmax = min(self.chm_data.shape[1], c + buf_px + 1)
                local_marker_mask = marker_mask[rmin:rmax, cmin:cmax]
                distance_from_this_marker = ndimage.distance_transform_edt(
                    ~local_marker_mask
                )
                distance_m = (
                    distance_from_this_marker * self.resolution_m_per_pixel
                )

                local_penalty = (
                    self.penalty_strength
                    * height_range
                    * (1 - np.exp(-distance_m / local_characteristic_distance))
                )

                proximity_penalty[rmin:rmax, cmin:cmax] = np.minimum(
                    proximity_penalty[rmin:rmax, cmin:cmax], local_penalty
                )
            else:
                print(
                    f"Tree {
                        self.marker_id_to_original_id.get(marker_id, marker_id)
                    }: isolated, using natural watershed boundaries"
                )
                continue

        proximity_penalty = np.where(
            np.isinf(proximity_penalty),
            self.penalty_strength * height_range,
            proximity_penalty,
        )

        inv_height = np.where(np.isfinite(smoothed_chm), -smoothed_chm, 0.0)
        surface = (
            inv_height
            + proximity_penalty
            + (self.gradient_weight * gradient_mag)
        )

        self.segments = watershed(
            surface,
            self.tree_markers,
            connectivity=2,
            compactness=self.watershed_compactness,
            mask=mask,
        )

        n_segments_before = len(np.unique(self.segments)) - 1
        min_area_m2 = self.min_area_m2
        min_size_pixels = int(min_area_m2 / (self.resolution_m_per_pixel**2))
        hole_size_pixels = int(
            self.min_hole_area_m2 / (self.resolution_m_per_pixel**2)
        )

        cleaned_segments = np.zeros_like(self.segments)
        for segment_id in np.unique(self.segments):
            if segment_id == 0:
                continue
            segment_mask = self.segments == segment_id
            cleaned_mask = remove_small_objects(
                segment_mask, min_size=min_size_pixels
            )
            cleaned_mask = remove_small_holes(
                cleaned_mask, area_threshold=hole_size_pixels
            )
            cleaned_segments[cleaned_mask] = segment_id

        self.segments = cleaned_segments
        n_segments_after = len(np.unique(self.segments)) - 1

        print(
            f"Adaptive watershed produced {n_segments_before} segments, "
            f"cleaned to {n_segments_after} segments"
        )

        if self.extent_gdf is not None:
            self._remove_boundary_segments()

        return True

    def _remove_boundary_segments(self):
        """
        Remove segments that overlap the raster boundary past a threshold.

        Returns:
            None
        """
        max_border_overlap_m = 0.5
        max_border_pixels = int(
            max_border_overlap_m / self.resolution_m_per_pixel
        )
        boundary_geoms = []
        for geom in self.extent_gdf.geometry:
            if hasattr(geom, "exterior"):
                boundary_geoms.append(geom.exterior)
            elif hasattr(geom, "geoms"):
                for sub_geom in geom.geoms:
                    boundary_geoms.append(sub_geom.exterior)
        if not boundary_geoms:
            return
        boundary_mask = rasterize(
            boundary_geoms,
            out_shape=self.segments.shape,
            transform=self.chm_profile["transform"],
            fill=0,
            default_value=1,
            dtype=np.uint8,
        ).astype(bool)
        segments_to_remove = []
        for seg_id in np.unique(self.segments):
            if seg_id == 0:
                continue
            segment_mask = self.segments == seg_id
            overlap_pixels = np.sum(segment_mask & boundary_mask)
            if overlap_pixels > max_border_pixels:
                segments_to_remove.append(seg_id)
        for seg_id in segments_to_remove:
            self.segments[self.segments == seg_id] = 0
        print(f"Removed {len(segments_to_remove)} boundary segments")

    @staticmethod
    def _remove_shapefile_if_exists(path_shp):
        """
        Remove all files associated with a shapefile if they exist.

        Args:
            path_shp (str): Path to the .shp file.

        Returns:
            None
        """
        base, _ = os.path.splitext(path_shp)
        exts = [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx"]
        for e in exts:
            p = base + e
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def save_refined_tree_tops(self, output_dir, prefix):
        """
        Save the refined tree top points as a shapefile.

        Args:
            output_dir (str): Output directory.
            prefix (str): Output filename prefix.

        Returns:
            None
        """
        if self.tree_markers is None:
            return
        marker_coords = np.where(self.tree_markers > 0)
        tree_ids = self.tree_markers[marker_coords]
        points = []
        heights = []
        original_ids = [
            self.marker_id_to_original_id.get(tid, tid) for tid in tree_ids
        ]
        transform = self.chm_profile["transform"]
        for r, c, _ in zip(marker_coords[0], marker_coords[1], tree_ids):
            x, y = xy(transform, int(r), int(c), offset="center")
            points.append(Point(x, y))
            heights.append(
                self.original_chm[r, c]
                if np.isfinite(self.original_chm[r, c])
                else np.nan
            )
        gdf = gpd.GeoDataFrame(
            {"tree_id": original_ids, "height": heights, "geometry": points},
            crs=self.chm_profile["crs"],
        )
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{prefix}_Treetops.shp")
        self._remove_shapefile_if_exists(out_path)
        gdf.to_file(out_path)
        print(f"Saved refined treetops: {out_path}")

    def save_results(self, output_dir, prefix):
        """
        Save segmented canopy polygons as a shapefile with statistics.

        Args:
            output_dir (str): Output directory.
            prefix (str): Output filename prefix.

        Returns:
            None
        """
        if self.segments is None:
            print("No segments to save.")
            return
        os.makedirs(output_dir, exist_ok=True)
        polygons = []
        labels = []
        transform = self.chm_profile["transform"]
        for geom, val in shapes(
            self.segments.astype(np.int32), transform=transform
        ):
            if val == 0:
                continue
            polygons.append(shape(geom))
            labels.append(int(val))
        if len(polygons) == 0:
            print("No polygons generated from segments.")
            return
        pix_area = self.resolution_m_per_pixel**2
        stats = []
        for lbl in labels:
            mask = self.segments == lbl
            area_m2 = np.sum(mask) * pix_area
            heights = self.original_chm[mask]
            max_h = (
                np.nanmax(heights) if np.any(np.isfinite(heights)) else np.nan
            )
            mean_h = (
                np.nanmean(heights) if np.any(np.isfinite(heights)) else np.nan
            )
            stats.append((area_m2, max_h, mean_h))
        tree_ids = [
            self.marker_id_to_original_id.get(lbl, lbl) for lbl in labels
        ]
        gdf = gpd.GeoDataFrame(
            {
                "tree_id": tree_ids,
                "geometry": polygons,
                "area_m2": [s[0] for s in stats],
                "max_h": [s[1] for s in stats],
                "mean_h": [s[2] for s in stats],
            },
            crs=self.chm_profile["crs"],
        )
        out_path = os.path.join(output_dir, f"{prefix}_Canopies.shp")
        self._remove_shapefile_if_exists(out_path)
        gdf.to_file(out_path)
        print(f"Saved canopy polygons: {out_path} ({len(gdf)} features)")


def prefix_from_chm(chm_path):
    """
    Get a filename prefix from a CHM raster path.

    Args:
        chm_path (str): Path to the CHM raster.

    Returns:
        str: Prefix for output files.
    """
    name = os.path.basename(chm_path)
    if name.lower().endswith("_chm.tif"):
        return name[:-8]
    elif name.lower().endswith(".tif"):
        return name[:-4]
    else:
        return os.path.splitext(name)[0]


def run_segmentation():
    """
    Command-line entry point for proximity-based canopy segmentation.

    Steps:
        1. Load CHM raster (optionally crop to extent)
        2. Refine tree top points to local maxima within a buffer
        3. Adaptive watershed segmentation using tree tops
        4. Save results as shapefiles (canopy polygons and refined tree tops)

    Command-line Args:
        --chm (str): Path to CHM TIFF
        --tree-markers (str): Point shapefile of tree markers
        --min-height (float): Minimum CHM height for segmentation mask
            (default: 1.0)
        --buffer-size (float): Buffer size in meters for tree marker
            refinement (default: 0.5)
        --extent (str, optional): Polygon shapefile defining processing extent
        --outdir (str): Output directory

    Returns:
        int: 0 if successful, 1 if error
    """
    parser = argparse.ArgumentParser(
        description="Proximity-based canopy segmentation"
    )
    parser.add_argument("--chm", required=True, help="Path to CHM TIFF")
    parser.add_argument(
        "--tree-markers",
        "-t",
        required=True,
        help="Point shapefile of tree markers",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=1.0,
        help=(
            "Minimum CHM height (meters) for segmentation mask (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--buffer-size",
        type=float,
        default=0.5,
        help="Buffer size in meters for tree marker refinement (default: 0.5)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=5.0,
        help=(
            "Minimum canopy area in m^2; smaller segments are "
            "removed (default: 5.0)"
        ),
    )
    parser.add_argument(
        "--min-hole-area",
        type=float,
        default=1.25,
        help=(
            "Holes within a segment smaller than this (m^2) are "
            "filled; independent of --min-area (default: 1.25)"
        ),
    )
    parser.add_argument(
        "--extent",
        "-e",
        required=False,
        help="Polygon shapefile defining processing extent (optional)",
    )
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help=(
            "Column name for original tree IDs in marker shapefile (optional)"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.chm):
        print("CHM not found.")
        return 1
    if not os.path.exists(args.tree_markers):
        print("Tree markers shapefile not found.")
        return 1
    if args.extent is not None and not os.path.exists(args.extent):
        print("Extent shapefile not found.")
        return 1

    prefix = prefix_from_chm(args.chm)
    outdir = args.outdir

    seg = TreeCanopySegmentation(
        args.chm,
        min_height=args.min_height,
        min_area_m2=args.min_area,
        min_hole_area_m2=args.min_hole_area,
    )

    if not seg.load_chm(extent_shapefile=args.extent):
        return 1

    coords = seg.load_tree_markers_from_shapefile(
        args.tree_markers,
        buffer_meters=args.buffer_size,
        id_column=args.id_column,
    )

    if coords is None:
        print("No valid refined tree tops found; exiting.")
        return 1

    if not seg.watershed_segment():
        return 1

    seg.save_results(outdir, prefix)
    seg.save_refined_tree_tops(outdir, prefix)
    return 0


if __name__ == "__main__":
    """
    Main entry point for running the segmentation from the command line.
    """
    import sys

    sys.exit(run_segmentation())
