"""Select the best raw image for each segmented canopy polygon."""

import easyidp as idp
import geopandas as gpd
import os
import json
import numpy as np
import pandas as pd
from skimage import io, color, filters, draw
import pyexiv2
import argparse


def gimbal_pitch(img_path):
    """
    Extract the gimbal pitch angle from image XMP metadata.

    Returns NaN if unavailable.
    """
    try:
        img = pyexiv2.Image(img_path)
        xmp = img.read_xmp()
        if "Xmp.drone-dji.GimbalPitchDegree" in xmp:
            return float(xmp["Xmp.drone-dji.GimbalPitchDegree"])
    except Exception:
        pass
    return np.nan


def sharpness_contrast_exposure(img_path, poly):
    """
    Compute sharpness, contrast, and exposure inside the given polygon.

    Returns (sharpness, contrast, exposure).
    """
    try:
        img = io.imread(img_path)
        if img.ndim == 3:
            img_gray = color.rgb2gray(img)
        else:
            img_gray = img.astype(np.float32) / 255.0
        mask = np.zeros(img_gray.shape, dtype=bool)
        rr, cc = draw.polygon(poly[:, 1], poly[:, 0], img_gray.shape)
        mask[rr, cc] = True
        if not np.any(mask):
            region = img_gray
            lap = filters.laplace(img_gray)
            sharpness = float(lap.var())
        else:
            region = img_gray[mask]
            lap = filters.laplace(img_gray)
            sharpness = float(lap[mask].var())
        contrast = float(region.std())
        exposure = float(region.mean())
        return sharpness, contrast, exposure
    except Exception:
        return np.nan, np.nan, np.nan


def select_best_image(image_paths, polygons):
    """
    Select the best image for a canopy from sharpness and metadata.

    Considers sharpness, contrast, exposure, and gimbal pitch.
    Returns (best_image_path, best_polygon).
    """
    data = []
    for img_path, poly in zip(image_paths, polygons):
        sharp, cont, exp = sharpness_contrast_exposure(img_path, poly)
        pitch = gimbal_pitch(img_path)
        data.append(
            {
                "img_path": img_path,
                "polygon": poly,
                "sharpness": sharp,
                "contrast": cont,
                "exposure": exp,
                "pitch": pitch,
            }
        )
    df = pd.DataFrame(data)
    df_filt = df[(df["exposure"] > 0.15) & (df["exposure"] < 0.85)]
    if df_filt.empty:
        df_filt = df

    nadir = df_filt[
        df_filt["pitch"].notnull() & (np.abs(df_filt["pitch"] + 90) < 5)
    ]
    if not nadir.empty:
        df_filt = nadir

    best_sharp = df_filt["sharpness"].max()
    close = df_filt[df_filt["sharpness"] >= 0.85 * best_sharp]
    if len(close) > 1:
        best = close.loc[close["contrast"].idxmax()]
    else:
        best = df_filt.loc[df_filt["sharpness"].idxmax()]

    return best["img_path"], best["polygon"]


def run_canopy_to_image():
    """
    Select the best image for each canopy polygon.

    Uses 3D geometry and image metadata. Saves a mapping of
    tree_id -> {best_image_path, polygon_coordinates}.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Select best images for each canopy using 3D geometry "
            "and image metadata."
        )
    )
    parser.add_argument(
        "--canopy_shapefile",
        type=str,
        required=True,
        help="Path to canopy polygons shapefile (.shp)",
    )
    parser.add_argument(
        "--dsm", type=str, required=True, help="Path to DSM raster (.tif)"
    )
    parser.add_argument(
        "--metashape_project",
        type=str,
        required=True,
        help="Path to Metashape project file (.psx)",
    )
    parser.add_argument(
        "--raw_images",
        type=str,
        required=True,
        help="Path to folder containing raw images",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output folder"
    )
    args = parser.parse_args()

    shapefile_path = args.canopy_shapefile
    dsm_path = args.dsm
    metashape_path = args.metashape_project
    raw_images_folder_path = args.raw_images
    output_folder_path = args.output

    # Convert shapefile to geojson
    gdf = gpd.read_file(shapefile_path)
    geojson_path = shapefile_path.replace(".shp", ".geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")

    # Load canopy polygons
    roi = idp.ROI()
    roi.read_geojson(geojson_path, name_field="tree_id")

    # Load DSM and add z values to polygons
    dsm = idp.GeoTiff(dsm_path)
    roi.get_z_from_dsm(dsm)

    # Read 3D reconstruction project from Metashape
    ms = idp.Metashape(project_path=metashape_path, chunk_id=0)

    # Back-project polygons onto raw images
    img_dict_ms = roi.back2raw(ms)

    # Sort images by distance to ROI, keep best 3 per canopy
    img_dict_sort = ms.sort_img_by_distance(
        img_dict_ms, roi, distance_thresh=10, num=3
    )

    # Process each tree to find the single best image and polygon
    best_selections = {}
    canopy_ids = list(img_dict_sort.keys())

    print(f"Processing {len(canopy_ids)} canopies...")

    for i, tree_id in enumerate(canopy_ids):
        if i % 10 == 0:
            print(
                f"Progress: {i}/{len(canopy_ids)} "
                f"({100 * i / len(canopy_ids):.1f}%)"
            )

        images = list(img_dict_sort[tree_id].keys())
        polygons = list(img_dict_sort[tree_id].values())
        image_paths = [
            os.path.join(
                os.path.dirname(raw_images_folder_path), img_rel_path
            ).replace("\\", "/")
            for img_rel_path in images
        ]

        best_img_path, best_polygon = select_best_image(image_paths, polygons)

        # Convert paths and polygons to serializable formats
        if hasattr(best_img_path, "iloc"):
            best_img_path_str = str(best_img_path.iloc[0]).replace("\\", "/")
        else:
            best_img_path_str = str(best_img_path).replace("\\", "/")

        polygon_coords = (
            best_polygon.tolist()
            if hasattr(best_polygon, "tolist")
            else best_polygon
        )

        best_selections[tree_id] = {
            "image_path": best_img_path_str,
            "polygon_coords": polygon_coords,
        }

    # Save best image and canopy mapping to JSON
    os.makedirs(output_folder_path, exist_ok=True)
    json_path = os.path.join(output_folder_path, "best_image_selections.json")
    with open(json_path, "w") as f:
        json.dump(best_selections, f, indent=2)

    print(f"\nBest selections saved to: {json_path}")
    print(
        f"Selected {len(best_selections)} trees with their best "
        f"images and polygons"
    )


if __name__ == "__main__":
    run_canopy_to_image()
