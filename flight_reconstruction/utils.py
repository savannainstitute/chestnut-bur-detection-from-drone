import pyexiv2
import signal
import atexit
import Metashape
import os
import sys
import psutil
import time

license_obj = None

def find_files(folder, valid_types):
    """
    Find files in a folder with valid extensions.

    Args:
        folder (str): Folder path.
        valid_types (list): List of valid file extensions.

    Returns:
        list: List of file paths.
    """
    try:
        valid_types = [ext.lower() for ext in valid_types]
        return [os.path.join(folder, entry.name)
                for entry in os.scandir(folder)
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_types]
    except Exception as e:
        print(f"Error scanning folder {folder}: {e}")
        return []

def get_utm_zone_from_gps(photo_paths):
    """
    Auto-detect UTM zone from GPS coordinates in first photo.

    Args:
        photo_paths (list): List of photo file paths.

    Returns:
        str: EPSG code string.
    """
    temp_doc = Metashape.Document()
    temp_chunk = temp_doc.addChunk()
    temp_chunk.addPhotos([photo_paths[0]])
    lat = temp_chunk.cameras[0].reference.location.y
    lon = temp_chunk.cameras[0].reference.location.x
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'N' if lat >= 0 else 'S'
    if hemisphere == 'N':
        epsg_code = f"EPSG::{32600 + utm_zone}"
    else:
        epsg_code = f"EPSG::{32700 + utm_zone}"
    print(f"Auto-detected UTM zone: {utm_zone}{hemisphere} ({epsg_code})")
    return epsg_code

def setup_rtk_accuracy(chunk, config):
    """
    Set RTK accuracy using actual XMP accuracy values when available, disable GPS otherwise.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.
        config (dict): Configuration dictionary.
    """
    if not config['gps']['enabled'] or not config['gps']['use_rtk']:
        return
    print("Setting up RTK accuracy...")
    accuracy_from_xmp = 0
    gps_disabled = 0
    for cam in chunk.cameras:
        try:
            meta = pyexiv2.Image(cam.photo.path)
            xmp_data = meta.read_xmp()
            meta.close()
            rtk_std_lon = xmp_data.get('Xmp.drone-dji.RtkStdLon')
            rtk_std_lat = xmp_data.get('Xmp.drone-dji.RtkStdLat') 
            rtk_std_hgt = xmp_data.get('Xmp.drone-dji.RtkStdHgt')
            if rtk_std_lon and rtk_std_lat and rtk_std_hgt:
                cam.reference.location_accuracy = Metashape.Vector([
                    float(rtk_std_lon),
                    float(rtk_std_lat), 
                    float(rtk_std_hgt)
                ])
                cam.reference.accuracy = Metashape.Vector([
                    float(rtk_std_lon),
                    float(rtk_std_lat), 
                    float(rtk_std_hgt)
                ])
                accuracy_from_xmp += 1
            else:
                cam.reference.location_enabled = False
                gps_disabled += 1
        except Exception as e:
            print(f"Warning: RTK setup failed for {cam.label}: {e}")
            cam.reference.location_enabled = False
            gps_disabled += 1
    print(f"RTK setup: {accuracy_from_xmp} cameras with XMP accuracy, {gps_disabled} cameras with GPS disabled")
    chunk.updateTransform()

def reset_region(chunk):
    """
    Reset the region and make it much larger than the points.
    Necessary because if points go outside the region, they get clipped when saving.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.

    Returns:
        bool: True if successful.
    """
    chunk.resetRegion()
    region_dims = chunk.region.size
    region_dims[2] *= 3 # Increase height by 3x
    chunk.region.size = region_dims
    print("Region reset to prevent point clipping.")
    return True

def optimize_camera_rtk(chunk, cam_optimize, progress=None):
    """
    Optimize cameras with configurable adaptive_fitting.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.
        cam_optimize (dict): Camera optimization config.
        progress (callable, optional): Progress callback.
    """
    chunk.optimizeCameras(
        fit_f=cam_optimize['fit_f'],
        fit_cx=cam_optimize['fit_cx'],
        fit_cy=cam_optimize['fit_cy'],
        fit_k1=cam_optimize['fit_k1'],
        fit_k2=cam_optimize['fit_k2'],
        fit_k3=cam_optimize['fit_k3'],
        fit_k4=cam_optimize['fit_k4'],
        fit_p1=cam_optimize['fit_p1'],
        fit_p2=cam_optimize['fit_p2'],
        fit_b1=cam_optimize['fit_b1'],
        fit_b2=cam_optimize['fit_b2'],
        fit_corrections=cam_optimize['fit_corrections'],
        tiepoint_covariance=cam_optimize['tiepoint_covariance'],
        adaptive_fitting=cam_optimize['adaptive_fitting'],
        progress=progress
    )

def filter_tie_points_usgs_part1(chunk, config):
    """
    First stage of USGS point filtering approach - provides better point retention in vegetation.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (ru_thresh, pa_thresh, re_thresh)
    """
    print("Performing USGS-style point filtering (stage 1)...")
    ru_config = config['tie_point_filtering']['reconstruction_uncertainty']
    pa_config = config['tie_point_filtering']['projection_accuracy']
    re_config = config['tie_point_filtering']['reprojection_error']
    cam_optimize = config['camera']['optimize']

    # Filter by reconstruction uncertainty
    fltr = Metashape.TiePoints.Filter()
    fltr.init(chunk, Metashape.TiePoints.Filter.ReconstructionUncertainty)
    values = fltr.values.copy()
    values.sort()
    threshold_index = int(len(values) * (1 - ru_config['percentile'] / 100))
    threshold_index = min(threshold_index, len(values) - 1)
    ru_thresh = values[threshold_index]
    if ru_thresh < ru_config['min_threshold']:
        ru_thresh = ru_config['min_threshold']
    fltr.removePoints(ru_thresh)
    print(f"Removed points with reconstruction uncertainty > {ru_thresh:.1f}")

    optimize_camera_rtk(chunk, cam_optimize, progress=progress_timer.update)

    # Filter by projection accuracy
    fltr = Metashape.TiePoints.Filter()
    fltr.init(chunk, Metashape.TiePoints.Filter.ProjectionAccuracy)
    values = fltr.values.copy()
    values.sort()
    threshold_index = int(len(values) * (1 - pa_config['percentile'] / 100))
    threshold_index = min(threshold_index, len(values) - 1)
    pa_thresh = values[threshold_index]
    if pa_thresh < pa_config['min_threshold']:
        pa_thresh = pa_config['min_threshold']
    fltr.removePoints(pa_thresh)
    print(f"Removed points with projection accuracy > {pa_thresh:.1f}")

    optimize_camera_rtk(chunk, cam_optimize, progress=progress_timer.update)

    # Initial pass of reprojection error filtering
    fltr = Metashape.TiePoints.Filter()
    fltr.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    values = fltr.values.copy()
    values.sort()
    threshold_index = int(len(values) * (1 - re_config['percentile'] / 100))
    threshold_index = min(threshold_index, len(values) - 1)
    re_thresh = values[threshold_index]
    if re_thresh < re_config['min_threshold']:
        re_thresh = re_config['min_threshold']
    fltr.removePoints(re_thresh)
    print(f"Removed points with reprojection error > {re_thresh:.2f}")

    optimize_camera_rtk(chunk, cam_optimize, progress=progress_timer.update)
    print("Stage 1 filtering complete")
    return ru_thresh, pa_thresh, re_thresh

def filter_tie_points_usgs_part2(chunk, config):
    """
    Second stage of USGS point filtering - additional pass for reprojection error.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.
        config (dict): Configuration dictionary.

    Returns:
        float: re_thresh
    """
    print("Performing USGS-style point filtering (stage 2)...")
    re_config = config['tie_point_filtering']['reprojection_error']
    cam_optimize = config['camera']['optimize']

    fltr = Metashape.TiePoints.Filter()
    fltr.init(chunk, Metashape.TiePoints.Filter.ReprojectionError)
    values = fltr.values.copy()
    values.sort()
    threshold_index = int(len(values) * (1 - re_config['percentile'] / 100))
    threshold_index = min(threshold_index, len(values) - 1)
    re_thresh = values[threshold_index]
    if re_thresh < re_config['min_threshold']:
        re_thresh = re_config['min_threshold']
    fltr.removePoints(re_thresh)
    print(f"Second pass: Removed points with reprojection error > {re_thresh:.2f}")

    optimize_camera_rtk(chunk, cam_optimize, progress=progress_timer.update)
    print("Stage 2 filtering complete")
    return re_thresh

def adaptive_subdivide(chunk, config_section_name, config):
    """
    Enable subdivide_task if estimated peak memory > 90% of available physical RAM.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.
        config_section_name (str): Section name in config.
        config (dict): Configuration dictionary.

    Returns:
        dict: Updated config.
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    enabled = [c for c in chunk.cameras if c.enabled]
    if not enabled:
        print(f"[WARN] No enabled cameras for {config_section_name}.")
        return config
    sensor = enabled[0].sensor
    width, height = sensor.width, sensor.height
    mp_per_image = (width * height) / 1e6
    n_images = len(enabled)
    downscale = config[config_section_name].get('downscale', 1)
    neighbors = config[config_section_name].get('max_neighbors', 16)
    est_needed = 0.8 * n_images * (mp_per_image / downscale**2) * (neighbors / 8) / 1000.0
    threshold = total_gb * 0.9  # 90% of available RAM
    if est_needed > threshold:
        print(f"Estimated {est_needed:.0f} GB > {threshold:.0f} GB (90% of RAM). "
              f"Enabling subdivide_task for {config_section_name}.")
        config[config_section_name]['subdivide_task'] = True
    else:
        config[config_section_name]['subdivide_task'] = False
        print(f"Estimated {est_needed:.0f} GB within ({total_gb:.0f} GB available).")
    print(f"{config_section_name}: n_images={n_images}, mp_per_image={mp_per_image:.1f}, "
          f"downscale={downscale}, neighbors={neighbors}, subdivide_task={config[config_section_name]['subdivide_task']}")
    return config

def activate_license():
    """
    Activate license using environment variable.

    Returns:
        Metashape.License or None
    """
    global license_obj
    license_key = os.environ.get('METASHAPE_LICENSE_KEY')
    if not license_key:
        print("Error: METASHAPE_LICENSE_KEY environment variable not set")
        return None
    print("Activating license...")
    license_obj = Metashape.License()
    license_obj.activate(license_key)
    print("License activated successfully")
    return license_obj

def deactivate_license():
    """
    Deactivate license - called by signal handlers and normal exit.
    """
    global license_obj
    if license_obj:
        print("Deactivating license...")
        license_obj.deactivate()
        print("License deactivated successfully")
        license_obj = None

def signal_handler(signum, frame):
    """
    Handle container shutdown signals.
    """
    print(f"Received signal {signum}, deactivating license...")
    deactivate_license()
    sys.exit(0)

def setup_license_cleanup():
    """
    Setup signal handlers and exit cleanup for license.
    """
    signal.signal(signal.SIGTERM, signal_handler)  # Docker stop
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)
    atexit.register(deactivate_license)

def has_valid_transform(chunk):
    """
    Check if chunk has a valid coordinate transform.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        transform = chunk.transform
        if not (transform.scale and transform.rotation and transform.translation):
            return False
        if not transform.scale or transform.scale == 0:
            return False
        return True
    except:
        return False

class ProgressTimer:
    """
    Utility class for progress reporting with estimated time left.
    """
    def __init__(self):
        self.reset()
    def reset(self):
        self.start_time = time.time()
        self.last_printed_percentage = -5
    def update(self, p):
        if p - self.last_printed_percentage >= 5 or p >= 100:
            elapsed = float(time.time() - self.start_time)
            if p > 0:
                remaining_sec = (elapsed / p) * (100 - p)
                print('Progress: {:.0f}%, est. time left: {:.0f} sec'.format(p, remaining_sec))
            else:
                print('Progress: {:.0f}%, est. time left: unknown'.format(p))
            self.last_printed_percentage = p

progress_timer = ProgressTimer()


def deep_merge(base, override):
    import copy
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def detect_sensor(photo_path):
    try:
        img = pyexiv2.Image(photo_path)
        exif = img.read_exif()
        img.close()
        model = (exif.get('Exif.Image.Model') or '').strip().upper().replace(' ', '')
    except Exception as e:
        print(f"[WARN] Could not read EXIF model from {photo_path}: {e}")
        return None
    if 'ZENMUSEP1' in model or model == 'P1':
        return 'p1'
    if 'M3M' in model:
        return 'm3m'
    return None


def resolve_sensor_profile(config, photos):
    name = str(config.get('sensor') or 'auto').lower()
    if name == 'auto':
        name = detect_sensor(photos[0]) if photos else None
    profiles = config.get('sensor_profiles') or {}
    if name and name in profiles:
        print(f"Sensor profile: {name}")
        return deep_merge(config, profiles[name] or {})
    print(f"Sensor profile: baseline (detected={name})")
    return deep_merge(config, {})


def interpolation_from_str(name):
    table = {
        'enabled': Metashape.Interpolation.EnabledInterpolation,
        'extrapolated': Metashape.Interpolation.Extrapolated,
        'disabled': Metashape.Interpolation.DisabledInterpolation,
    }
    return table.get(str(name).lower(), Metashape.Interpolation.EnabledInterpolation)


def surface_type_from_str(name):
    table = {
        'arbitrary': Metashape.SurfaceType.Arbitrary,
        'height_field': Metashape.SurfaceType.HeightField,
    }
    return table.get(str(name).lower(), Metashape.SurfaceType.Arbitrary)


def dem_source_from_str(name):
    table = {
        'model': Metashape.DataSource.ModelData,
        'point_cloud': Metashape.DataSource.PointCloudData,
        'depth_maps': Metashape.DataSource.DepthMapsData,
    }
    return table.get(str(name).lower(), Metashape.DataSource.ModelData)


def get_elevation_by_label(chunk, label):
    for elevation in getattr(chunk, 'elevations', []):
        if getattr(elevation, 'label', '') == label:
            return elevation
    return None


def build_boundary_shape(chunk, epsg_code, buffer_m, label='processing_boundary'):
    from shapely.geometry import MultiPoint

    crs_out = Metashape.CoordinateSystem(epsg_code)
    T = chunk.transform.matrix
    pts = []
    for cam in chunk.cameras:
        if cam.enabled and cam.transform:
            p = crs_out.project(T.mulp(cam.center))
            pts.append((p.x, p.y))
    if len(pts) < 3:
        print("[WARN] Fewer than 3 aligned cameras; boundary shape not built.")
        return False
    hull = MultiPoint(pts).convex_hull.buffer(float(buffer_m))
    coords = list(hull.exterior.coords)
    if chunk.shapes is None:
        chunk.shapes = Metashape.Shapes()
    chunk.shapes.crs = crs_out
    if any(s.label == label for s in chunk.shapes.shapes):
        return True
    shape = chunk.shapes.addShape()
    shape.label = label
    try:
        shape.geometry = Metashape.Geometry.Polygon([Metashape.Vector([x, y]) for x, y in coords])
    except Exception:
        shape.geometry = Metashape.Geometry.Polygon([Metashape.Vector([x, y, 0.0]) for x, y in coords])
    shape.boundary_type = Metashape.Shape.BoundaryType.OuterBoundary
    print(f"Boundary shape built from {len(pts)} cameras, buffer {buffer_m} m")
    return True


def has_boundary_shape(chunk, label='processing_boundary'):
    try:
        return chunk.shapes is not None and any(s.label == label for s in chunk.shapes.shapes)
    except Exception:
        return False


def retry_unaligned_cameras(chunk, align_cfg, progress=None):
    unaligned = [c for c in chunk.cameras if c.enabled and c.transform is None]
    if not unaligned:
        return
    print(f"{len(unaligned)} camera(s) unaligned after first pass; retrying...")
    chunk.alignCameras(cameras=unaligned,
                       adaptive_fitting=align_cfg.get('adaptive_fitting', False),
                       min_image=align_cfg.get('min_image', 2),
                       reset_alignment=False,
                       progress=progress)
    still = [c.label for c in chunk.cameras if c.enabled and c.transform is None]
    print(f"After retry: {len(still)} camera(s) still unaligned" + (": " + ", ".join(still[:10]) if still else ""))


def filter_point_cloud_confidence(chunk, min_confidence, progress=None):
    pc = chunk.point_cloud
    n0 = pc.point_count
    pc.setConfidenceFilter(0, int(min_confidence) - 1)
    pc.removePoints(list(Metashape.PointClass.values.values()), progress=progress)
    pc.resetFilters()
    pc.compactPoints()
    n1 = pc.point_count
    pc.meta['confidence_filter/min_confidence'] = str(int(min_confidence))
    print(f"Confidence filter: removed {n0 - n1:,} of {n0:,} points with confidence < {min_confidence}")
    return n0 - n1


def point_cloud_confidence_filtered(chunk):
    try:
        return bool(chunk.point_cloud) and 'confidence_filter/min_confidence' in chunk.point_cloud.meta
    except Exception:
        return False


def rasters_share_grid(paths):
    try:
        from osgeo import gdal
    except Exception:
        return None
    grids = []
    for path in paths:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        grids.append((ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize))
        ds = None
    ref = grids[0]
    tol = abs(ref[0][1]) * 1e-6
    return all(g[1:] == ref[1:] and all(abs(g[0][k] - ref[0][k]) <= tol for k in range(6)) for g in grids)


def compute_chm_with_gdal(dsm_path, dtm_path, out_path, nodata=-32767.0, block=2048):
    from osgeo import gdal
    import numpy as np
    dsm = gdal.Open(dsm_path, gdal.GA_ReadOnly)
    gt = dsm.GetGeoTransform()
    w, h = dsm.RasterXSize, dsm.RasterYSize
    bounds = (gt[0], gt[3] + gt[5] * h, gt[0] + gt[1] * w, gt[3])
    warped = gdal.Warp('/vsimem/dtm_on_dsm.tif', dtm_path, format='GTiff', outputBounds=bounds,
                       width=w, height=h, resampleAlg='bilinear', dstNodata=nodata)
    out = gdal.GetDriverByName('GTiff').Create(out_path, w, h, 1, gdal.GDT_Float32,
                                               options=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES'])
    out.SetGeoTransform(gt)
    out.SetProjection(dsm.GetProjection())
    ob = out.GetRasterBand(1)
    ob.SetNoDataValue(nodata)
    db, tb = dsm.GetRasterBand(1), warped.GetRasterBand(1)
    dnd = db.GetNoDataValue()
    for y in range(0, h, block):
        rows = min(block, h - y)
        for x in range(0, w, block):
            cols = min(block, w - x)
            a = db.ReadAsArray(x, y, cols, rows).astype(np.float32)
            b = tb.ReadAsArray(x, y, cols, rows).astype(np.float32)
            chm = a - b
            chm[(a == dnd) | (b == nodata) | ~np.isfinite(chm)] = nodata
            ob.WriteArray(chm, x, y)
    ob.FlushCache()
    out.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64])
    out = None
    warped = None
    gdal.Unlink('/vsimem/dtm_on_dsm.tif')
