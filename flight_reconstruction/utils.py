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


# ---------------------------------------------------------------------------
# Sensor profiles and config handling
# ---------------------------------------------------------------------------

def deep_merge(base, override):
    """
    Recursively merge `override` into a copy of `base`. Dicts merge key-wise; any other value replaces.

    Args:
        base (dict): Baseline configuration.
        override (dict): Overrides.

    Returns:
        dict: New merged dictionary (inputs are not modified).
    """
    import copy
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def detect_sensor(photo_path):
    """
    Detect the sensor profile name from the EXIF camera model of one photo.

    Args:
        photo_path (str): Path to an image.

    Returns:
        tuple: (profile_name or None, raw EXIF model string or None)
    """
    model = None
    try:
        img = pyexiv2.Image(photo_path)
        exif = img.read_exif()
        img.close()
        model = (exif.get('Exif.Image.Model') or '').strip()
    except Exception as e:
        print(f"[WARN] Could not read EXIF model from {photo_path}: {e}")
        return None, None
    upper = model.upper().replace(' ', '')
    if 'ZENMUSEP1' in upper or upper == 'P1':
        return 'p1', model
    if 'M3M' in upper:
        return 'm3m', model
    return None, model


def resolve_sensor_profile(config, sensor_arg, photos):
    """
    Apply the sensor profile (from --sensor, config['sensor'], or EXIF auto-detection) over the baseline config.

    Args:
        config (dict): Loaded YAML config.
        sensor_arg (str or None): Value of --sensor (overrides config['sensor']).
        photos (list): Image paths (first one is used for auto-detection).

    Returns:
        dict: Merged config with a `_resolved` block recording what was applied.
    """
    requested = (sensor_arg or config.get('sensor') or 'auto').lower()
    exif_model = None
    if requested == 'auto':
        name, exif_model = detect_sensor(photos[0]) if photos else (None, None)
        if name is None:
            print(f"[WARN] Sensor auto-detection failed (EXIF model={exif_model!r}); using baseline config.")
            name = 'baseline'
    else:
        name = requested
    profiles = config.get('sensor_profiles') or {}
    if name in profiles:
        merged = deep_merge(config, profiles[name] or {})
    else:
        if name != 'baseline':
            print(f"[WARN] No sensor profile named {name!r} in config; using baseline config.")
        merged = deep_merge(config, {})
    merged['_resolved'] = {
        'sensor_profile': name,
        'sensor_requested': requested,
        'exif_model': exif_model,
        'metashape_version': Metashape.app.version,
    }
    print(f"Sensor profile: {name} (requested={requested}, EXIF model={exif_model!r})")
    return merged


def write_config_used(config, path):
    """
    Write the fully merged configuration actually used for a run.
    """
    import yaml
    try:
        with open(path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"Config used written to {path}")
    except Exception as e:
        print(f"[WARN] Could not write config used: {e}")


def read_image_list(list_path, folder):
    """
    Read an image list (one filename per line, relative to `folder` or absolute).

    Returns:
        list: Absolute paths that exist. Missing entries are reported.
    """
    wanted = []
    with open(list_path, 'r') as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith('#'):
                continue
            wanted.append(name)
    paths, missing = [], []
    for name in wanted:
        p = name if os.path.isabs(name) else os.path.join(folder, name)
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing.append(name)
    print(f"Image list {list_path}: {len(wanted)} listed, {len(paths)} found, {len(missing)} missing")
    if missing:
        print("  Missing (first 10): " + ", ".join(missing[:10]))
    return paths


# ---------------------------------------------------------------------------
# Enum helpers
# ---------------------------------------------------------------------------

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
        'heightfield': Metashape.SurfaceType.HeightField,
    }
    return table.get(str(name).lower(), Metashape.SurfaceType.Arbitrary)


def face_count_from_str(name):
    table = {
        'high': Metashape.FaceCount.HighFaceCount,
        'medium': Metashape.FaceCount.MediumFaceCount,
        'low': Metashape.FaceCount.LowFaceCount,
    }
    return table.get(str(name).lower(), Metashape.FaceCount.HighFaceCount)


def dem_source_from_str(name):
    table = {
        'model': Metashape.DataSource.ModelData,
        'mesh': Metashape.DataSource.ModelData,
        'point_cloud': Metashape.DataSource.PointCloudData,
        'depth_maps': Metashape.DataSource.DepthMapsData,
    }
    return table.get(str(name).lower(), Metashape.DataSource.ModelData)


def make_compression(section):
    """
    Build a Metashape.ImageCompression from a config section with tiff_* keys.
    """
    comp = Metashape.ImageCompression()
    comp.tiff_big = bool(section.get('tiff_big', True))
    comp.tiff_tiled = bool(section.get('tiff_tiled', False))
    comp.tiff_overviews = bool(section.get('tiff_overviews', True))
    table = {
        'lzw': Metashape.ImageCompression.TiffCompressionLZW,
        'deflate': Metashape.ImageCompression.TiffCompressionDeflate,
        'jpeg': Metashape.ImageCompression.TiffCompressionJPEG,
        'packbits': Metashape.ImageCompression.TiffCompressionPackbits,
        'none': Metashape.ImageCompression.TiffCompressionNone,
    }
    comp.tiff_compression = table.get(str(section.get('tiff_compression', 'lzw')).lower(),
                                      Metashape.ImageCompression.TiffCompressionLZW)
    if 'jpeg_quality' in section:
        comp.jpeg_quality = int(section['jpeg_quality'])
    return comp


def get_elevation_by_label(chunk, label):
    for elevation in getattr(chunk, 'elevations', []):
        if getattr(elevation, 'label', '') == label:
            return elevation
    return None


# ---------------------------------------------------------------------------
# Region and boundary
# ---------------------------------------------------------------------------

def _crs_aligned_transform(chunk):
    """
    Matrix mapping chunk-internal coordinates to a CRS-aligned local frame (metres) at the region centre,
    plus the internal-to-metre scale factor.
    """
    import math
    T = chunk.transform.matrix
    centre_world = T.mulp(chunk.region.center)
    m = chunk.crs.localframe(centre_world) * T
    s = math.sqrt(m[0, 0] ** 2 + m[0, 1] ** 2 + m[0, 2] ** 2)
    return m, s


def set_region_from_cameras(chunk, cfg):
    """
    Set the reconstruction region from aligned camera positions (XY + buffer) and a ground estimate (Z),
    aligned to the chunk CRS. Falls back to the resetRegion() result if too few cameras are aligned.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.
        cfg (dict): config['region'].

    Returns:
        bool: True if the camera-based region was applied.
    """
    chunk.resetRegion()
    m, s = _crs_aligned_transform(chunk)
    cams = [m.mulp(c.center) for c in chunk.cameras if c.enabled and c.transform]
    if len(cams) < 3:
        print("[WARN] Fewer than 3 aligned cameras; keeping resetRegion() result.")
        return False
    xs = [p.x for p in cams]
    ys = [p.y for p in cams]
    zs = [p.z for p in cams]

    # Ground estimate from a sample of valid tie points (2nd / 98th percentile of Z)
    tz = []
    pts = chunk.tie_points.points if chunk.tie_points else []
    n = len(pts)
    step = max(1, n // 20000)
    for i in range(0, n, step):
        p = pts[i]
        if p.valid:
            c = p.coord
            tz.append(m.mulp(Metashape.Vector([c.x, c.y, c.z])).z)
    tz.sort()
    if tz:
        ground_low = tz[int(0.02 * (len(tz) - 1))]
        ground_high = tz[int(0.98 * (len(tz) - 1))]
    else:
        ground_low = min(zs) - 60.0
        ground_high = min(zs) - 20.0
        print("[WARN] No tie points for ground estimate; assuming ground 20-60 m below cameras.")

    buf = float(cfg.get('xy_buffer_m', 15.0))
    xmin, xmax = min(xs) - buf, max(xs) + buf
    ymin, ymax = min(ys) - buf, max(ys) + buf
    zmin = ground_low - float(cfg.get('z_below_m', 15.0))
    zmax = max(zs) + float(cfg.get('z_above_m', 10.0))

    R = Metashape.Matrix([[m[0, 0], m[0, 1], m[0, 2]],
                          [m[1, 0], m[1, 1], m[1, 2]],
                          [m[2, 0], m[2, 1], m[2, 2]]]) * (1.0 / s)
    centre_local = Metashape.Vector([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
    size_m = Metashape.Vector([xmax - xmin, ymax - ymin, zmax - zmin])

    chunk.region.rot = R.t()
    chunk.region.center = m.inv().mulp(centre_local)
    chunk.region.size = size_m * (1.0 / s)
    print(f"Region from cameras: size {size_m.x:.1f} x {size_m.y:.1f} x {size_m.z:.1f} m "
          f"(ground p02/p98 {ground_low:.1f}/{ground_high:.1f} m, cameras {min(zs):.1f}-{max(zs):.1f} m, "
          f"{len(cams)} aligned cameras)")
    return True


def build_boundary_shape(chunk, epsg_code, buffer_m, export_path=None, label='processing_boundary'):
    """
    Add an outer-boundary polygon (buffered convex hull of aligned camera positions) to the chunk shapes.
    Idempotent: if a shape with `label` exists it is kept.

    Args:
        chunk (Metashape.Chunk): Metashape chunk.
        epsg_code (str): e.g. "EPSG::32617".
        buffer_m (float): Buffer distance in metres.
        export_path (str, optional): Shapefile path to export the boundary to.

    Returns:
        tuple or None: (minx, miny, maxx, maxy) of the boundary in the projected CRS.
    """
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
        return None
    hull = MultiPoint(pts).convex_hull.buffer(float(buffer_m))
    coords = list(hull.exterior.coords)

    if chunk.shapes is None:
        chunk.shapes = Metashape.Shapes()
    chunk.shapes.crs = crs_out
    existing = [s for s in chunk.shapes.shapes if s.label == label]
    if existing:
        print(f"Boundary shape '{label}' already present; keeping it.")
    else:
        shape = chunk.shapes.addShape()
        shape.label = label
        try:
            shape.geometry = Metashape.Geometry.Polygon([Metashape.Vector([x, y]) for x, y in coords])
        except Exception:
            shape.geometry = Metashape.Geometry.Polygon([Metashape.Vector([x, y, 0.0]) for x, y in coords])
        shape.boundary_type = Metashape.Shape.BoundaryType.OuterBoundary
        print(f"Boundary shape built from {len(pts)} cameras, buffer {buffer_m} m, area {hull.area:.0f} m2")
    if export_path:
        try:
            chunk.exportShapes(export_path, save_polygons=True,
                               format=Metashape.ShapesFormat.ShapesFormatSHP, crs=crs_out)
            print(f"Boundary exported to {export_path}")
        except Exception as e:
            print(f"[WARN] Boundary export failed: {e}")
    return hull.bounds


def has_boundary_shape(chunk, label='processing_boundary'):
    try:
        return chunk.shapes is not None and any(s.label == label for s in chunk.shapes.shapes)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def retry_unaligned_cameras(chunk, align_cfg, progress=None):
    """
    Re-run alignCameras on cameras that have no transform after the first pass.

    Returns:
        list: Labels of cameras still unaligned afterwards.
    """
    unaligned = [c for c in chunk.cameras if c.enabled and c.transform is None]
    if not unaligned:
        print("All enabled cameras aligned.")
        return []
    print(f"{len(unaligned)} camera(s) unaligned after first pass: "
          + ", ".join(c.label for c in unaligned[:10]) + (" ..." if len(unaligned) > 10 else ""))
    chunk.alignCameras(cameras=unaligned,
                       adaptive_fitting=align_cfg.get('adaptive_fitting', False),
                       min_image=align_cfg.get('min_image', 2),
                       reset_alignment=False,
                       progress=progress)
    still = [c.label for c in chunk.cameras if c.enabled and c.transform is None]
    print(f"After retry: {len(still)} camera(s) still unaligned" + (": " + ", ".join(still[:10]) if still else ""))
    return still


def apply_precalibration(chunk, precal_cfg):
    """
    Load a saved calibration XML into every sensor as the initial (optionally fixed) calibration.
    """
    if not precal_cfg or not precal_cfg.get('enabled'):
        return False
    path = precal_cfg.get('path')
    if not path or not os.path.isfile(path):
        print(f"[WARN] Precalibration enabled but file not found: {path}")
        return False
    calib = Metashape.Calibration()
    calib.load(path)
    for sensor in chunk.sensors:
        sensor.user_calib = calib
        sensor.fixed_calibration = bool(precal_cfg.get('fixed', False))
    print(f"Precalibration loaded from {path} (fixed={bool(precal_cfg.get('fixed', False))}) "
          f"for {len(chunk.sensors)} sensor(s)")
    return True


def save_calibration_and_correlations(chunk, out_prefix, warn_threshold=0.95):
    """
    Save each sensor's adjusted calibration as XML and write a parameter correlation matrix CSV
    (from Calibration.covariance_matrix). Prints pairs with |r| > warn_threshold.

    Returns:
        list: (sensor_index, param_a, param_b, r) for flagged pairs.
    """
    import csv
    import math
    flagged = []
    for i, sensor in enumerate(chunk.sensors):
        calib = sensor.calibration
        xml_path = f"{out_prefix}_calibration_sensor{i}.xml"
        try:
            calib.save(xml_path)
            print(f"Calibration saved to {xml_path}")
        except Exception as e:
            print(f"[WARN] Could not save calibration for sensor {i}: {e}")
        cov = getattr(calib, 'covariance_matrix', None)
        params = list(getattr(calib, 'covariance_params', []) or [])
        if cov is None or not params:
            print(f"Sensor {i}: no calibration covariance available (run optimizeCameras first).")
            continue
        n = len(params)
        sd = [math.sqrt(max(cov[k, k], 0.0)) for k in range(n)]
        csv_path = f"{out_prefix}_calibration_correlations_sensor{i}.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['param'] + params)
            for a in range(n):
                row = [params[a]]
                for b in range(n):
                    r = cov[a, b] / (sd[a] * sd[b]) if sd[a] > 0 and sd[b] > 0 else float('nan')
                    row.append(f"{r:.4f}")
                    if b > a and abs(r) > warn_threshold:
                        flagged.append((i, params[a], params[b], r))
                w.writerow(row)
        print(f"Calibration correlations written to {csv_path}")
    for i, a, b, r in flagged:
        print(f"[WARN] Sensor {i}: |corr({a},{b})| = {abs(r):.3f} > {warn_threshold}")
    return flagged


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------

def filter_point_cloud_confidence(chunk, min_confidence, progress=None):
    """
    Remove points with confidence < min_confidence from the active point cloud and compact it.

    Returns:
        int: Number of points removed.
    """
    pc = chunk.point_cloud
    n0 = pc.point_count
    pc.setConfidenceFilter(0, int(min_confidence) - 1)
    pc.removePoints(list(Metashape.PointClass.values.values()), progress=progress)
    pc.resetFilters()
    pc.compactPoints()
    n1 = pc.point_count
    removed = n0 - n1
    pc.meta['confidence_filter/min_confidence'] = str(int(min_confidence))
    pc.meta['confidence_filter/removed'] = str(removed)
    pc.meta['confidence_filter/before'] = str(n0)
    print(f"Confidence filter: removed {removed:,} of {n0:,} points with confidence < {min_confidence} "
          f"({100.0 * removed / max(n0, 1):.2f}%); {n1:,} remain")
    return removed


def point_cloud_confidence_filtered(chunk):
    try:
        return bool(chunk.point_cloud) and 'confidence_filter/min_confidence' in chunk.point_cloud.meta
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Raster checks (GDAL, optional)
# ---------------------------------------------------------------------------

def check_raster_grid_alignment(paths):
    """
    Compare geotransform and size of exported rasters.

    Args:
        paths (dict): label -> path.

    Returns:
        bool or None: True if all match (within 1e-6 of a pixel), False if not, None if GDAL unavailable.
    """
    try:
        from osgeo import gdal
    except Exception:
        print("[INFO] GDAL not importable; skipping grid alignment check.")
        return None
    info = {}
    for label, path in paths.items():
        if not path or not os.path.isfile(path):
            continue
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        info[label] = (ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize, ds.RasterCount,
                       band.GetNoDataValue(), gdal.GetDataTypeName(band.DataType))
        ds = None
    for label, (gt, w, h, nb, nd, dt) in info.items():
        print(f"  {label}: {w}x{h} px, res {gt[1]:.6f}, origin ({gt[0]:.3f}, {gt[3]:.3f}), "
              f"bands {nb}, nodata {nd}, {dt}")
    if len(info) < 2:
        return True
    ref_label, ref = next(iter(info.items()))
    ref_gt, ref_w, ref_h = ref[0], ref[1], ref[2]
    ok = True
    for label, (gt, w, h, _nb, _nd, _dt) in info.items():
        tol = abs(ref_gt[1]) * 1e-6
        if (w, h) != (ref_w, ref_h) or any(abs(gt[k] - ref_gt[k]) > tol for k in range(6)):
            print(f"[WARN] Grid mismatch between {ref_label} and {label}")
            ok = False
    if ok:
        print("Raster grids are identical.")
    return ok


def compute_chm_with_gdal(dsm_path, dtm_path, out_path, nodata=-32767.0, block=2048):
    """
    Compute CHM = DSM - DTM on the DSM grid with GDAL (DTM resampled bilinearly), windowed.
    Used only when Metashape's DSM/DTM/CHM grids do not match.
    """
    from osgeo import gdal
    import numpy as np
    dsm = gdal.Open(dsm_path, gdal.GA_ReadOnly)
    gt = dsm.GetGeoTransform()
    w, h = dsm.RasterXSize, dsm.RasterYSize
    bounds = (gt[0], gt[3] + gt[5] * h, gt[0] + gt[1] * w, gt[3])
    warped = gdal.Warp('/vsimem/dtm_on_dsm.tif', dtm_path, format='GTiff', outputBounds=bounds,
                       width=w, height=h, resampleAlg='bilinear', dstNodata=nodata)
    drv = gdal.GetDriverByName('GTiff')
    out = drv.Create(out_path, w, h, 1, gdal.GDT_Float32,
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
            bad = (a == dnd) | (b == nodata) | ~np.isfinite(chm)
            chm[bad] = nodata
            ob.WriteArray(chm, x, y)
    ob.FlushCache()
    out.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64])
    out = None
    warped = None
    gdal.Unlink('/vsimem/dtm_on_dsm.tif')
    print(f"CHM computed with GDAL on the DSM grid: {out_path}")