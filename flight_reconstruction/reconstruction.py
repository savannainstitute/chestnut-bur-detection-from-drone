import Metashape # type: ignore
import os
import gc
import sys
import time
import yaml
import argparse

from flight_reconstruction import utils

def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to configuration YAML file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        SystemExit: If loading fails.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Metashape automated reconstruction')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--folder', type=str,
                        help='Folder to process (flat folder of images)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output folder (default: <folder>/outputs). Use a new folder for every variant run.')
    parser.add_argument('--image-list', type=str, default=None,
                        help='Text file with one image filename per line (relative to --folder); '
                             'restricts processing to those images (e.g. a test sub-block)')
    parser.add_argument('--sensor', type=str, default=None, choices=['auto', 'm3m', 'p1'],
                        help='Sensor profile to apply from config sensor_profiles (default: config `sensor`, auto)')
    parser.add_argument('--override-file', type=str, default=None,
                        help='Extra YAML deep-merged over the resolved config (A/B variants)')
    parser.add_argument('--allow-existing', action='store_true',
                        help='Resume in an output folder that already contains a project (otherwise refused)')
    args = parser.parse_args()
    return args

def configure_processors():
    """
    Configure Metashape to use only high VRAM GPUs (>= 8 GB VRAM) and disable CPU for GPU steps.

    Returns:
        None
    """
    devices = Metashape.app.enumGPUDevices()
    if len(devices) > 0:
        gpu_names = [device.get('name', f'GPU {i}') for i, device in enumerate(devices)]
        gpu_vrams = [device.get('mem_size', 0) for device in devices]  # VRAM in bytes

        print("Available GPUs:")
        for i, (name, vram) in enumerate(zip(gpu_names, gpu_vrams)):
            if vram:
                print(f"  [{i}] {name} ({vram / (1024 ** 3):.1f} GB VRAM)")
            else:
                print(f"  [{i}] {name} (VRAM unknown)")

        # Select only GPUs with >= 8 GB VRAM
        selected_indices = [
            i for i, vram in enumerate(gpu_vrams)
            if vram and vram >= 8 * (1024 ** 3)
        ]
        if not selected_indices:
            print("No high VRAM GPUs found. Using all available GPUs.")
            selected_indices = list(range(len(devices)))

        # Set GPU mask
        gpu_mask = sum(1 << i for i in selected_indices)
        Metashape.app.gpu_mask = gpu_mask
        print(f"Enabled GPU(s): {', '.join(gpu_names[i] for i in selected_indices)} (mask={gpu_mask})")

        print("Setting GPU backend...")
        Metashape.app.settings.setValue("main/gpu_enable_opencl", "1") # true
        Metashape.app.settings.setValue("main/gpu_enable_cuda", "0") # false

        # Always disable CPU for GPU steps - avoids memory fragmentation
        Metashape.app.cpu_enable = False
        print("CPU disabled during dedicated GPU processing steps")

    else:
        # No GPU available, use CPU
        Metashape.app.cpu_enable = True
        print("No GPUs detected, using CPU processing")

def check_processing_status(chunk):
    """
    Check what processing steps have already been completed.

    Args:
        chunk (Metashape.Chunk): The Metashape chunk object.

    Returns:
        dict: Status of each processing step.
    """
    elevations = getattr(chunk, "elevations", [])
    status = {
        'photos_loaded': len(chunk.cameras) > 0,
        'tie_points_exist': len(chunk.tie_points.points) > 0 if chunk.tie_points else False,
        'boundary_built': utils.has_boundary_shape(chunk),
        'depth_maps_built': bool(chunk.depth_maps),
        'point_cloud_built': bool(chunk.point_cloud),
        'confidence_filtered': utils.point_cloud_confidence_filtered(chunk),
        'ground_points_classified': chunk.point_cloud.point_count_by_class.get(2, 0) > 0 if chunk.point_cloud else False,
        'model_built': bool(chunk.model),
        'dsm_built': any(getattr(e, "label", "") == "DSM" for e in elevations),
        'dsm_raw_built': any(getattr(e, "label", "") == "DSM_raw" for e in elevations),
        'dtm_built': any(getattr(e, "label", "") == "DTM" for e in elevations),
        'chm_built': any(getattr(e, "label", "") == "CHM" for e in elevations),
        'orthomosaic_built': bool(chunk.orthomosaic)
    }
    return status

def dem_grid_from_elevation(elevation):
    """
    Capture the grid (resolution + bounding box in the elevation's projection) of a built DEM so the
    DTM and CHM can be built on exactly the same grid.

    Returns:
        dict: {'resolution': float, 'bbox': Metashape.BBox}
    """
    bbox = Metashape.BBox()
    bbox.min = Metashape.Vector([elevation.left, elevation.bottom])
    bbox.max = Metashape.Vector([elevation.right, elevation.top])
    return {'resolution': elevation.resolution, 'bbox': bbox}

def export_dem(chunk, path, projection, compression, nodata, progress):
    """Export the active elevation asset as a single-band float GeoTIFF with a nodata tag and no alpha."""
    chunk.exportRaster(
        path=path,
        projection=projection,
        nodata_value=nodata,
        save_alpha=False,
        source_data=Metashape.DataSource.ElevationData,
        image_compression=compression,
        progress=progress
    )

def run_reconstruction():
    """
    Run the full reconstruction pipeline.

    Returns:
        None
    """
    start_time = time.time()
    doc = None
    try:
        # License cleanup handlers and activation
        utils.setup_license_cleanup()
        license_obj = utils.activate_license()
        if not license_obj:
            sys.exit(1)

        # Load config and parse args
        args = parse_arguments()
        config = load_config(args.config)

        # Input folder and photos
        input_folder = args.folder
        if not input_folder or not os.path.isdir(input_folder):
            print(f"Input folder not found: {input_folder}")
            sys.exit(1)
        lowest_folder_name = os.path.basename(os.path.normpath(input_folder))
        output_folder = args.output_dir or os.path.join(input_folder, "outputs")
        project_path = os.path.join(output_folder, f"project_{lowest_folder_name}.psx")
        if os.path.exists(project_path) and not args.allow_existing:
            print(f"Refusing to run: {project_path} already exists.\n"
                  f"Pass --allow-existing to resume that project, or --output-dir <new folder> for a new run.")
            sys.exit(1)
        os.makedirs(output_folder, exist_ok=True)

        valid_exts = [".JPG", ".JPEG", ".TIF", ".TIFF"]
        photos = utils.find_files(input_folder, valid_exts)
        if args.image_list:
            photos = utils.read_image_list(args.image_list, input_folder)
        if not photos:
            print(f"No valid image files found in {input_folder}")
            sys.exit(1)
        print(f"Found {len(photos)} image files.")

        # Resolve sensor profile and optional A/B override, then record the config actually used
        config = utils.resolve_sensor_profile(config, args.sensor, photos)
        if args.override_file:
            override = load_config(args.override_file)
            config = utils.deep_merge(config, override)
            config['_resolved']['override_file'] = os.path.abspath(args.override_file)
        config['_resolved']['image_list'] = os.path.abspath(args.image_list) if args.image_list else None
        config['_resolved']['n_images'] = len(photos)
        config['_resolved']['input_folder'] = os.path.abspath(input_folder)
        utils.write_config_used(config, os.path.join(output_folder, f"{lowest_folder_name}_config_used.yml"))

        # UTM zone for all projected outputs (uses the first photo's GPS, no alignment needed)
        epsg_code = utils.get_utm_zone_from_gps(photos)
        utm_projection = Metashape.OrthoProjection()
        utm_projection.crs = Metashape.CoordinateSystem(epsg_code)
        utm_crs = Metashape.CoordinateSystem(epsg_code)

        # Open or create project and chunk
        doc = Metashape.Document()
        if os.path.exists(project_path):
            print(f"Opening existing project: {project_path}")
            doc.open(project_path, read_only=False, ignore_lock=True)
            if len(doc.chunks) > 0:
                chunk = doc.chunks[0]
                print(f"Using existing chunk with {len(chunk.cameras)} cameras.")
            else:
                chunk = doc.addChunk()
                print("Created new chunk in existing project.")
        else:
            print(f"Creating new project: {project_path}")
            doc.save(project_path)
            chunk = doc.addChunk()
            print("Created new chunk.")

        # Check processing status
        status = check_processing_status(chunk)
        print(f"Processing status: {status}")

        # Configure processors
        configure_processors()

        # Add photos if not loaded
        if not status['photos_loaded']:
            print("Adding photos...")
            chunk.addPhotos(filenames=photos, load_xmp_accuracy=True, progress=utils.progress_timer.update)
            utils.progress_timer.reset()
            print(f"Added {len(chunk.cameras)} photos.")
            # Set camera labels to include lowest folder name for downstream tools
            # (image_selection resolves image paths from these labels)
            for camera in chunk.cameras:
                path = camera.photo.path
                parent_dir = os.path.basename(os.path.dirname(path))
                camera.label = f"{parent_dir}/{os.path.basename(path)}"
            doc.save()

            # RTK accuracy setup
            print("Setting RTK accuracy...")
            utils.setup_rtk_accuracy(chunk, config)
            doc.save()

            # Optional precalibration
            utils.apply_precalibration(chunk, config['camera'].get('precalibrated'))

            # Image quality analysis
            print("Analyzing image quality...")
            chunk.analyzeImages(progress=utils.progress_timer.update)
            utils.progress_timer.reset()
            low_quality_cameras = []
            for camera in chunk.cameras:
                if "Image/Quality" in camera.meta and camera.meta["Image/Quality"]:
                    quality = float(camera.meta["Image/Quality"])
                    if quality < config['image_quality']['quality_threshold']:
                        low_quality_cameras.append(camera)
            # Disable low quality cameras
            if low_quality_cameras:
                quality_threshold = config['image_quality']['quality_threshold']
                print(f"Disabling {len(low_quality_cameras)} low quality cameras below {quality_threshold}:")
                for camera in low_quality_cameras:
                    quality = float(camera.meta["Image/Quality"])
                    print(f"  {camera.label} (quality: {quality:.3f})")
                    camera.enabled = False
            else:
                print("No low quality cameras found.")
            doc.save()
        else:
            print("Photos already loaded, skipping...")

        # Photo matching and alignment
        if not status['tie_points_exist']:
            try:
                print("Matching photos...")
                match_params = config['photo_matching']
                chunk.matchPhotos(downscale=match_params['downscale'],
                                keypoint_limit=match_params['keypoint_limit'],
                                keypoint_limit_per_mpx=match_params['keypoint_limit_per_mpx'],
                                tiepoint_limit=match_params['tiepoint_limit'],
                                generic_preselection=match_params['generic_preselection'],
                                reference_preselection=match_params['reference_preselection'],
                                filter_mask=match_params['filter_mask'],
                                mask_tiepoints=match_params['mask_tiepoints'],
                                filter_stationary_points=match_params['filter_stationary_points'],
                                keep_keypoints=match_params['keep_keypoints'],
                                guided_matching=match_params['guided_matching'],
                                subdivide_task=match_params['subdivide_task'],
                                progress=utils.progress_timer.update)
                utils.progress_timer.reset()
                print("Photo matching finished.")

                print("Aligning cameras...")
                align_cfg = config['camera']['align']
                chunk.alignCameras(
                    adaptive_fitting=align_cfg['adaptive_fitting'],
                    min_image=align_cfg['min_image'],
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                print("Camera alignment finished.")
                if align_cfg.get('retry_unaligned', False):
                    utils.retry_unaligned_cameras(chunk, align_cfg, progress=utils.progress_timer.update)
                    utils.progress_timer.reset()
                n_aligned = sum(1 for c in chunk.cameras if c.transform is not None)
                print(f"Aligned cameras: {n_aligned} of {len(chunk.cameras)}")

                # Reset region to prevent point clipping (before optimization)
                set_region(chunk, config)

                # Camera optimization
                print("Optimizing cameras...")
                utils.optimize_camera_rtk(chunk, config['camera']['optimize'], progress=utils.progress_timer.update)
                utils.progress_timer.reset()
                print("Camera optimization finished.")
                doc.save()

                # USGS-style filtering
                print("Filtering tie points...")
                utils.filter_tie_points_usgs_part1(chunk, config)
                utils.filter_tie_points_usgs_part2(chunk, config)
                utils.progress_timer.reset()
                print("Tie point filtering finished.")

                # Reset region after filtering
                set_region(chunk, config)
                doc.save()

                # Save the adjusted calibration and its parameter correlations for review / reuse
                utils.save_calibration_and_correlations(
                    chunk, os.path.join(output_folder, lowest_folder_name))

                # Export camera positions (projected CRS)
                camera_file = os.path.join(output_folder, f"{lowest_folder_name}_camera_positions.txt")
                chunk.exportCameras(camera_file, format=Metashape.CamerasFormat.CamerasFormatOPK, crs=utm_crs)
                print(f"Camera positions exported to {camera_file} ({epsg_code})")

                doc.save()
                print("Alignment and optimization finished.")
            except Exception as e:
                print(f"Error during photo matching/alignment: {e}")
                sys.exit(1)
        else:
            print("Tie points already exist, skipping matching and alignment...")

        # Check if transform is valid before geometry-dependent steps
        if not utils.has_valid_transform(chunk):
            print("No valid transform found. Cannot proceed with point cloud, model, or DEM generation.")
            sys.exit(1)

        # Outer boundary shape for clipping exports
        boundary_cfg = config.get('boundary', {}) or {}
        if boundary_cfg.get('enabled', False) and not status['boundary_built']:
            try:
                boundary_shp = os.path.join(output_folder, f"{lowest_folder_name}_boundary.shp")
                utils.build_boundary_shape(chunk, epsg_code, boundary_cfg.get('buffer_m', 12.0), export_path=boundary_shp)
                doc.save()
            except Exception as e:
                print(f"Error building boundary shape: {e}")
                sys.exit(1)
        elif boundary_cfg.get('enabled', False):
            print("Boundary shape already present, skipping...")
        use_boundary = boundary_cfg.get('enabled', False) and utils.has_boundary_shape(chunk)

        # Depth maps
        if not status['depth_maps_built']:
            try:
                print("Building depth maps...")

                # Set subdivide_task for depth maps based on estimated RAM usage
                config = utils.adaptive_subdivide(chunk, 'depth_maps', config)

                # Convert filter_mode from string to Metashape enum
                filter_mode_str = config['depth_maps']['filter_mode'].lower()
                if filter_mode_str == "mild":
                    filter_mode = Metashape.FilterMode.MildFiltering
                elif filter_mode_str == "moderate":
                    filter_mode = Metashape.FilterMode.ModerateFiltering
                elif filter_mode_str == "aggressive":
                    filter_mode = Metashape.FilterMode.AggressiveFiltering
                else:
                    filter_mode = Metashape.FilterMode.NoFiltering

                chunk.buildDepthMaps(
                    downscale=config['depth_maps']['downscale'],
                    filter_mode=filter_mode,
                    reuse_depth=config['depth_maps']['reuse_depth'],
                    max_neighbors=config['depth_maps']['max_neighbors'],
                    subdivide_task=config['depth_maps']['subdivide_task'],
                    max_gpu_multiplier=config['depth_maps']['max_gpu_multiplier'],
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                print("Depth maps finished building.")
                doc.save()
            except Exception as e:
                print(f"Error building depth maps: {e}")
                sys.exit(1)
        else:
            print("Depth maps already built, skipping...")

        # Point cloud
        if not status['point_cloud_built']:
            try:
                source_data_str = config['point_cloud']['source_data'].lower()
                if source_data_str == "depth_maps":
                    source_data = Metashape.DataSource.DepthMapsData
                else:
                    source_data = Metashape.DataSource.PointCloudData
                print("Building dense cloud...")
                chunk.buildPointCloud(
                    source_data=source_data,
                    point_colors=config['point_cloud']['point_colors'],
                    point_confidence=config['point_cloud']['point_confidence'],
                    keep_depth=config['point_cloud']['keep_depth'],
                    max_neighbors=config['point_cloud']['max_neighbors'],
                    subdivide_task=config['point_cloud']['subdivide_task'],
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                doc.save()
                print("Point cloud finished building.")
            except Exception as e:
                print(f"Error building point cloud: {e}")
                sys.exit(1)
        else:
            print("Point cloud already built, skipping...")

        # Confidence filter (removes low-confidence points before classification, DEMs and export)
        conf_cfg = config['point_cloud'].get('confidence_filter', {}) or {}
        if conf_cfg.get('enabled', False):
            if not status['confidence_filtered'] and not status['ground_points_classified']:
                try:
                    if not config['point_cloud']['point_confidence']:
                        print("[WARN] confidence_filter enabled but point_confidence is false; skipping filter.")
                    else:
                        print("Filtering point cloud by confidence...")
                        utils.filter_point_cloud_confidence(chunk, conf_cfg.get('min_confidence', 2),
                                                            progress=utils.progress_timer.update)
                        utils.progress_timer.reset()
                        doc.save()
                except Exception as e:
                    print(f"Error filtering point cloud by confidence: {e}")
                    sys.exit(1)
            else:
                print("Point cloud already confidence-filtered (or classified), skipping filter...")

        if not status['ground_points_classified']:
            try:
                print("Classifying ground points...")
                ground_config = config['classify_ground_points']
                chunk.point_cloud.classifyGroundPoints(
                    max_angle=ground_config['max_angle'],
                    max_distance=ground_config['max_distance'],
                    cell_size=ground_config['cell_size'],
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                doc.save()
                print("Ground points classified.")
                pc_file = os.path.join(output_folder, f"{lowest_folder_name}_point_cloud.{config['point_cloud']['export']['format']}")
                format_str = config['point_cloud']['export']['format'].lower()
                if format_str == "las":
                    export_format = Metashape.PointCloudFormat.PointCloudFormatLAS
                elif format_str == "laz":
                    export_format = Metashape.PointCloudFormat.PointCloudFormatLAZ
                elif format_str == "e57":
                    export_format = Metashape.PointCloudFormat.PointCloudFormatE57
                elif format_str == "ply":
                    export_format = Metashape.PointCloudFormat.PointCloudFormatPLY
                else:
                    export_format = Metashape.PointCloudFormat.PointCloudFormatXYZ
                chunk.exportPointCloud(
                    pc_file,
                    source_data=Metashape.DataSource.PointCloudData,
                    save_point_color=config['point_cloud']['export']['save_point_color'],
                    save_point_normal=config['point_cloud']['export']['save_point_normal'],
                    save_point_confidence=config['point_cloud']['export']['save_point_confidence'],
                    format=export_format,
                    crs=utm_crs,
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                print("Point cloud exported.")
            except Exception as e:
                print(f"Error classifying ground points: {e}")
                sys.exit(1)
        else:
            print("Ground points already classified, skipping...")

        # Mesh (3d model)
        model_cfg = config['model']
        if model_cfg.get('enabled', True) and not status['model_built']:
            try:
                print("Building 3D model...")
                model_kwargs = dict(
                    surface_type=utils.surface_type_from_str(model_cfg.get('surface_type', 'arbitrary')),
                    interpolation=utils.interpolation_from_str(model_cfg.get('interpolation', 'enabled')),
                    face_count=utils.face_count_from_str(model_cfg.get('face_count', 'high')),
                    source_data=Metashape.DataSource.DepthMapsData,
                    vertex_colors=model_cfg['vertex_colors'],
                    vertex_confidence=model_cfg['vertex_confidence'],
                    keep_depth=model_cfg['keep_depth'],
                    subdivide_task=model_cfg['subdivide_task'],
                    progress=utils.progress_timer.update
                )
                if use_boundary and model_cfg.get('clip_to_boundary', False):
                    model_kwargs['clip_to_boundary'] = True
                chunk.buildModel(**model_kwargs)
                utils.progress_timer.reset()
                doc.save()
                print("3D model finished building.")
            except Exception as e:
                print(f"Error building model: {e}")
                sys.exit(1)
        elif not model_cfg.get('enabled', True):
            print("Model disabled in config, skipping...")
        else:
            print("3D model already built, skipping...")

        # Compression settings: DEMs and orthomosaic have separate sections
        dem_compression = utils.make_compression(config['dem'])
        ortho_compression = utils.make_compression(config['orthomosaic']['export'])
        dem_nodata = config['dem']['nodata']
        dem_interp = utils.interpolation_from_str(config['dem'].get('interpolation', 'enabled'))
        dem_source_str = str(config['dem'].get('source', 'model')).lower()
        dem_source = utils.dem_source_from_str(dem_source_str)
        if dem_source == Metashape.DataSource.ModelData and not chunk.model:
            print("[WARN] dem.source is 'model' but no mesh exists; using point cloud for the DSM.")
            dem_source = Metashape.DataSource.PointCloudData
            dem_source_str = 'point_cloud'

        dsm_file = os.path.join(output_folder, f"{lowest_folder_name}_dsm.tif")
        dsm_raw_file = os.path.join(output_folder, f"{lowest_folder_name}_dsm_raw.tif")
        dtm_file = os.path.join(output_folder, f"{lowest_folder_name}_dtm.tif")
        chm_file = os.path.join(output_folder, f"{lowest_folder_name}_chm.tif")

        # DSM
        if not status['dsm_built']:
            try:
                print(f"Building DSM from {dem_source_str}...")
                try:
                    chunk.buildDem(
                        source_data=dem_source,
                        interpolation=dem_interp,
                        projection=utm_projection,
                        subdivide_task=config['dem']['subdivide_task'],
                        resolution=config['dem']['resolution'],
                        progress=utils.progress_timer.update
                    )
                except Exception as e:
                    if dem_source == Metashape.DataSource.DepthMapsData:
                        print(f"[WARN] DSM from depth maps failed ({e}); falling back to point cloud.")
                        dem_source = Metashape.DataSource.PointCloudData
                        chunk.buildDem(
                            source_data=dem_source,
                            interpolation=dem_interp,
                            projection=utm_projection,
                            subdivide_task=config['dem']['subdivide_task'],
                            resolution=config['dem']['resolution'],
                            progress=utils.progress_timer.update
                        )
                    else:
                        raise
                utils.progress_timer.reset()
                chunk.elevation.label = "DSM"
                doc.save()
                export_dem(chunk, dsm_file, utm_projection, dem_compression, dem_nodata, utils.progress_timer.update)
                utils.progress_timer.reset()
                print("DSM exported.")
            except Exception as e:
                print(f"Error building DSM: {e}")
                sys.exit(1)
        else:
            print("DSM already built, skipping...")

        dsm_asset = utils.get_elevation_by_label(chunk, "DSM")
        if dsm_asset is None:
            print("DSM asset not found; cannot continue.")
            sys.exit(1)
        dem_grid = dem_grid_from_elevation(dsm_asset)
        print(f"DSM grid: resolution {dem_grid['resolution']:.5f} m, extent "
              f"({dsm_asset.left:.2f}, {dsm_asset.bottom:.2f}) - ({dsm_asset.right:.2f}, {dsm_asset.top:.2f})")

        # DSM without interpolation: true coverage map for diagnostics (not used downstream)
        if config['dem'].get('export_raw_coverage', False) and not status['dsm_raw_built']:
            try:
                print(f"Building DSM_raw (no interpolation) from {dem_source_str}...")
                chunk.buildDem(
                    source_data=dem_source,
                    interpolation=Metashape.Interpolation.DisabledInterpolation,
                    projection=utm_projection,
                    region=dem_grid['bbox'],
                    subdivide_task=config['dem']['subdivide_task'],
                    resolution=dem_grid['resolution'],
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                chunk.elevation.label = "DSM_raw"
                doc.save()
                export_dem(chunk, dsm_raw_file, utm_projection, dem_compression, dem_nodata, utils.progress_timer.update)
                utils.progress_timer.reset()
                print("DSM_raw exported.")
            except Exception as e:
                print(f"[WARN] DSM_raw failed (diagnostic product only): {e}")

        # DTM (same grid as the DSM)
        if not status['dtm_built']:
            try:
                print("Building DTM from ground points in point cloud...")
                chunk.buildDem(
                    source_data=Metashape.DataSource.PointCloudData,
                    classes=[Metashape.PointClass.Ground],
                    interpolation=dem_interp,
                    projection=utm_projection,
                    region=dem_grid['bbox'],
                    subdivide_task=config['dem']['subdivide_task'],
                    resolution=dem_grid['resolution'],
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                chunk.elevation.label = "DTM"
                doc.save()
                export_dem(chunk, dtm_file, utm_projection, dem_compression, dem_nodata, utils.progress_timer.update)
                utils.progress_timer.reset()
                print("DTM exported.")
            except Exception as e:
                print(f"Error building DTM: {e}")
                sys.exit(1)
        else:
            print("DTM already built, skipping...")

        # CHM (same grid as the DSM)
        if not status['chm_built']:
            try:
                print("Creating Canopy Height Model (CHM)...")
                dsm_asset = utils.get_elevation_by_label(chunk, "DSM")
                dtm_asset = utils.get_elevation_by_label(chunk, "DTM")
                if dsm_asset is not None and dtm_asset is not None:
                    chunk.transformRaster(
                        asset=dsm_asset.key,
                        operand_asset=dtm_asset.key,
                        subtract=True,
                        nodata_value=dem_nodata,
                        projection=utm_projection,
                        region=dem_grid['bbox'],
                        resolution=dem_grid['resolution'],
                        replace_asset=False,
                        clip_to_boundary=use_boundary
                    )
                    chunk.elevation.label = "CHM"
                    doc.save()
                    export_dem(chunk, chm_file, utm_projection, dem_compression, dem_nodata, utils.progress_timer.update)
                    utils.progress_timer.reset()
                    print("CHM exported.")
                else:
                    print("DSM or DTM asset not found, CHM not created.")
            except Exception as e:
                print(f"Error building CHM: {e}")
                sys.exit(1)
        else:
            print("CHM already built, skipping...")

        # Verify the three exported grids match; if not, recompute the CHM on the DSM grid with GDAL
        try:
            print("Checking DSM/DTM/CHM grid alignment...")
            aligned = utils.check_raster_grid_alignment({'DSM': dsm_file, 'DTM': dtm_file, 'CHM': chm_file})
            if aligned is False:
                # Keep Metashape's CHM for reference and make the GDAL CHM (exactly on the DSM grid) the
                # canonical <name>_chm.tif that canopy segmentation reads.
                chm_ms_file = os.path.join(output_folder, f"{lowest_folder_name}_chm_metashape.tif")
                chm_gdal_file = os.path.join(output_folder, f"{lowest_folder_name}_chm_gdal.tif")
                utils.compute_chm_with_gdal(dsm_file, dtm_file, chm_gdal_file, nodata=float(dem_nodata))
                if os.path.isfile(chm_file):
                    os.replace(chm_file, chm_ms_file)
                os.replace(chm_gdal_file, chm_file)
                print(f"[WARN] Metashape CHM grid differed from the DSM; {os.path.basename(chm_file)} is now the "
                      f"GDAL DSM-DTM on the DSM grid and Metashape's CHM was kept as {os.path.basename(chm_ms_file)}.")
        except Exception as e:
            print(f"[WARN] Grid alignment check failed: {e}")

        # Set DSM as active elevation surface before building orthomosaic
        dsm_asset = utils.get_elevation_by_label(chunk, "DSM")
        if dsm_asset is not None:
            chunk.elevation = dsm_asset

        # Orthomosaic
        if not status['orthomosaic_built']:
            try:
                surface_str = str(config['orthomosaic'].get('surface', 'model')).lower()
                if surface_str == 'dem':
                    surface_data = Metashape.DataSource.ElevationData
                else:
                    surface_data = Metashape.DataSource.ModelData
                    if not chunk.model:
                        print("[WARN] orthomosaic.surface is 'model' but no mesh exists; using the DSM.")
                        surface_data = Metashape.DataSource.ElevationData
                        surface_str = 'dem'
                print(f"Building orthomosaic on {surface_str} surface...")
                blend_str = config['orthomosaic']['blending_mode'].lower()
                if blend_str == "mosaic":
                    blend_mode = Metashape.BlendingMode.MosaicBlending
                elif blend_str == "average":
                    blend_mode = Metashape.BlendingMode.AverageBlending
                elif blend_str == "max":
                    blend_mode = Metashape.BlendingMode.MaxBlending
                elif blend_str == "min":
                    blend_mode = Metashape.BlendingMode.MinBlending
                else:
                    blend_mode = Metashape.BlendingMode.DisabledBlending

                chunk.buildOrthomosaic(
                    surface_data=surface_data,
                    blending_mode=blend_mode,
                    ghosting_filter=config['orthomosaic']['ghosting_filter'],
                    fill_holes=config['orthomosaic']['fill_holes'],
                    cull_faces=config['orthomosaic']['cull_faces'],
                    refine_seamlines=config['orthomosaic']['refine_seamlines'],
                    projection=utm_projection,
                    resolution=config['orthomosaic'].get('resolution', 0),
                    subdivide_task=config['orthomosaic']['subdivide_task'],
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                doc.save()
                print("Orthomosaic finished building.")

                ortho_file = os.path.join(output_folder, f"{lowest_folder_name}_orthomosaic.tif")
                chunk.exportRaster(
                    ortho_file,
                    source_data=Metashape.DataSource.OrthomosaicData,
                    projection=utm_projection,
                    image_compression=ortho_compression,
                    save_alpha=True,
                    white_background=config['orthomosaic']['export']['white_background'],
                    progress=utils.progress_timer.update
                )
                utils.progress_timer.reset()
                print("Orthomosaic exported.")

                gc.collect()
            except Exception as e:
                print(f"Error building orthomosaic: {e}")
                sys.exit(1)
        else:
            print("Orthomosaic already built, skipping...")

        try:
            report_file = os.path.join(output_folder, f"{lowest_folder_name}_report.pdf")
            resolved = config.get('_resolved', {})
            user_settings = [
                ("Sensor profile", str(resolved.get('sensor_profile'))),
                ("EXIF model", str(resolved.get('exif_model'))),
                ("Metashape module", str(Metashape.app.version)),
                ("Config used", f"{lowest_folder_name}_config_used.yml"),
                ("Image list", str(resolved.get('image_list'))),
                ("Override file", str(resolved.get('override_file'))),
                ("DSM source", dem_source_str),
                ("Orthomosaic surface", str(config['orthomosaic'].get('surface', 'model'))),
            ]
            try:
                chunk.exportReport(report_file, title=f"Reconstruction {lowest_folder_name}",
                                   user_settings=user_settings)
            except Exception as e:
                print(f"[WARN] exportReport with user_settings failed ({e}); exporting plain report.")
                chunk.exportReport(report_file)
            print("Report exported.")

            elapsed = time.time() - start_time
            print(f"Processing finished for {lowest_folder_name} in {elapsed / 3600:.2f} h; results saved to {output_folder}.")

            doc.save()
            doc = None
            gc.collect()
            print("Document closed and memory released.")
        except Exception as e:
            print(f"Error during export: {e}")
            if doc is not None:
                doc.save()
                doc = None
            gc.collect()
            sys.exit(1)

    except Exception as e:
        print(f"Error processing folder {args.folder if 'args' in locals() else '?'}: {e}")
        sys.exit(1)
    finally:
        if 'doc' in locals() and doc is not None:
            doc.save()
            doc = None
        gc.collect()

def set_region(chunk, config):
    """
    Apply the configured region mode after alignment.
    """
    region_cfg = config.get('region', {}) or {}
    mode = str(region_cfg.get('mode', 'legacy_x3')).lower()
    if mode == 'cameras':
        if not utils.set_region_from_cameras(chunk, region_cfg):
            utils.reset_region(chunk)
    else:
        utils.reset_region(chunk)

if __name__ == '__main__':
    run_reconstruction()
    gc.collect()
