import logging
import os
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from rasterio import features
from shapely.ops import cascaded_union
from rasterio.crs import CRS
from affine import Affine

from src.data.config import BANDS_S2, CODES_FLOODMAP, UNOSAT_CLASS_TO_TXT
from src.data.utils import filter_land, filter_pols


def generate_floodmap_v1(
    register: Dict, worldfloods_root: str, filterland: bool = True
) -> gpd.GeoDataFrame:
    """
    Generates a floodmap and updates the register from data V1 (stored in map folder).

    Args:
        register:
        worldfloods_root:
        filterland:

    Returns:
        floodmap: An standarized format in with the same fields as activations.generate_floodmap

    """

    # TODO add area_of_interest map.shp for all the files!
    area_of_interest = gpd.read_file(
        os.path.join(
            worldfloods_root,
            "maps",
            "area_of_interest",
            register["layer name"],
            "map.shp",
        )
    )
    area_of_interest_pol = cascaded_union(area_of_interest["geometry"])

    # TODO assert CRS is 'EPSG:4326' for all the polygons!!
    mapdf = filter_pols(
        gpd.read_file(
            os.path.join(
                worldfloods_root,
                "maps",
                register["resource folder"],
                register["layer name"],
                "map.shp",
            )
        ),
        area_of_interest_pol,
    )

    assert mapdf.shape[0] > 0, f"No polygons within bounds for {register}"
    if register["source"] == "CopernicusEMS":
        column_water_class = mapdf["notation"]
    elif register["source"] == "unosat":
        mapdf.loc[mapdf.Water_Clas.isna(), "Water_Clas"] = 5
        column_water_class = mapdf["Water_Clas"].apply(lambda x: UNOSAT_CLASS_TO_TXT[x])
    elif register["source"] == "glofimr":
        column_water_class = "Not Applicable"
    else:
        raise NotImplementedError(f"error in source for {register}")

    floodmap = gpd.GeoDataFrame({"geometry": mapdf.geometry}, crs=mapdf.crs)
    floodmap["w_class"] = column_water_class
    floodmap["source"] = "flood"

    if "hydro" in register:
        register_hydro = register["hydro"]
        assert (
            register_hydro["source"] == "CopernicusEMS"
        ), f"Unexpected hydro file for source {register}"
        mapdf_hydro = filter_pols(
            gpd.read_file(
                os.path.join(
                    worldfloods_root,
                    "maps",
                    register_hydro["resource folder"],
                    register_hydro["layer name"],
                    "map.shp",
                )
            ),
            area_of_interest_pol,
        )

        mapdf_hydro = (
            filter_land(mapdf_hydro)
            if filterland and (mapdf_hydro.shape[0] > 0)
            else mapdf_hydro
        )
        if mapdf_hydro.shape[0] > 0:
            mapdf_hydro["source"] = "hydro"
            mapdf_hydro = mapdf_hydro.rename({"obj_type": "w_class"}, axis=1)
            mapdf_hydro = mapdf_hydro[["geometry", "w_class", "source"]].copy()
            floodmap = pd.concat([floodmap, mapdf_hydro], axis=0, ignore_index=True)

    floodmap.loc[floodmap.w_class.isna(), "w_class"] = "Not Applicable"

    # Concat area of interest
    area_of_interest["source"] = "area_of_interest"
    area_of_interest["w_class"] = "area_of_interest"
    area_of_interest = area_of_interest[["geometry", "w_class", "source"]].copy()
    floodmap = pd.concat([floodmap, area_of_interest], axis=0, ignore_index=True)

    # TODO set crs of the floodmap

    # TODO assert w_class in CODES_FLOODMAP

    # TODO save in the register file the new stuff (filenames)?

    return floodmap


def compute_water(
    tiffs2: str,
    floodmap: gpd.GeoDataFrame,
    window: Optional[rasterio.windows.Window] = None,
    permanent_water_path: str = None,
    keep_streams: bool = False,
) -> np.ndarray:
    """
    Rasterise flood map and add JRC permanent water layer

    Args:
        tiffs2: Tif file S2 (either remote or local)
        floodmap: geopandas dataframe with the annotated polygons
        window: rasterio.windows.Window to read. Could also be slices (slice(100, 200), slice(100, 200)
        permanent_water_path: Whether or not user JRC permanent water layer (from tifffimages/PERMANENTWATERJRC folder)
        keep_streams: A boolean flag to indicate whether to include streams in the water mask

    Returns:
        water_mask : np.uint8 raster same shape as tiffs2 {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
    """

    with rasterio.open(tiffs2) as src_s2:
        if window is None:
            out_shape = src_s2.shape
            transform = src_s2.transform
        else:
            out_shape = rasterio.windows.shape(
                window, height=src_s2.height, width=src_s2.width
            )
            transform = rasterio.windows.transform(window, src_s2.transform)
        target_crs = str(src_s2.crs).lower()

    if str(floodmap.crs).lower() != target_crs:
        floodmap = floodmap.to_crs(crs=target_crs)

    # area_of_interest contains the extent that was labeled (values out of this pol should be marked as invalid)
    floodmap_aoi = floodmap[floodmap["w_class"] == "area_of_interest"]

    floodmap_rasterise = floodmap
    if floodmap_aoi.shape[0] > 0:
        # TODO we are filtering now hydro_l look into all_touched in rasterio.features.rasterize
        floodmap_rasterise = floodmap_rasterise[
            (floodmap["w_class"] != "area_of_interest")
        ]

    if not keep_streams:
        floodmap_rasterise = floodmap_rasterise[floodmap["source"] != "hydro_l"]

    shapes_rasterise = (
        (g, CODES_FLOODMAP[w])
        for g, w in floodmap_rasterise[["geometry", "w_class"]].itertuples(
            index=False, name=None
        )
    )

    water_mask = features.rasterize(
        shapes=shapes_rasterise,
        fill=0,
        out_shape=out_shape,
        dtype=np.int16,
        transform=transform,
        all_touched=True,
    )

    # Load valid mask using the area_of_interest polygons (valid pixels are those within area_of_interest polygons)
    if floodmap_aoi.shape[0] > 0:
        shapes_rasterise = (
            (g, 1)
            for g, w in floodmap_aoi[["geometry", "w_class"]].itertuples(
                index=False, name=None
            )
        )
        valid_mask = features.rasterize(
            shapes=shapes_rasterise,
            fill=0,
            out_shape=out_shape,
            dtype=np.uint8,
            transform=transform,
            all_touched=True,
        )
        water_mask[valid_mask == 0] = -1

    if permanent_water_path is not None:
        logging.info("\t Adding permanent water")
        permament_water = rasterio.open(permanent_water_path).read(1, window=window)

        # Set to permanent water
        water_mask[(water_mask != -1) | (permament_water == 3)] = 3

        # Seasonal water (permanent_water == 2) will not be used

    return water_mask


# TODO: Have a single function. No need of 2 versions
def _read_s2img_cloudmask_v1(
    s2tiff: str,
    window: Optional[rasterio.windows.Window] = None,
    cloudprob_image_path: Optional[str] = None,
    cloudprob_in_lastband: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function of generate_gt_v1 and generate_gt_v2

    Args:
        s2tiff:
        window:
        cloudprob_image_path:
        cloudprob_in_lastband:

    Returns:
        s2img: C,H,W array with len(BANDS_S2) channels
        cloud_mask: H, W array with cloud probability

    """
    bands_read = list(range(1, len(BANDS_S2)))
    with rasterio.open(s2tiff, "r") as s2_rst:
        s2_img = s2_rst.read(bands_read, window=window)
    # print(cloudprob_in_lastband)
    if cloudprob_in_lastband:
        with rasterio.open(s2tiff, "r") as s2_rst:
            last_band = s2_rst.count
            cloud_mask = s2_rst.read(last_band, window=window)
            cloud_mask = (
                cloud_mask.astype(np.float32) / 100.0
            )  # cloud mask in the last band is from 0 - 100
    elif cloudprob_image_path is None:
        from src.data import cloud_masks

        # Compute cloud mask
        cloud_mask = cloud_masks.compute_cloud_mask(s2_img)

    elif isinstance(cloudprob_image_path, str):
        with rasterio.open(cloudprob_image_path, "r") as cld_rst:
            cloud_mask = cld_rst.read(1, window=window)

    else:
        raise ValueError(f"Unrecognized input type: {cloudprob_image_path}")

    return s2_img, cloud_mask


# TODO: Have a single function. No need of 2 versions
def _read_s2img_cloudmask_v2(
    s2tiff: str,
    window: Optional[rasterio.windows.Window] = None,
    cloudprob_tiff: Optional[str] = None,
    cloudprob_in_lastband: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function of generate_gt_v1 and generate_gt_v2

    Args:
        s2tiff:
        window:
        cloudprob_tiff:
        cloudprob_in_lastband:

    Returns:
        s2img: C,H,W array with len(BANDS_S2) channels
        cloud_mask: H, W array with cloud probability

    """
    bands_read = list(range(1, len(BANDS_S2) + 1))
    # bands_read = list(range(1, 4))  # bands in rasterio are 1-based!
    with rasterio.open(s2tiff, "r") as s2_rst:
        s2_img = s2_rst.read(bands_read, window=window)
    if cloudprob_in_lastband:
        with rasterio.open(s2tiff, "r") as s2_rst:
            last_band = s2_rst.count
            cloud_mask = s2_rst.read(last_band, window=window)
            cloud_mask = (
                cloud_mask.astype(np.float32) / 100.0
            )  # cloud mask in the last band is from 0 - 100
    else:
        if cloudprob_tiff is None:
            from src.data import cloud_masks

            # Compute cloud mask
            cloud_mask = cloud_masks.compute_cloud_mask(s2_img)
        else:
            with rasterio.open(cloudprob_tiff, "r") as cld_rst:
                cloud_mask = cld_rst.read(1, window=window)

    return s2_img, cloud_mask


def generate_land_water_cloud_gt(
    s2_image_path: str,
    floodmap_path: str,
    window: Optional[rasterio.windows.Window] = None,
    permanent_water_image_path: Optional[str] = None,
    keep_streams: bool = False,
    cloudprob_image_path: Optional[str] = None,
    cloudprob_in_lastband: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Old ground truth generating function (inherited from worldfloods_internal.compute_meta_tif.generate_mask_meta_clouds)

    Args:
        s2_image_path:
        floodmap:
        window:
        permanent_water_image_path:
        keep_streams: A boolean flag to indicate whether to include streams in the water mask
        cloudprob_image_path:
        cloudprob_in_lastband:

    Returns:
        gt (np.ndarray): (H, W) np.uint8 array with encodding: {-1: invalid, 0: land, 1: water, 2: clouds}
        meta: dictionary with metadata information

    """
    # open floodmap with geopandas
    floodmap = gpd.read_file(floodmap_path)
    # =========================================
    # Generate Cloud Mask given S2 Data
    # =========================================
    s2_img, cloud_mask = _read_s2img_cloudmask_v1(
        s2_image_path,
        window=window,
        cloudprob_image_path=cloudprob_image_path,
        cloudprob_in_lastband=cloudprob_in_lastband,
    )
    # =========================================
    # Compute Water Mask
    # =========================================
    water_mask = compute_water(
        s2_image_path,
        floodmap[floodmap["w_class"] != "area_of_interest"],
        window=window,
        permanent_water_path=permanent_water_image_path,
        keep_streams=keep_streams,
    )

    gt = _generate_gt_v1_fromarray(s2_img, cloudprob=cloud_mask, water_mask=water_mask)

    # Compute metadata of the ground truth
    metadata = {}
    metadata["gtversion"] = "v1"
    metadata["encoding_values"] = {-1: "invalid", 0: "land", 1: "water", 2: "cloud"}
    metadata["shape"] = list(water_mask.shape)
    metadata["s2_image_path"] = os.path.basename(s2_image_path)
    metadata["permanent_water_image_path"] = (
        os.path.basename(permanent_water_image_path)
        if permanent_water_image_path is not None
        else "None"
    )
    metadata["cloudprob_tiff"] = (
        os.path.basename(cloudprob_image_path)
        if not cloudprob_in_lastband and cloudprob_image_path is not None
        else "None"
    )
    metadata["method clouds"] = "s2cloudless"

    # Compute stats of the GT
    metadata["pixels invalid S2"] = int(np.sum(gt == 0))
    metadata["pixels clouds S2"] = int(np.sum(gt == 3))
    metadata["pixels water S2"] = int(np.sum(gt == 2))
    metadata["pixels land S2"] = int(np.sum(gt == 1))

    # Compute stats of the water mask
    metadata["pixels flood water S2"] = int(np.sum(water_mask == 1))
    metadata["pixels hydro water S2"] = int(np.sum(water_mask == 2))
    metadata["pixels permanent water S2"] = int(np.sum(water_mask == 3))

    # Sanity check: values of masks add up (water_mask is set to zero in _generate_gt_v1_fromarray where clouds or invalids)
    assert (
        metadata["pixels flood water S2"]
        + metadata["pixels hydro water S2"]
        + metadata["pixels permanent water S2"]
    ) == metadata[
        "pixels water S2"
    ], f'Different number of water pixels than expected {metadata["pixels flood water S2"]} {metadata["pixels hydro water S2"]} {metadata["pixels permanent water S2"]}, {metadata["pixels water S2"]} '

    with rasterio.open(s2_image_path) as s2_src:
        metadata["bounds"] = s2_src.bounds
        metadata["crs"] = s2_src.crs
        metadata["transform"] = s2_src.transform

    return gt, metadata


def generate_water_cloud_binary_gt(
    s2_image_path: str,
    floodmap_path: str,
    metadata_floodmap: Dict,
    window: Optional[rasterio.windows.Window] = None,
    keep_streams: bool = False,
    permanent_water_image_path: Optional[str] = None,
    cloudprob_image_path: Optional[str] = None,
    cloudprob_in_lastband: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    New ground truth generating function for multioutput binary classification

    Args:
        s2_image_path:
        floodmap:
        metadata_floodmap: Metadata of the floodmap (if satellite is optical will mask the land/water GT)
        window:
        permanent_water_image_path:
        cloudprob_image_path:
        cloudprob_in_lastband:

    Returns:
        gt (np.ndarray): (2, H, W) np.uint8 array where:
            First channel encodes the cloud GT {0: invalid, 1: clear, 2: cloud}
            Second channel encodes the land/water GT {0: invalid, 1: land, 2: water}
        meta: dictionary with metadata information

    """
    # open floodmap with geopandas
    floodmap = gpd.read_file(floodmap_path)

    # =========================================
    # Generate Cloud Mask given S2 Data
    # =========================================
    s2_img, cloud_mask = _read_s2img_cloudmask_v1(
        s2_image_path,
        window=window,
        cloudprob_image_path=cloudprob_image_path,
        cloudprob_in_lastband=cloudprob_in_lastband,
    )

    water_mask = compute_water(
        s2_image_path,
        floodmap[floodmap["w_class"] != "area_of_interest"],
        window=window,
        permanent_water_path=permanent_water_image_path,
        keep_streams=keep_streams,
    )

    # TODO this should be invalid if it is Sentinel-2 and it is exactly the same date ('satellite date' is the same as the date of retrieval of s2tiff)
    if metadata_floodmap["satellite"] == "Sentinel-2":

        invalid_clouds_threshold = 0.5
    else:
        invalid_clouds_threshold = None

    gt = _generate_gt_fromarray(
        s2_img,
        cloudprob=cloud_mask,
        water_mask=water_mask,
        invalid_clouds_threshold=invalid_clouds_threshold,
    )

    # Compute metadata of the ground truth
    metadata = {}
    metadata["gtversion"] = "v2"
    metadata["encoding_values"] = [
        {0: "invalid", 1: "clear", 2: "cloud"},
        {0: "invalid", 1: "land", 2: "water"},
    ]
    metadata["shape"] = list(water_mask.shape)
    metadata["s2_image_path"] = os.path.basename(s2_image_path)
    metadata["permanent_water_image_path"] = (
        os.path.basename(permanent_water_image_path)
        if permanent_water_image_path is not None
        else "None"
    )
    metadata["cloudprob_image_path"] = (
        os.path.basename(cloudprob_image_path)
        if not cloudprob_in_lastband and cloudprob_image_path is not None
        else "None"
    )
    metadata["method clouds"] = "s2cloudless"

    # Compute stats of the GT
    metadata["pixels invalid S2"] = int(np.sum(gt[0] == 0))
    metadata["pixels clouds S2"] = int(np.sum(gt[0] == 2))
    metadata["pixels water S2"] = int(np.sum(gt[1] == 2))
    metadata["pixels land S2"] = int(np.sum(gt[1] == 1))

    # Compute stats of the water mask
    metadata["pixels flood water S2"] = int(np.sum(water_mask == 1))
    metadata["pixels hydro water S2"] = int(np.sum(water_mask == 2))
    metadata["pixels permanent water S2"] = int(np.sum(water_mask == 3))

    # Sanity check: values of masks add up
    # assert (
    #     metadata["pixels flood water S2"]
    #     + metadata["pixels hydro water S2"]
    #     + metadata["pixels permanent water S2"]
    # ) == metadata[
    #     "pixels water S2"
    # ], f'Different number of water pixels than expected {metadata["pixels flood water S2"]} {metadata["pixels hydro water S2"]} {metadata["pixels permanent water S2"]}, {metadata["pixels water S2"]} '

    with rasterio.open(s2_image_path) as s2_src:
        metadata["bounds"] = s2_src.bounds
        metadata["crs"] = s2_src.crs
        metadata["transform"] = s2_src.transform

    return gt, metadata


def generate_gt_v1(
    s2tiff: str,
    floodmap: gpd.GeoDataFrame,
    metadata_floodmap: Dict,
    window: Optional[rasterio.windows.Window] = None,
    permanent_water_tiff: Optional[str] = None,
    cloudprob_tiff: Optional[str] = None,
    cloudprob_in_lastband: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Old ground truth generating function (inherited from worldfloods_internal.compute_meta_tif.generate_mask_meta_clouds)

    Args:
        s2tiff:
        floodmap:
        metadata_floodmap: Metadata of the floodmap (not used here but kept for compatibility with generate_gt_v2)
        window:
        permanent_water_tiff:
        cloudprob_tiff:
        cloudprob_in_lastband:

    Returns:
        gt (np.ndarray): (H, W) np.uint8 array with encodding: {-1: invalid, 0: land, 1: water, 2: clouds}
        meta: dictionary with metadata information

    """
    # =========================================
    # Generate Cloud Mask given S2 Data
    # =========================================
    s2_img, cloud_mask = _read_s2img_cloudmask_v1(
        s2tiff,
        window=window,
        cloudprob_tiff=cloudprob_tiff,
        cloudprob_in_lastband=cloudprob_in_lastband,
    )
    # =========================================
    # Compute Water Mask
    # =========================================
    water_mask = compute_water(
        s2tiff,
        floodmap[floodmap["w_class"] != "area_of_interest"],
        window=window,
        permanent_water_path=permanent_water_tiff,
    )

    gt = _generate_gt_v1_fromarray(s2_img, cloudprob=cloud_mask, water_mask=water_mask)

    # Compute metadata of the ground truth
    metadata = {}
    metadata["gtversion"] = "v1"
    metadata["encoding_values"] = {-1: "invalid", 0: "land", 1: "water", 2: "cloud"}
    metadata["shape"] = list(water_mask.shape)
    metadata["s2tiff"] = os.path.basename(s2tiff)
    metadata["permanent_water_tiff"] = (
        os.path.basename(permanent_water_tiff)
        if permanent_water_tiff is not None
        else "None"
    )
    metadata["cloudprob_tiff"] = (
        os.path.basename(cloudprob_tiff)
        if not cloudprob_in_lastband and cloudprob_tiff is not None
        else "None"
    )
    metadata["method clouds"] = "s2cloudless"

    # Compute stats of the GT
    metadata["pixels invalid S2"] = int(np.sum(gt == 0))
    metadata["pixels clouds S2"] = int(np.sum(gt == 3))
    metadata["pixels water S2"] = int(np.sum(gt == 2))
    metadata["pixels land S2"] = int(np.sum(gt == 1))

    # Compute stats of the water mask
    metadata["pixels flood water S2"] = int(np.sum(water_mask == 1))
    metadata["pixels hydro water S2"] = int(np.sum(water_mask == 2))
    metadata["pixels permanent water S2"] = int(np.sum(water_mask == 3))

    # Sanity check: values of masks add up (water_mask is set to zero in _generate_gt_v1_fromarray where clouds or invalids)
    assert (
        metadata["pixels flood water S2"]
        + metadata["pixels hydro water S2"]
        + metadata["pixels permanent water S2"]
    ) == metadata[
        "pixels water S2"
    ], f'Different number of water pixels than expected {metadata["pixels flood water S2"]} {metadata["pixels hydro water S2"]} {metadata["pixels permanent water S2"]}, {metadata["pixels water S2"]} '

    with rasterio.open(s2tiff) as s2_src:
        metadata["bounds"] = s2_src.bounds

    return gt, metadata


def generate_gt_v2(
    s2tiff: str,
    floodmap: gpd.GeoDataFrame,
    metadata_floodmap: Dict,
    window: Optional[rasterio.windows.Window] = None,
    permanent_water_tiff: Optional[str] = None,
    cloudprob_tiff: Optional[str] = None,
    cloudprob_in_lastband: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    New ground truth generating function for multioutput binary classification

    Args:
        s2tiff:
        floodmap:
        metadata_floodmap: Metadata of the floodmap (if satellite is optical will mask the land/water GT)
        window:
        permanent_water_tiff:
        cloudprob_tiff:
        cloudprob_in_lastband:

    Returns:
        gt (np.ndarray): (2, H, W) np.uint8 array where:
            First channel encodes the cloud GT {0: invalid, 1: clear, 2: cloud}
            Second channel encodes the land/water GT {0: invalid, 1: land, 2: water}
        meta: dictionary with metadata information

    """
    s2_img, cloud_mask = _read_s2img_cloudmask_v2(
        s2tiff,
        window=window,
        cloudprob_tiff=cloudprob_tiff,
        cloudprob_in_lastband=cloudprob_in_lastband,
    )

    water_mask = compute_water(
        s2tiff,
        floodmap[floodmap["w_class"] != "area_of_interest"],
        window=window,
        permanent_water_path=permanent_water_tiff,
    )

    # TODO this should be invalid if it is Sentinel-2 and it is exactly the same date ('satellite date' is the same as the date of retrieval of s2tiff)
    invalid_clouds_land_pixels = metadata_floodmap["satellite"] == "Sentinel-2"

    gt = _generate_gt_fromarray(
        s2_img,
        cloudprob=cloud_mask,
        water_mask=water_mask,
        invalid_clouds_land_pixels=invalid_clouds_land_pixels,
    )

    # Compute metadata of the ground truth
    metadata = {}
    metadata["gtversion"] = "v2"
    metadata["encoding_values"] = [
        {0: "invalid", 1: "clear", 2: "cloud"},
        {0: "invalid", 1: "land", 2: "water"},
    ]
    metadata["shape"] = list(water_mask.shape)
    metadata["s2tiff"] = os.path.basename(s2tiff)
    metadata["permanent_water_tiff"] = (
        os.path.basename(permanent_water_tiff)
        if permanent_water_tiff is not None
        else "None"
    )
    metadata["cloudprob_tiff"] = (
        os.path.basename(cloudprob_tiff)
        if not cloudprob_in_lastband and cloudprob_tiff is not None
        else "None"
    )
    metadata["method clouds"] = "s2cloudless"

    # Compute stats of the GT
    metadata["pixels invalid S2"] = int(np.sum(gt[0] == 0))
    metadata["pixels clouds S2"] = int(np.sum(gt[0] == 2))
    metadata["pixels water S2"] = int(np.sum(gt[1] == 2))
    metadata["pixels land S2"] = int(np.sum(gt[1] == 1))

    # Compute stats of the water mask
    metadata["pixels flood water S2"] = int(np.sum(water_mask == 1))
    metadata["pixels hydro water S2"] = int(np.sum(water_mask == 2))
    metadata["pixels permanent water S2"] = int(np.sum(water_mask == 3))

    # Sanity check: values of masks add up
    assert (
        metadata["pixels flood water S2"]
        + metadata["pixels hydro water S2"]
        + metadata["pixels permanent water S2"]
    ) == metadata[
        "pixels water S2"
    ], f'Different number of water pixels than expected {metadata["pixels flood water S2"]} {metadata["pixels hydro water S2"]} {metadata["pixels permanent water S2"]}, {metadata["pixels water S2"]} '

    with rasterio.open(s2tiff) as s2_src:
        metadata["bounds"] = s2_src.bounds

    return gt, metadata


def _generate_gt_v1_fromarray(
    s2_img: np.ndarray, cloudprob: np.ndarray, water_mask: np.ndarray
) -> np.ndarray:
    """
    Generate Ground Truth of V1 of WorldFloods multi-class classification problem

    :param s2_img: (C, H, W) used to find the invalid values in the input
    :param cloudprob:  probability value between [0,1]
    :param water_mask:  {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
    :return:
    """

    assert np.all(
        water_mask != -1
    ), "V1 does not expect masked inputs in the water layer"

    invalids = np.all(s2_img == 0, axis=0)

    # Set cloudprobs to zero in invalid pixels
    cloudprob[invalids] = 0
    cloudmask = cloudprob > 0.5

    # Set watermask values for compute stats
    water_mask[invalids | cloudmask] = 0

    # Create gt mask {0: invalid, 1:land, 2: water, 3: cloud}
    gt = np.ones(water_mask.shape, dtype=np.uint8)
    gt[water_mask > 0] = 2
    gt[cloudmask] = 3
    gt[invalids] = 0

    return gt


# THIS FUNCTION WORKS
# TODO: Would be nice to return the original water mask and
# do all the extra processing at the DataLoader end.
def _generate_gt_fromarray(
    s2_img: np.ndarray,
    cloudprob: np.ndarray,
    water_mask: np.ndarray,
    invalid_clouds_threshold: Optional[float] = 0.5,
) -> np.ndarray:
    """

    Generate Ground Truth of WorldFloods V2 (multi-output binary classification problem)

    Args:
        s2_img: (C, H, W) array
        cloudprob: (H, W) array
        water_mask: (H, W) array {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
        invalid_clouds_threshold: set in the land/water ground truth land pixels as invalid
        (this is a safety check for Copernicus EMS data derived from optical satellites since
        they don't mark cloudy pixels in the provided products), default=0.5

    Returns:
        (2, H, W) np.uint8 array where:
            First channel encodes {0: invalid, 1: clear, 2: cloud}
            Second channel encodes {0: invalid, 1: land, 2: water}

        A pixel is set to invalid if it's invalid in the water_mask layer or invalid in the s2_img (all values to zero)

    """

    invalids = np.all(s2_img == 0, axis=0) | (water_mask == -1)

    # Set cloudprobs to zero in invalid pixels
    cloudgt = np.ones(water_mask.shape, dtype=np.uint8)
    cloudgt[cloudprob > 0.5] = 2
    cloudgt[invalids] = 0

    # For clouds we could set to invalid only if the s2_img is invalid (not the water mask)?

    # Set watermask values for compute stats
    watergt = np.ones(water_mask.shape, dtype=np.uint8)  # whole image is 1
    watergt[water_mask >= 1] = 2  # only water is 2
    watergt[invalids] = 0

    if invalid_clouds_threshold is not None:
        # Set to invalid land pixels that are cloudy if the satellite is Sentinel-2
        watergt[(water_mask == 0) | (cloudprob > invalid_clouds_threshold)] = 0

    stacked_cloud_water_mask = np.stack([cloudgt, watergt], axis=0)

    return stacked_cloud_water_mask


def _get_image_geocoords(image_path: str) -> Tuple[CRS, Affine]:
    """Get important geocoordinates from saved image"""
    with rasterio.open(image_path) as src_image:
        crs = src_image.crs
        transform = src_image.transform
    return crs, transform