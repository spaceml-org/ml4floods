import logging
import os
from typing import Dict, Optional, Tuple, List, Union

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio import features

from ml4floods.data.worldfloods.configs import BANDS_S2, BANDS_L8
from ml4floods.data.config import CODES_FLOODMAP
from ml4floods.data import utils
from skimage.morphology import binary_opening, disk
import torch


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
        water_mask : np.int16 raster same shape as tiffs2 {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
    """

    with utils.rasterio_open_read(tiffs2) as src_s2:
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
        floodmap.to_crs(crs=target_crs, inplace=True)

    # area_of_interest contains the extent that was labeled (values out of this pol should be marked as invalid)
    floodmap_aoi = floodmap[floodmap["w_class"] == "area_of_interest"]

    floodmap_rasterise = floodmap
    if floodmap_aoi.shape[0] > 0:
        floodmap_rasterise = floodmap_rasterise[floodmap["w_class"] != "area_of_interest"]

    if not keep_streams:
        floodmap_rasterise = floodmap_rasterise[floodmap["source"] != "hydro_l"]

    shapes_rasterise = (
        (g, CODES_FLOODMAP[w])
        for g, w in floodmap_rasterise[["geometry", "w_class"]].itertuples(
            index=False, name=None
        ) if g and not g.is_empty
    )

    water_mask = features.rasterize(
        shapes=shapes_rasterise,
        fill=0,
        out_shape=out_shape,
        dtype=np.int16,
        transform=transform,
        all_touched=keep_streams,
    )

    # Load valid mask using the area_of_interest polygons (valid pixels are those within area_of_interest polygons)
    if floodmap_aoi.shape[0] > 0:
        shapes_rasterise = (
            (g, 1)
            for g, w in floodmap_aoi[["geometry", "w_class"]].itertuples(
                index=False, name=None
            ) if g and not g.is_empty
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
        with utils.rasterio_open_read(permanent_water_path) as srcperm:
            permanent_water = srcperm.read(1, window=window)

        # Set to permanent water
        # Only interested in permanent water labelled as 3 and valid water masks.
        # Adding the third label for permanent water in water mask.
        water_mask[(water_mask != -1) & (permanent_water == 3)] = 3

        # Seasonal water (permanent_water == 2) will not be used

    return water_mask


def read_s2img_cloudmask(
    s2tiff: str,
    window: Optional[rasterio.windows.Window] = None,
    cloudprob_image_path: Optional[str] = None,
    bands_read: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to load a s2 image and its cloud mask

    Args:
        s2tiff: path to S2 tiff file
        window: Optional window to read
        cloudprob_image_path: Optional path to geojson or tiff file with cloud mask.
        bands_read: bands to read from `BANDS_S2` [0-based]

    Returns:
        s2img: (C, H, W) array with len(bands_read) channels.
        cloud_prob: (H, W) array with cloud probability

    """
    if bands_read is None:
        bands_read = list(range(1, len(BANDS_S2) + 1))
    else:
        bands_read = [b+1 for b in bands_read]

    with utils.rasterio_open_read(s2tiff) as s2_rst:
        s2_img = s2_rst.read(bands_read, window=window)
        bands = s2_rst.descriptions
        cloudprob_in_lastband = (len(bands) > 14) and (bands[14] == "probability")
        transform = s2_rst.transform if window is None else rasterio.windows.transform(window, s2_rst.transform)
        crs = str(s2_rst.crs).lower()
        shape = s2_img.shape[1:]

    if cloudprob_image_path:
        fs = utils.get_filesystem(cloudprob_image_path)
    else:
        fs = None

    if fs and fs.exists(cloudprob_image_path):
        ext = os.path.splitext(cloudprob_image_path)[1]
        if ext == ".tif":
            with utils.rasterio_open_read(cloudprob_image_path) as cld_rst:
                cloud_prob = cld_rst.read(1, window=window)

        elif  ext == ".geojson":
            # check if empty file -> cloud_masks=np.zeros(shape)
            clouds_vec = utils.read_geojson_from_gcp(cloudprob_image_path)
            if clouds_vec.shape[0] == 0:
                cloud_prob = np.zeros(s2_img.shape[1:], dtype=np.float32)
            else:
                # Convert clouds_vec crs to s2 crs
                if str(clouds_vec.crs).lower() != crs:
                    clouds_vec.to_crs(crs=crs, inplace=True)

                clouds_vec = clouds_vec[clouds_vec["class"].isin(["CLOUD", "cloud"])]
                shapes_rasterise = (
                    (g, 1)
                    for g,w in clouds_vec[["geometry", "class"]].itertuples(
                    index=False, name=None
                )
                )
                cloud_prob = features.rasterize(
                    shapes=shapes_rasterise,
                    fill=0,
                    out_shape=shape,
                    dtype=np.float32,
                    transform=transform,
                    all_touched=True,
                )
    elif cloudprob_in_lastband:
        with utils.rasterio_open_read(s2tiff) as s2_rst:
            last_band = s2_rst.count
            cloud_prob = s2_rst.read(last_band, window=window)
            cloud_prob = (
                cloud_prob.astype(np.float32) / 100.0
            )  # cloud mask in the last band is from 0 - 100
    else:
        from ml4floods.data import cloud_masks
        # Compute cloud mask
        cloud_prob = cloud_masks.compute_cloud_mask(s2_img)

    return s2_img, cloud_prob


def generate_land_water_cloud_gt(
    s2_image_path: str,
    floodmap: gpd.GeoDataFrame,
    metadata_floodmap: Dict,
    window: Optional[rasterio.windows.Window] = None,
    keep_streams: bool = False,
    permanent_water_image_path: Optional[str] = None,
    cloudprob_image_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Old ground truth generating function (inherited from worldfloods_internal.compute_meta_tif.generate_mask_meta_clouds)

    Args:
        s2_image_path:
        floodmap:
        metadata_floodmap:
        window:
        permanent_water_image_path:
        keep_streams: A boolean flag to indicate whether to include streams in the water mask
        cloudprob_image_path:

    Returns:
        gt (np.ndarray): (H, W) np.uint8 array with encodding: {0: "invalid", 1: "land", 2: "water", 3: "cloud"}
        meta: dictionary with metadata information

    """
    # open floodmap with geopandas
    # =========================================
    # Generate Cloud Mask given S2 Data
    # =========================================
    s2_img, cloud_mask = read_s2img_cloudmask(
        s2_image_path,
        window=window,
        cloudprob_image_path=cloudprob_image_path,
    )
    # =========================================
    # Compute Water Mask
    # =========================================
    water_mask = compute_water(
        s2_image_path,
        floodmap,
        window=window,
        permanent_water_path=permanent_water_image_path,
        keep_streams=keep_streams,
    )

    gt = _generate_gt_v1_fromarray(s2_img, cloudprob=cloud_mask, water_mask=water_mask)

    # Compute metadata of the ground truth
    metadata = metadata_floodmap.copy()
    metadata["gtversion"] = "v1"
    metadata["encoding_values"] = {0: "invalid", 1: "land", 2: "water", 3: "cloud"}
    metadata["shape"] = list(water_mask.shape)
    metadata["s2_image_path"] = os.path.basename(s2_image_path)
    metadata["permanent_water_image_path"] = (
        os.path.basename(permanent_water_image_path)
        if permanent_water_image_path is not None
        else "None"
    )
    metadata["cloudprob_image_path"] = (
        os.path.basename(cloudprob_image_path)
        if cloudprob_image_path is not None
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

    with utils.rasterio_open_read(s2_image_path) as s2_src:
        metadata["bounds"] = s2_src.bounds
        metadata["crs"] = s2_src.crs
        metadata["transform"] = s2_src.transform

    return gt, metadata


def generate_water_cloud_binary_gt(
    s2_image_path: str,
    floodmap: gpd.GeoDataFrame,
    metadata_floodmap: Dict,
    window: Optional[rasterio.windows.Window] = None,
    keep_streams: bool = False,
    permanent_water_image_path: Optional[str] = None,
    cloudprob_image_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    New ground truth generating function for multioutput binary classification

    Args:
        s2_image_path:
        floodmap:
        metadata_floodmap: Metadata of the floodmap (if satellite is optical will mask the land/water GT)
        window:
        keep_streams:
        permanent_water_image_path:
        cloudprob_image_path:

    Returns:
        gt (np.ndarray): (2, H, W) np.uint8 array where:
            First channel encodes the cloud GT {0: invalid, 1: clear, 2: cloud}
            Second channel encodes the land/water GT {0: invalid, 1: land, 2: water}
        meta: dictionary with metadata information

    """

    # =========================================
    # Generate Cloud Mask given S2 Data
    # =========================================
    s2_img, cloud_mask = read_s2img_cloudmask(
        s2_image_path,
        window=window,
        cloudprob_image_path=cloudprob_image_path,
    )
    custom_cloud_mask = (cloudprob_image_path is not None) and cloudprob_image_path.endswith(".geojson")

    water_mask = compute_water(
        s2_image_path,
        floodmap,
        window=window,
        permanent_water_path=permanent_water_image_path,
        keep_streams=keep_streams,
    )

    # # TODO this should be invalid if it is Sentinel-2 and it is exactly the same date ('satellite date' is the same as the date of retrieval of s2tiff)
    # if metadata_floodmap["satellite"] == "Sentinel-2":
    #     invalid_clouds_threshold = 0.5
    # else:
    #     invalid_clouds_threshold = None

    gt = _generate_gt_fromarray(
        s2_img,
        cloudprob=cloud_mask,
        water_mask=water_mask,
        custom_clouds=custom_cloud_mask,
    )

    # Compute metadata of the ground truth
    metadata = metadata_floodmap.copy()

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
        if cloudprob_image_path is not None
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

    with utils.rasterio_open_read(s2_image_path) as s2_src:
        metadata["bounds"] = s2_src.bounds
        metadata["crs"] = s2_src.crs
        metadata["transform"] = s2_src.transform

    return gt, metadata


def _generate_gt_v1_fromarray(
    s2_img: np.ndarray, cloudprob: np.ndarray, water_mask: np.ndarray
) -> np.ndarray:
    """
    Generate Ground Truth of V1 of WorldFloods multi-class classification problem

    :param s2_img: (C, H, W) used to find the invalid values in the input
    :param cloudprob:  probability value between [0,1]
    :param water_mask:  {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
    :return: (H, W) np.uint8 {0: invalid, 1: land, 2:water, 3: cloud}
    """

    invalids = np.all(s2_img == 0, axis=0) | (water_mask == -1)

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

def get_brightness(s2_image:Union[np.ndarray, torch.Tensor], channels_input:Optional[List[int]]=None,
                   collection_name:str="S2") -> Union[np.ndarray, torch.Tensor]:
    """
    Returns brightness of visible bands of s2
    Args:
        s2_image: (C, H, W) array
        channels_input: 0-based list of indexes of s2_image channels (expected that len(channels_input) == s2_image.shape[0])
        collection_name: S2 or Landsat
    Returns:
        (H, W) array
    """
    if channels_input is not None:
        assert len(channels_input) == s2_image.shape[0], \
            f"Given {len(channels_input)} channels expected {s2_image.shape[0]}"

        if collection_name == "Landsat":
            bands_read_names = [BANDS_L8[iband] for iband in channels_input]
        else:
            bands_read_names = [BANDS_S2[i] for i in channels_input]
    else:
        assert len(BANDS_S2) == s2_image.shape[0], \
            f"Given {len(BANDS_S2)} channels expected {s2_image.shape[0]}"
        if collection_name == "Landsat":
            bands_read_names = BANDS_L8
        else:
            bands_read_names = BANDS_S2

    idxs = [bands_read_names.index(b) for b in ["B4", "B3", "B2"]]

    if isinstance(s2_image, torch.Tensor):
        rgb_img = s2_image[idxs, ...].float() ** 2
        return torch.sqrt(torch.sum(rgb_img, dim=0))

    rgb_img = s2_image[idxs, ...].astype(np.float32) ** 2
    return np.sqrt(np.sum(rgb_img, axis=0))


CLOUDS_THRESHOLD = .5
BRIGHTNESS_THRESHOLD = 3_500


def _generate_gt_fromarray(
    s2_img: np.ndarray,
    cloudprob: np.ndarray,
    water_mask: np.ndarray,
    custom_clouds: bool=False,
) -> np.ndarray:
    """

    Generate Ground Truth of WorldFloods Extra (multi-output binary classification problem)

    Args:
        s2_img: (C, H, W) array
        cloudprob: (H, W) array
        water_mask: (H, W) array {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
        custom_clouds: whether it uses a custom gt or not.
            If it is not custom it will mask the bright cloud pixels in the s2image in the water mask.

    Returns:
        (2, H, W) np.uint8 array where:
            First channel encodes {0: invalid, 1: clear, 2: cloud}
            Second channel encodes {0: invalid, 1: land, 2: water}

        A pixel is set to invalid if it's invalid in the water_mask layer or invalid in the s2_img (all values to zero)

    """

    invalids = np.any(np.isnan(s2_img), axis=0) | np.all(s2_img[:len(BANDS_S2)] == 0, axis=0) | (water_mask == -1)

    # Set cloudprobs to zero in invalid pixels
    cloudgt = np.ones(water_mask.shape, dtype=np.uint8)
    cloudgt[cloudprob > CLOUDS_THRESHOLD] = 2
    cloudgt[invalids] = 0

    # For clouds we could set to invalid only if the s2_img is invalid (not the water mask)?

    # Set watermask values for compute stats
    watergt = np.ones(water_mask.shape, dtype=np.uint8)  # whole image is 1
    watergt[water_mask >= 1] = 2  # only water is 2
    watergt[invalids] = 0

    clouds = cloudprob > CLOUDS_THRESHOLD
    if custom_clouds:
        # set to invalid in watergt clouds in gt (assume manually added to include only thick clouds!)
        watergt[clouds] = 0
    else:
        # set to invalid in watergt for bright clouds
        brightness = get_brightness(s2_img) # (H, W)
        clouds &= (brightness >= BRIGHTNESS_THRESHOLD)

        # binary opening of bright clouds
        open_clouds = binary_opening(clouds, disk(3)).astype(np.bool)

        # Set to invalid bright clouds
        watergt[open_clouds] = 0

    # if invalid_clouds_threshold is not None:
    #     # Set to invalid land pixels that are cloudy if the satellite is Sentinel-2
    #     watergt[(water_mask == 0) & (cloudprob > invalid_clouds_threshold)] = 0

    stacked_cloud_water_mask = np.stack([cloudgt, watergt], axis=0)

    return stacked_cloud_water_mask
