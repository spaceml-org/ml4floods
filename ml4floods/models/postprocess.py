from rasterio import features
import rasterio
from shapely.geometry import shape, mapping, Polygon
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np
from typing import List, Optional, Union
from ml4floods.data.create_gt import get_brightness, BRIGHTNESS_THRESHOLD
from ml4floods.data import utils
import torch
import geopandas as gpd
import os
from shapely.ops import unary_union
import pandas as pd
import warnings


def preprocess_water_probabilities(prob_water_mask, thres_h=0.6, thres_l=0.4, distance= 5, conn=2, watershed_line=True):
    mask0 = prob_water_mask > thres_h
    local_maxi = peak_local_max(prob_water_mask, indices=False, footprint=np.ones((distance * 2 + 1, distance * 2 + 1)),
                                labels=(prob_water_mask > thres_l))
    local_maxi[mask0] = True
    seed_msk = ndi.label(local_maxi)[0]

    prob_water_mask = watershed(-prob_water_mask, seed_msk, mask=(prob_water_mask > thres_l), watershed_line=watershed_line)
    return measure.label(prob_water_mask, connectivity=conn, background=0).astype('uint8')


def get_water_polygons(binary_water_mask: np.ndarray, min_area:float=25.5,
                       polygon_buffer:int=0, tolerance:float=1., transform: Optional[rasterio.Affine]=None) -> List[Polygon]:
    """

    Args:
        binary_water_mask: (H, W) binary mask to rasterise
        min_area: polygons with pixel area lower than this will be filtered
        polygon_buffer: buffering of the polygons
        tolerance: to simplify the polygons
        transform: affine transformation of the binary_water_mask raster

    Returns:
        list of rasterised polygons

    """
    assert binary_water_mask.ndim == 2, f"Expected mask with 2 dim found {binary_water_mask.shape}"

    geoms_polygons = []
    polygon_generator = features.shapes(binary_water_mask.astype(np.int16),
                                        binary_water_mask)

    for polygon, value in polygon_generator:
        p = shape(polygon)
        if polygon_buffer > 0:
            p = p.buffer(polygon_buffer)
        if p.area >= min_area:
            p = p.simplify(tolerance=tolerance)
            if transform is not None:
                p = transform_polygon(p, transform) # Convert polygon to raster coordinates
            geoms_polygons.append(p)

    return geoms_polygons

def transform_polygon(polygon:Polygon, transform: rasterio.Affine) -> Polygon:
    """
    Transforms a polygon from pixel coordinates to the coordinates specified by the affine transform

    Args:
        polygon: polygon to transform
        transform: Affine transformation

    Returns:
        polygon with coordinates transformed by the affine transformation

    """
    geojson_dict = mapping(polygon)
    out_coords = []
    for pol in geojson_dict["coordinates"]:
        pol_out = []
        for coords in pol:
            pol_out.append(transform * coords)

        out_coords.append(pol_out)

    geojson_dict["coordinates"] = out_coords

    return shape(geojson_dict)


def get_mask_watertypes(mndwi: Union[np.ndarray, torch.Tensor],
                        water_mask:Union[np.ndarray, torch.Tensor],
                        permanent_water:Optional[Union[np.ndarray, torch.Tensor]]=None):
    "Water mask (H, W) with interpretation {0: invalids, 1: land, 2: flood water, 3: thick cloud, 4:permanent water, 5: flood trace}"
    # 2: flood water
    # 4: permanent_water
    # 5: flood_trace
    water_mask_types = water_mask.copy()
    water_mask_types[(water_mask == 2) & (mndwi < 0)] = 5
    if permanent_water is not None:
        water_mask_types[(water_mask != 3) & (permanent_water == 3)] = 4

    return water_mask_types


def get_pred_mask_v2(inputs: Union[np.ndarray, torch.Tensor], prediction: Union[np.ndarray, torch.Tensor],
                     channels_input:Optional[List[int]]=None,
                     th_water:float = 0.5, th_cloud:float = 0.5, mask_clouds:bool = True,
                     th_brightness:float=BRIGHTNESS_THRESHOLD,
                     collection_name:str="S2") -> Union[np.ndarray, torch.Tensor]:
    """
    Receives an output of a WFV2 model (multioutput binary) and returns the corresponding 3-class segmentation mask

    Args:
        inputs: S2 image (C, H, W)
        prediction: corresponding model output (2, H, W)
        channels_input: 0-based list of indexes of s2_image channels (expected that len(channels_input) == s2_image.shape[0])
        th_water: threshold for the class water
        th_cloud: threshold for the class cloud
        th_brightness: threshold for the brightness to differenciate thick from thin clouds
        mask_clouds: If False ignores brightness and outputs the prediction mask with thin clouds
        collection_name:
        
    Returns:
        Water mask (H, W) with interpretation {0: invalids, 1: land, 2: water, 3: cloud}
    """
    if isinstance(inputs, torch.Tensor):
        mask_invalids = torch.all(inputs == 0, dim=0).cpu()
        output = torch.ones(prediction.shape[-2:], dtype=torch.uint8, device=torch.device("cpu"))
        prediction = prediction.cpu()
    else:
        mask_invalids = np.all(inputs == 0, axis=0)
        output = np.ones(prediction.shape[-2:], dtype=np.uint8)

    cloud_mask = (prediction[0] > th_cloud)
    # TODO erode cloud mask as in create_gt??

    water_mask = (prediction[1] > th_water)
    output[water_mask] = 2
    
    if mask_clouds:
        br = get_brightness(inputs, channels_input=channels_input,
                            collection_name=collection_name) > th_brightness
        if hasattr(br, "cpu"):
            br = br.cpu()
        output[cloud_mask & br] = 3
    else:
        output[cloud_mask] = 3
    
    output[mask_invalids] = 0
    return output

def compute_cloud_coverage(data:gpd.GeoDataFrame) -> float:
    area_total = data[data["class"] == "area_imaged"].geometry.area.sum()
    area_clouds = data[data["class"] == "cloud"].geometry.area.sum()
    return float(area_clouds / area_total)


def get_floodmap_pre(flooding_date_pre:str, geojsons:List[str]) -> Optional[gpd.GeoDataFrame]:
    """
    From a list of predicted geojsons returns the GeoDataFrame with lowest cloud coverage.

    Args:
        flooding_date_pre:
        geojsons:

    Returns:
        floodmap with lowest cloud coverage

    """
    best_floodmap_pre = None
    cloud_cover = 1
    for g in geojsons:
        date_iter = os.path.splitext(os.path.basename(g))[0]

        if date_iter < flooding_date_pre:
            data = utils.read_geojson_from_gcp(g)
            cc_iter = compute_cloud_coverage(data)
            if cc_iter < cloud_cover:
                best_floodmap_pre = data
                cloud_cover = cc_iter

    return best_floodmap_pre


def compute_flood_water(floodmap_post_data:gpd.GeoDataFrame, best_pre_flood_data:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Computes the difference between water in the pre and post floodmap

    Args:
        floodmap_post_data:
        best_pre_flood_data:

    Returns:

    """
    warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)

    if floodmap_post_data.crs != best_pre_flood_data.crs:
        best_pre_flood_data = best_pre_flood_data.to_crs(floodmap_post_data.crs)
    else:
        best_pre_flood_data = best_pre_flood_data.copy()

    pre_flood_water_or_cloud = unary_union(best_pre_flood_data[(best_pre_flood_data["class"] == "water") | (best_pre_flood_data["class"] == "cloud")].geometry)

    # TODO deal with valid pixels in pre-post flood (area_imaged)

    # pre_flood_cloud = unary_union(best_pre_flood_data[(best_pre_flood_data["class"] == "cloud")].geometry)
    # pre_flood_water_minus_cloud = pre_flood_water.difference(pre_flood_cloud)

    geoms_flood = floodmap_post_data[floodmap_post_data["class"] == "water"].geometry.apply(
        lambda g: g.difference(pre_flood_water_or_cloud))
    geoms_flood = geoms_flood[~geoms_flood.isna() & ~geoms_flood.is_empty]

    geoms_trace = floodmap_post_data[(floodmap_post_data["class"] =="flood_trace")].geometry.apply(
        lambda g: g.difference(pre_flood_water_or_cloud))
    geoms_trace = geoms_trace[~geoms_trace.isna() & ~geoms_trace.is_empty]

    data_post_flood = gpd.GeoDataFrame(geometry=geoms_flood, crs=floodmap_post_data.crs)
    data_post_flood["class"] = "water-post-flood"
    data_post_flood_trace = gpd.GeoDataFrame(geometry=geoms_trace, crs=floodmap_post_data.crs)
    data_post_flood_trace["class"] = "flood-trace"
    data_post_flood = pd.concat([data_post_flood_trace, data_post_flood], ignore_index=True)

    # Expand multipolygons to single polygons
    data_post_flood = data_post_flood.explode(ignore_index=True)

    # simplify small polygons
    data_post_flood["geometry"] = data_post_flood["geometry"].simplify(tolerance=10)

    best_pre_flood_data.loc[best_pre_flood_data["class"] == "water", "class"] = "water-pre-flood"
    best_pre_flood_data.loc[best_pre_flood_data["class"] == "cloud", "class"] = "cloud-pre-flood"
    best_pre_flood_data.loc[best_pre_flood_data["class"] == "area_imaged", "class"] = "area_imaged-pre-flood"
    post_flood_propagate = floodmap_post_data[(floodmap_post_data["class"] == "area_imaged") | (floodmap_post_data["class"] == "cloud")]

    return pd.concat([best_pre_flood_data, post_flood_propagate, data_post_flood],
                     ignore_index=True)