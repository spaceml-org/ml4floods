from rasterio import features
import rasterio
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np
from typing import List, Optional, Union
from ml4floods.data.create_gt import get_brightness, BRIGHTNESS_THRESHOLD
from ml4floods.data import utils, vectorize
import torch
import geopandas as gpd
from shapely.ops import unary_union
from shapely import validation
import pandas as pd
import warnings
from tqdm import tqdm
from datetime import datetime
import os
import shapely
from shapely.set_operations import union_all

def _watershed_processing(prob_water_mask, thres_h=0.6, thres_l=0.4, distance= 5, conn=2, watershed_line=True):
    """
    Ancilliary function used to apply the watershed algorithm (to go from river lines river polygons). It doesn't
    work very well but maybe worth keeping in case it's useful in the future.

    Args:
        prob_water_mask:
        thres_h:
        thres_l:
        distance:
        conn:
        watershed_line:

    Returns:

    """
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
    Vectorize a binary mask excluding polygons smaller than `min_area` pixels.

    Args:
        binary_water_mask: (H, W) binary mask to rasterise
        min_area: polygons with pixel area lower than this will be filtered
        polygon_buffer: buffering of the polygons
        tolerance: to simplify the polygons, in pixel units
        transform: affine transformation of the binary_water_mask raster

    Returns:
        list of rasterised polygons

    """
    return vectorize.get_polygons(binary_water_mask, min_area, polygon_buffer, tolerance, transform)


def get_mask_watertypes(mndwi: Union[np.ndarray, torch.Tensor],
                        water_mask:Union[np.ndarray, torch.Tensor],
                        permanent_water:Optional[Union[np.ndarray, torch.Tensor]]=None) -> Union[np.ndarray, torch.Tensor]:
    """
    This function returns a single mask that differenciates flood_water from permanent_water and flood traces.
    We distinguish flood traces using the mndwi index.

    Args:
        mndwi: (H, W) tensor MNDWI index
        water_mask:  (H, W) tensor with output classes provided by the model {0:invalid, 1: land, 2:water, 3:thick cloud}
        permanent_water: (H, W) tensor of the JRC permanent water layer coded as:
         https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_3_YearlyHistory

    Returns:
        Water mask (H, W) with interpretation {0: invalids, 1: land, 2: flood water, 3: thick cloud, 4:permanent water, 5: flood trace}
    """
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
    # erode cloud mask as in create_gt??

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

def compute_cloud_coverage(floodmap:gpd.GeoDataFrame, area_imaged:Optional[Union[Polygon, MultiPolygon]]=None) -> float:
    """
    Returns cloud coverage from a vectorized floodmap

    Args:
        floodmap: GeoDataFrame with column 'class' with values 'area_imaged' and 'cloud'
        area_imaged: Polygon that indicates the area imaged. if not provided it will take it from the floodmap

    Returns:
        percentage of the image covered by clouds
    """
    if area_imaged is None:
        area_total = floodmap[floodmap["class"] == "area_imaged"].geometry.area.sum()
    else:
        area_total = area_imaged.area

    area_clouds = floodmap[floodmap["class"] == "cloud"].geometry.area.sum()
    return float(area_clouds / area_total)


def get_area_missing_or_cloud(floodmap:gpd.GeoDataFrame,
                              area_imaged:Union[Polygon,MultiPolygon]) -> Union[Polygon, MultiPolygon]:
    """
    Returns a Polygon with the area of the floodmap that hasn't been imaged (i.e. is out of `floodmap[floodmap['class'] == "area_imaged"]`)
    and that is not covered by clouds.

    Args:
        floodmap: GeoDataFrame with column 'class' with values 'area_imaged' and 'cloud'
        area_imaged: Polygon that indicates the area imaged.

    Returns:
        Polygon with the area of the floodmap that hasn't been imaged and that is not covered by clouds
    """

    area_missing = area_imaged.difference(unary_union(floodmap[(floodmap["class"] == "area_imaged")].geometry))
    clouds = unary_union(floodmap[(floodmap["class"] == "cloud")].geometry)
    area_missing_or_cloud =  clouds.union(area_missing)

    # Remove Lines or Points from missing area
    if area_missing_or_cloud.geom_type == "GeometryCollection":
        area_missing_or_cloud = unary_union(
            [gc for gc in area_missing_or_cloud.geoms if (gc.geom_type == "Polygon") or (gc.geom_type == "MultiPolygon")])

    return area_missing_or_cloud

def get_area_missing_or_cloud_or_land(floodmap:gpd.GeoDataFrame,
                                      area_imaged:Union[Polygon,MultiPolygon]) -> Union[Polygon, MultiPolygon]:
    """
    Returns a Polygon with the area of the floodmap that hasn't been imaged (i.e. is out of `floodmap[floodmap['class'] == "area_imaged"]`)
    and that is not covered by clouds or land.

    Args:
        floodmap: GeoDataFrame with column 'class' with values 'area_imaged' and 'cloud'
        area_imaged: Polygon that indicates the area imaged.

    Returns:
        Polygon with the area of the floodmap that hasn't been imaged and that is not covered by clouds or land
    """
    area_imaged_current = unary_union(floodmap[floodmap["class"] == "area_imaged"].geometry)
    area_missing = area_imaged.difference(area_imaged_current)
    clouds = unary_union(floodmap[(floodmap["class"] == "cloud")].geometry)
    # polygons_in_floodmap = unary_union(geodataframe_polygonsonly_valid(floodmap[floodmap["class"] != "area_imaged"]).geometry, 
    #                                    grid_size = 1)
    polygons_in_floodmap = union_all(geodataframe_polygonsonly_valid(floodmap[floodmap["class"] != "area_imaged"]).geometry, grid_size=1)
    
    land = area_imaged_current.difference(polygons_in_floodmap)
    area_missing_or_cloud_or_land =  clouds.union(area_missing).union(land)

    # Remove Lines or Points from missing area
    if area_missing_or_cloud_or_land.geom_type == "GeometryCollection":
        area_missing_or_cloud_or_land = unary_union(
            [gc for gc in area_missing_or_cloud_or_land.geoms if (gc.geom_type == "Polygon") or (gc.geom_type == "MultiPolygon")])
    
    return area_missing_or_cloud_or_land


def get_area_valid(floodmap:gpd.GeoDataFrame) -> Union[Polygon, MultiPolygon]:
    """
    Returns the polygon with the area valid (i.e. area that is imaged and is not cloud)

    Args:
        floodmap: GeoDataFrame with column 'class' with values 'area_imaged' and 'cloud'

    Returns:
        Polygon with the area valid

    """
    area_valid = unary_union(floodmap[(floodmap["class"] == "area_imaged")].geometry)
    clouds = unary_union(floodmap[(floodmap["class"] == "cloud")].geometry)
    return area_valid.difference(clouds)


def get_floodmap_pre(geojsons:List[str], verbose:bool=False) -> gpd.GeoDataFrame:
    """
    From a list of predicted GeoJSONs returns the GeoDataFrame with all water.


    Args:
        geojsons: List[GeoDataFrame] with column 'class' with values {"water", "cloud", "area_imaged", "flood_trace"}
        verbose:

    Returns:
        floodmap with the lowest cloud coverage. Classes: {"water", "cloud", "area_imaged"}

    """

    # fill the clouds in pre with data from other geojsons

    datas = []
    # Compute total area observed
    area_imaged = Polygon() # empty polygon
    crs = None
    for g in geojsons:
        data = utils.read_geojson_from_gcp(g)
        if crs is None:
            crs = data.crs
        elif crs != data.crs:
            data = data.to_crs(crs=crs)
        data = data[["class","geometry"]].copy()

        area_imaged = unary_union(data[data["class"] == "area_imaged"].geometry).union(area_imaged)
        datas.append(data)

    # Compute cloud coverage of all pre-flood data taking into account area observed
    ccs = []
    for data in datas:
        ccs.append(compute_cloud_coverage(data, area_imaged=area_imaged))

    # Sort data based on ccs
    idx_sorted = np.argsort(ccs)
    datas_sorted = [datas[idxdates] for idxdates in idx_sorted]

    return mosaic_floodmaps(datas_sorted, area_imaged, verbose=verbose)


def get_floodmap_post(geojsons:List[str],verbose:bool=False,
                      mode:str="first") -> gpd.GeoDataFrame:
    """
    From a list of sorted GeoJSONs returns the GeoDataFrame with all flood water smartly joined

    Args:
        geojsons: List[GeoDataFrame] with column 'class' with values {"water", "cloud", "area_imaged", "flood_trace"}
        verbose:
        mode: "first" or "max". If "first" it will prioritize the water and land of the first floodmap.
            If "max" it will take the maximum of water of all floodmaps.

    Returns:
        floodmap with lowest cloud coverage. Classes: {"water", "cloud", "area_imaged"}

    """

    # fill the clouds in pre with data from other geojsons

    datas = []
    # Compute total area observed
    area_imaged = Polygon() # empty polygon
    crs = None
    for g in geojsons:
        data = utils.read_geojson_from_gcp(g)
        if crs is None:
            crs = data.crs
        elif crs != data.crs:
            data = data.to_crs(crs=crs)
        data = data[["class","geometry"]].copy()

        area_imaged = unary_union(data[data["class"] == "area_imaged"].geometry).union(area_imaged)
        datas.append(data)

    return mosaic_floodmaps(datas, area_imaged, 
                            classes_water=["water", "flood_trace"], 
                            verbose=verbose, mode=mode)


def mosaic_floodmaps(datas:List[gpd.GeoDataFrame],
                     area_imaged:Union[Polygon, MultiPolygon],
                     classes_water:List[str]=["water"],
                     verbose:bool=False,
                     mode:str="first") -> gpd.GeoDataFrame:
    """
    Mosaics the floodmaps iteratively taking into account the valid area of each of them.

    Args:
        datas: List[GeoDataFrame] with column 'class' with values {"water", "cloud", "area_imaged", "flood_trace"}
        area_imaged: Polygon with the total area imaged
        classes_water: which water classes from ["water", "flood_trace"] to include in the output floodmap. For pre-flood
        maps we don't include the flood_trace class.
        verbose:
        mode: "first" or "max". If "first" it will take the water from the first floomap. If "max" it will return
        the maximum water area from all floodmaps.

    Returns:
        a GeoDataFrame with the mosaic of the input dataframes
    """

    # Loop data computing pre-flood water
    warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)
    best_floodmap = datas[0]
    crs = best_floodmap.crs
    area_missing_or_cloud = get_area_missing_or_cloud(best_floodmap, area_imaged=area_imaged)
    if mode == "max":
        area_not_mapped = get_area_missing_or_cloud_or_land(best_floodmap, area_imaged=area_imaged)
    else:
        area_not_mapped = area_missing_or_cloud

    # This dataframe will be filled with water polygons
    condition = None
    for c in classes_water:
        if condition is None:
            condition = best_floodmap["class"] == c
        else:
            condition|= (best_floodmap["class"] == c)

    best_floodmap = best_floodmap[condition].copy()
    for idx, data in enumerate(datas[1:]):
        if area_not_mapped.is_empty or not (area_not_mapped.geom_type in ["Polygon", "MultiPolygon", "GeometryCollection"]):
            if verbose:
                print(f"All area is covered in idx {idx+1}. Area missing empty: {area_not_mapped.is_empty} Geom type: {area_not_mapped.geom_type}")
            break

        if data.crs != best_floodmap.crs:
            data = data.to_crs(data.crs, inplace=False)

        # Add water polygons that intersect the missing data
        for c in classes_water:
            water_geoms = data[data["class"] == c].geometry.apply(lambda g: g.intersection(area_not_mapped))

            water_geoms = water_geoms[~water_geoms.isna() & ~water_geoms.is_empty]
            water_geoms = water_geoms.explode(ignore_index=True)
            water_geoms = water_geoms[water_geoms.geometry.geom_type == "Polygon"]

            if water_geoms.shape[0] > 0:
                water_data = gpd.GeoDataFrame(geometry=water_geoms, crs=crs)
                water_data["class"] = c
                best_floodmap = pd.concat([best_floodmap, water_data], ignore_index=True)

        # Update area missing or cloud
        area_missing_or_cloud = get_area_missing_or_cloud(data, area_imaged).intersection(area_missing_or_cloud)

        # Remove points or LineStrings
        if area_missing_or_cloud.geom_type == "GeometryCollection":
            area_missing_or_cloud = unary_union([gc for gc in area_missing_or_cloud.geoms if (gc.geom_type == "Polygon") or (gc.geom_type == "MultiPolygon")])
        
        # update area_missing
        if mode == "max":
            area_not_mapped = get_area_missing_or_cloud_or_land(best_floodmap, area_imaged=area_imaged).intersection(area_not_mapped)
        else:
            area_not_mapped = area_missing_or_cloud


    # Join adjacent polygons
    best_floodmap = geodataframe_polygonsonly_valid(best_floodmap)
    best_floodmap['geometry'] = best_floodmap['geometry'].apply(lambda x: shapely.set_precision(x, grid_size=1))
    best_floodmap = best_floodmap.dissolve(by="class").reset_index()
    # Explode multipoligons to polygons
    best_floodmap = best_floodmap.explode(ignore_index=True)
    stuff_concat = [best_floodmap]
    
    # Add clouds
    if not area_missing_or_cloud.is_empty and (area_missing_or_cloud.geom_type in ["Polygon", "MultiPolygon", "GeometryCollection"]):
        # If there is something missing must be cloud (because area_imaged is the union of all the area missing in all the pre-floodmaps)
        cloud_data = gpd.GeoDataFrame(geometry=[area_missing_or_cloud], crs=crs)
        cloud_data = cloud_data.explode(ignore_index=True)
        cloud_data["class"] = "cloud"
        stuff_concat.append(cloud_data)

    # Add area_imaged
    area_imaged_data = gpd.GeoDataFrame(geometry=[area_imaged], crs=crs)
    area_imaged_data = area_imaged_data.explode(ignore_index=True)
    area_imaged_data["class"] = "area_imaged"
    stuff_concat.append(area_imaged_data)

    # Filter stuff that are not polygons
    result =  pd.concat(stuff_concat, ignore_index=True)
    assert (result.geometry.geom_type != "MultiPolygon").all(), "Everything should be flattened! found some MultiPolygon"
    assert (result.geometry.geom_type != "GeometryCollection").all(), "Everything should be flattened! found some GeometryCollection"
    # Remove geometries that are not polyongs and exclude polygons with area >= 400m^2
    result = result[(result.geometry.geom_type == "Polygon") & (result.geometry.area >= 20*20)].copy()
    result["geometry"] = result["geometry"].simplify(tolerance=10)
    result = result[~result.geometry.isna() & ~result.geometry.is_empty]

    return result



def compute_pre_post_flood_water(floodmap_post_data:gpd.GeoDataFrame, best_pre_flood_data:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Computes the difference between water in the pre and post floodmap

    Args:
        floodmap_post_data: Floodmap with classes {'area_imaged' 'water' 'cloud' 'flood_trace'}
        best_pre_flood_data: Floodmap with classes {'area_imaged' 'water' 'cloud' 'flood_trace'}

    Returns:
        Floodmap with classes {'area_imaged'
                               'water-post-flood',
                               'water-pre-flood,
                                'cloud',
                                'flood-trace',
                                'area_imaged-pre-flood',
                                'cloud-pre-flood'}

    """
    warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning) 

    if floodmap_post_data.crs != best_pre_flood_data.crs:
        best_pre_flood_data = best_pre_flood_data.to_crs(floodmap_post_data.crs)

    area_imaged_post = unary_union(floodmap_post_data[floodmap_post_data["class"] == "area_imaged"].geometry)

    area_missing_pre = get_area_missing_or_cloud(best_pre_flood_data, area_imaged_post)
    pre_flood_water_or_missing_pre = unary_union(best_pre_flood_data[best_pre_flood_data["class"] == "water"].geometry).union(area_missing_pre)
    pre_flood_water_or_missing_pre = validation.make_valid(pre_flood_water_or_missing_pre)

    # pre_flood_cloud = unary_union(best_pre_flood_data[(best_pre_flood_data["class"] == "cloud")].geometry)
    # pre_flood_water_minus_cloud = pre_flood_water.difference(pre_flood_cloud)

    geoms_flood = floodmap_post_data[floodmap_post_data["class"] == "water"].geometry.apply(
        lambda g: validation.make_valid(g.difference(pre_flood_water_or_missing_pre)))
    geoms_flood = geodataframe_polygonsonly_valid(geoms_flood)
    geoms_flood = geoms_flood[~geoms_flood.isna() & ~geoms_flood.is_empty]
    geoms_flood = geoms_flood.explode(ignore_index=True)
    geoms_flood = geoms_flood[geoms_flood.geometry.geom_type == "Polygon"]

    geoms_trace = floodmap_post_data[(floodmap_post_data["class"] =="flood_trace")].geometry.apply(
        lambda g: validation.make_valid(g.difference(pre_flood_water_or_missing_pre)))

    geoms_trace = geodataframe_polygonsonly_valid(geoms_trace)
    geoms_trace = geoms_trace[~geoms_trace.isna() & ~geoms_trace.is_empty]
    geoms_trace = geoms_trace.explode(ignore_index=True)
    geoms_trace = geoms_trace[geoms_trace.geometry.geom_type == "Polygon"]

    data_post_flood = gpd.GeoDataFrame(geometry=geoms_flood, crs=floodmap_post_data.crs)
    data_post_flood["class"] = "water-post-flood"
    data_post_flood_trace = gpd.GeoDataFrame(geometry=geoms_trace, crs=floodmap_post_data.crs)
    data_post_flood_trace["class"] = "flood-trace"
    data_post_flood = pd.concat([data_post_flood_trace, data_post_flood], ignore_index=True)

    # Expand multipolygons to single polygons
    data_post_flood = data_post_flood.explode(ignore_index=True)

    # Remove geometries that are not polyongs and exclude polygons with area >= 400m^2
    data_post_flood = data_post_flood[(data_post_flood.geometry.geom_type == "Polygon") & (data_post_flood.geometry.area >= 20 * 20)].copy()

    # simplify polygons
    data_post_flood["geometry"] = data_post_flood["geometry"].simplify(tolerance=10)

    # Copy stuff from pre-flood data
    best_pre_flood_data_propagate = best_pre_flood_data[best_pre_flood_data["class"] != "flood_trace"].copy()
    best_pre_flood_data_propagate.loc[best_pre_flood_data_propagate["class"] == "water", "class"] = "water-pre-flood"
    best_pre_flood_data_propagate.loc[best_pre_flood_data_propagate["class"] == "cloud", "class"] = "cloud-pre-flood"
    best_pre_flood_data_propagate.loc[best_pre_flood_data_propagate["class"] == "area_imaged", "class"] = "area_imaged-pre-flood"

    post_flood_propagate = floodmap_post_data[(floodmap_post_data["class"] == "area_imaged") | (floodmap_post_data["class"] == "cloud")]

    result = pd.concat([best_pre_flood_data_propagate, post_flood_propagate, data_post_flood],
                     ignore_index=True)
    result = result[~result.geometry.isna() & ~result.geometry.is_empty]
    return result


def geometrycollection_to_multipolygon(x:GeometryCollection) -> Union[MultiPolygon, Polygon]:
    if x.geom_type == "GeometryCollection":
        x = unary_union(
            [gc for gc in x.geoms if (gc.geom_type == "Polygon") or (gc.geom_type == "MultiPolygon")])
    return x


def geodataframe_polygonsonly_valid(df:Union[gpd.GeoDataFrame, gpd.GeoSeries]) -> gpd.GeoDataFrame:
    if isinstance(df,gpd.GeoSeries):
        df = df.geometry.apply(lambda x: validation.make_valid(geometrycollection_to_multipolygon(x)))
        df = df.geometry.buffer(1e-9)
    elif isinstance(df,gpd.GeoDataFrame):
        df['geometry'] = df.geometry.apply(lambda x: validation.make_valid(geometrycollection_to_multipolygon(x)))
        df['geometry'] = df.geometry.buffer(1e-9)
        
    df = df[(~df.geometry.is_empty) & df.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return df.explode(ignore_index=True)


def spatial_aggregation(floodmaps_paths:List[str], dst_crs:str= "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Mosaics all the pre-post floodmaps into a single map reprojecting the polygons to the specified CRS

    Args:
        floodmaps_paths: List of paths to floodmaps as generated with get_floodmap_post and get_floodmaps_pre
        dst_crs:

    Returns:
        geoDataFrame with same classes as floodmaps_paths files

    """
    data_all = None

    for f in tqdm(floodmaps_paths):
        data = utils.read_geojson_from_gcp(f)
        data = data[~data.geometry.isna() & ~data.geometry.is_empty & (data.geometry.area > 10 ** 2)].copy()
        data = data.to_crs(dst_crs)

        is_valid_geoms = data.is_valid
        if not is_valid_geoms.all():
            # reasons_invalidity = [f"{validation.explain_validity(g)}\n" for g in data.geometry[~is_valid_geoms]]
            # print(f"\tProduct {f} There are {(~is_valid_geoms).sum()} geoms invalid of {is_valid_geoms.shape[0]}\n {reasons_invalidity}")
            data = geodataframe_polygonsonly_valid(data)

        if data_all is None:
            data_all = data
        else:
            data_all = pd.concat([data_all, data], ignore_index=True)

    print(f"\t{len(floodmaps_paths)} Products joined {data_all.shape}")
    # Save as geojson
    data_all = data_all.dissolve(by="class").reset_index()
    print(f"\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Correctly dissolved. New shape: {data_all.shape}")
    data_all = data_all.explode(ignore_index=True)
    print(f"\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Correctly exploded. New shape: {data_all.shape}")

    data_all = data_all[~data_all.geometry.isna() & ~data_all.geometry.is_empty]
    print(f"\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Removed NA or empty polygons. New shape: {data_all.shape}")

    return data_all


def vectorize_jrc_permanent_water_layer(permanent_water:np.ndarray,
                                        crs:str, transform:rasterio.transform.Affine) -> Optional[gpd.GeoDataFrame]:
    """
    Vectorize cloud class

    Args:
        permanent_water: (H, W) array with predictions.
        https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_3_YearlyHistory?hl=en
        Values 0 nodata 1 not water, 2 seasonal water, 3: permanent water
        crs:
        transform:

    Returns:
        gpd.GeoDataFrame with column "class" with values "seasonal" and "permanent"
    """

    data_out = []
    for c, class_name in zip([2, 3],["seasonal", "permanent"]):
        geoms_polygons = get_water_polygons(permanent_water == c,
                                            transform=transform)
        if len(geoms_polygons) > 0:
            data_out.append(gpd.GeoDataFrame({"geometry": geoms_polygons,
                                              "class": class_name},
                                             crs=crs))

    if len(data_out) == 1:
        return data_out[0]
    elif len(data_out) > 1:
        return pd.concat(data_out, ignore_index=True)
    return None

def load_vectorized_permanent_water(aoi_folder:str, year:Optional[str] = None) -> Optional[gpd.GeoDataFrame]:
    """
    This function caches the vectorized permanent water product as a GeoJSON if it doesn't exist and return the vectorized
    product as a GeoDataFrame

    Args:
        aoi_folder: in this folder there should exist a subfolder called "PERMANENTWATERJRC" with the tif file
        (JRC water product exported from the GEE)
        https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_3_YearlyHistory?hl=en
        year: year to load (must exist in {aoi_folder}/PERMANENTWATERJRC/{year}.tif)

    Returns:
        gpd.GeoDataFrame with column "class" with values "seasonal" and "permanent"

    """
    fs = utils.get_filesystem(aoi_folder)
    if year is None:
        year = "*"
    files_permanent = fs.glob(os.path.join(aoi_folder, "PERMANENTWATERJRC", f"{year}.tif").replace("\\", "/"))
    if len(files_permanent) == 0:
        permanent_water_floodmap = None
    else:
        namefile_permanent = os.path.basename(os.path.splitext(files_permanent[0])[0])
        path_permanent_water_vec = os.path.join(aoi_folder, "PERMANENTWATERJRC_vec",
                                                namefile_permanent + ".geojson").replace("\\", "/")
        if not fs.exists(path_permanent_water_vec):
            with rasterio.open(f"gs://{files_permanent[0]}") as rst:
                permanent_raster = rst.read(1)
                crs = rst.crs
                transform = rst.transform

            permanent_water_floodmap = vectorize_jrc_permanent_water_layer(permanent_raster, crs=crs,
                                                                           transform=transform)
            if permanent_water_floodmap is not None:
                utils.write_geojson_to_gcp(path_permanent_water_vec, permanent_water_floodmap)
        else:
            permanent_water_floodmap = utils.read_geojson_from_gcp(path_permanent_water_vec)

    return permanent_water_floodmap

def add_permanent_water_to_floodmap(jrc_vectorized_map:gpd.GeoDataFrame, floodmap:gpd.GeoDataFrame,
                                    water_class:Optional[str]=None) -> gpd.GeoDataFrame:
    """
    Adds the "permanent" polygons of jrc_vectorized_map to floodmap

    Args:
        jrc_vectorized_map:
        floodmap:
        water_class:

    Returns:
        floodmap with permanent water polygons
    """
    classes = 'water-pre-flood' if floodmap.is_empty.all() else floodmap["class"].unique() 
    if water_class is None:
        if "water" in classes:
            water_class = "water"
        elif "water-pre-flood" in classes:
            water_class = "water-pre-flood"
        else:
            raise AttributeError(f"water_class not provided and we couldn't guess it")

    jrc_vectorized_map_copy = jrc_vectorized_map[jrc_vectorized_map["class"] == "permanent"].copy()
    jrc_vectorized_map_copy = jrc_vectorized_map_copy[["geometry", "class"]]
    if jrc_vectorized_map_copy.shape[0] == 0:
        return floodmap

    jrc_vectorized_map_copy["class"] = water_class
    jrc_vectorized_map_copy.to_crs(floodmap.crs, inplace=True)

    floodmap = pd.concat([floodmap, jrc_vectorized_map_copy], ignore_index=True)
    floodmap = geodataframe_polygonsonly_valid(floodmap)
    floodmap = floodmap.dissolve(by="class").reset_index()
    floodmap = floodmap.explode(ignore_index=True)

    # fix water in flood_trace or flood-trace classes
    if "flood_trace" in classes:
        class_flood_trace = "flood_trace"
    elif "flood-trace" in classes:
        class_flood_trace = "flood-trace"
    else:
        return floodmap

    water_polygon = unary_union(floodmap[floodmap["class"] == water_class].geometry)
    geoms_trace = floodmap[(floodmap["class"] == class_flood_trace)].geometry.apply(
        lambda g: g.difference(water_polygon))

    geoms_trace = geoms_trace[~geoms_trace.isna() & ~geoms_trace.is_empty]
    geoms_trace = geoms_trace.explode(ignore_index=True)
    geoms_trace = geoms_trace[geoms_trace.geometry.geom_type == "Polygon"]

    # Add back to floodmap
    floodmap = floodmap[floodmap["class"] != class_flood_trace].reset_index()

    floodmap_trace = gpd.GeoDataFrame(geometry=geoms_trace, crs=floodmap.crs)
    floodmap_trace["class"] = class_flood_trace

    floodmap = pd.concat([floodmap, floodmap_trace], ignore_index=True)
        
    return floodmap

