import logging
import pandas as pd
import numpy as np
from rasterio import features
import rasterio
import geopandas as gpd
import os
from glob import glob
from src.data.utils import filter_pols, filter_land
from src.data import utils
from typing import Optional

import rasterio.windows

"""
Map from names in the floodmap to rasterised value of watermask (see function compute_water)

The meaning of the codes are: 
 {0: 'land', 1: 'flood', 2: 'hydro', 3: 'permanent_water_jrc'}

"""
CODES_FLOODMAP = {
    # CopernicusEMS (flood)
    'Flooded area': 1,
    'Not Applicable': 1,
    'Flood trace': 1,
    'Dike breach': 1,
    'Standing water': 1,
    'Erosion': 1,
    'River': 2,
    'Riverine flood': 1,
    # CopernicusEMS (hydro)
    'BH140-River': 2,
    'BH090-Land Subject to Inundation': 2,
    'BH080-Lake': 2,
    'BA040-Open Water': 2,
    # 'BA030-Island': 2, islands are excluded! see filter_land func
    'BH141-River Bank': 2,
    'BH130-Reservoir': 2,
    'BH141-Stream': 2,
    # UNOSAT
    "preflood water": 2,
    # "Flooded area": 1,  # 'flood water' DUPLICATED
    "flood-affected land / possible flood water": 1,
    # "Flood trace": 1,  # 'probable flash flood-affected land' DUPLICATED
    "satellite detected water": 1,
    # "Not Applicable": 1,  # unknown see document DUPLICATED
    "possible saturated, wet soil/ possible flood water": 1,
    "aquaculture (wet rice)": 1,
    "tsunami-affected land": 1,
    "ran of kutch water": 1,
    "maximum flood water extent (cumulative)": 1
}


# Unosat names definition https://docs.google.com/document/d/1i-Fz0o8isGTpRr39HqvUOQBs0yh8_Pz_WcyF5JK0bM0/edit#heading=h.3neqeg3hyp0t
UNOSAT_CLASS_TO_TXT = {
    0: "preflood water",
    1: "Flooded area",  # 'flood water'
    2: "flood-affected land / possible flood water",
    3: "Flood trace",  # 'probable flash flood-affected land'
    4: "satellite detected water",
    5: "Not Applicable",  # unknown see document
    6: "possible saturated, wet soil/ possible flood water",
    9: "aquaculture (wet rice)",
    14: "tsunami-affected land",
    77: "ran of kutch water",
    99: "maximum flood water extent (cumulative)"
}

def generate_floodmap_v1(register, filename_floodmap, pol_bounds_s2, worldfloods_root, filterland=True):
    """ Generates a floodmap (shapefile) with the joined info of the hydro and flood content. """

    mapdf = filter_pols(gpd.read_file(os.path.join(worldfloods_root, "maps/", register["resource folder"], register["layer name"],
                                                   "map.shp")),
                        pol_bounds_s2)
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

    floodmap = gpd.GeoDataFrame({"geometry": mapdf.geometry},
                                crs=mapdf.crs)
    floodmap["w_class"] = column_water_class
    floodmap["source"] = "flood"

    if "hydro" in register:
        register_hydro = register["hydro"]
        assert register_hydro["source"] == "CopernicusEMS", f"Unexpected hydro file for source {register}"
        mapdf_hydro = filter_pols(gpd.read_file(os.path.join(worldfloods_root, "maps/", register_hydro["resource folder"], register_hydro["layer name"],
                                                             "map.shp")),
                                  pol_bounds_s2)

        mapdf_hydro = filter_land(mapdf_hydro) if filterland and (mapdf_hydro.shape[0] > 0) else mapdf_hydro
        if mapdf_hydro.shape[0] > 0:
            mapdf_hydro["source"] = "hydro"
            mapdf_hydro = mapdf_hydro.rename({"obj_type": "w_class"}, axis=1)
            mapdf_hydro = mapdf_hydro[["geometry", "w_class", "source"]].copy()
            floodmap = pd.concat([floodmap, mapdf_hydro], axis=0, ignore_index=True)

    floodmap.loc[floodmap.w_class.isna(), 'w_class'] = "Not Applicable"

    if filename_floodmap is not None:
        # Remove files if exist
        name_to_glob = filename_floodmap.replace(".shp", ".*")
        for fremove in glob(name_to_glob):
            os.remove(fremove)

        floodmap.to_file(filename_floodmap)

    return floodmap


def compute_water(tiffs2:str, floodmap:gpd.GeoDataFrame, window: Optional[rasterio.windows.Window]=None,
                  permanent_water_path:str=None):
    """
    Rasterise flood map and add JRC permanent water layer

    :param tiffs2: Tif file S2 (either remote or local)
    :param floodmap: geopandas dataframe with the annotated polygons
    :param window: rasterio.windows.Window to read. Could also be slices (slice(100, 200), slice(100, 200)
    :param permanent_water_path: Whether or not user JRC permanent water layer (from tifffimages/PERMANENTWATERJRC folder)

    :return: water_mask : np.uint8 raster same shape as tiffs2 {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
    """


    # area_of_interest contains the extent that was labeled (values out of this pol should be marked as invalid)
    floodmap_aoi = floodmap[floodmap["w_class"] == "area_of_interest"]
    if floodmap_aoi.shape[0] > 0:
        floodmap_rasterise = floodmap[floodmap["w_class"] != "area_of_interest"]

    shapes_rasterise = ((g, CODES_FLOODMAP[w]) for g, w in floodmap_rasterise[['geometry', 'w_class']].itertuples(index=False,
                                                                                                                  name=None))

    with rasterio.open(tiffs2) as src_s2:
        if window is None:
            out_shape = src_s2.shape
            transform = src_s2.transform
        else:
            out_shape = rasterio.windows.shape(window, height=src_s2.height, width=src_s2.width)
            transform = rasterio.windows.transform(window, src_s2.transform)

    water_mask = features.rasterize(shapes=shapes_rasterise, fill=0,
                                    out_shape=out_shape,
                                    dtype=np.int16,
                                    transform=transform)

    # Load valid mask using the area_of_interest polygons (valid pixels are those within area_of_interest polygons)
    if floodmap_aoi.shape[0] > 0:
        shapes_rasterise = ((g, 1) for g, w in
                            floodmap_aoi[['geometry', 'w_class']].itertuples(index=False,
                                                                             name=None))
        valid_mask = features.rasterize(shapes=shapes_rasterise, fill=0,
                                        out_shape=out_shape,
                                        dtype=np.uint8,
                                        transform=transform)
        water_mask[valid_mask == 0] = -1

    if permanent_water_path is not None:
        logging.info("\t Adding permanent water")
        permament_water = rasterio.open(permanent_water_path).read(1, window=window)

        # Set to permanent water
        water_mask[(water_mask != -1) & (permament_water == 3)] = 3

        # Seasonal water (permanent_water == 2) will not be used

    return water_mask


def load_compute_cloud_mask(tiffs2, s2_img=None, save_mask=False, window=None):
    """load or compute cloud mask for s2 image """

    filename_cloudmask = tiffs2.replace("/S2/","/cloudprob/")

    # Check if _edited file exists and use that
    filename_cloudmask_edited = filename_cloudmask.replace("/cloudprob/", "/cloudprob_edited/")

    if utils.check_file_in_bucket_exists(filename_cloudmask_edited):
        logging.info("\t Loading cloud mask edited file")
        clouds = rasterio.open(filename_cloudmask_edited).read(1, window=window)
    elif utils.check_file_in_bucket_exists(filename_cloudmask):
        clouds = rasterio.open(filename_cloudmask).read(1, window=window)
    else:
        logging.info("\t Cloud mask not found. Computing")
        from src.data import cloud_masks
        with rasterio.open(tiffs2) as src_s2:
            if s2_img is None:
                s2_img = src_s2.read(window=window)

            if not save_mask:
                filename_cloudmask = None
            else:
                assert not filename_cloudmask.startswith("gs://"), "Cannot save directly in the bucket"
                logging.info("\t Cloud mask will be saved")
            clouds = cloud_masks.compute_cloud_mask_save(filename_cloudmask, s2_img,
                                                         src_s2.profile)

    return clouds


def generate_gt(s2_img, clouds, water_mask):
    """

    :param s2_img: S2 12 band image
    :param clouds:  probability value between [0,1]
    :param water_mask:  {-1: invalid, 0: land, 1: flood, 2: hydro, 3: permanentwaterjrc}
    :return:
    """
    invalids = np.all(s2_img == 0, axis=0) & (water_mask == -1)

    # Set cloudprobs to zero in invalid pixels
    clouds[invalids] = 0
    cloudmask = clouds > .5

    # Set watermask values for compute stats
    water_mask[invalids] = 0
    water_mask[cloudmask] = 0

    # Create gt mask {0: invalid, 1:land, 2: water, 3: cloud}
    gt = np.ones(water_mask.shape, dtype=np.uint8)
    gt[water_mask > 0] = 2
    gt[cloudmask] = 3
    gt[invalids] = 0

    return gt