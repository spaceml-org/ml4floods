"""
This script contains all the utility functions that are not specific to a particular kind of dataset.
These are mainly used for explorations, testing, and demonstrations.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import warnings

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from datetime import datetime

from ml4floods.data.config import CLASS_LAND_COPERNICUSEMSHYDRO
import pickle
import fsspec
from contextlib import contextmanager
import rasterio


def filter_land(gpddats: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Filter land from pandas dataframe (land class specified in hydrology maps from CopernicusEMS) """
    isnot_land = gpddats.obj_type.apply(
        lambda g: g not in CLASS_LAND_COPERNICUSEMSHYDRO
    )

    if np.sum(isnot_land) == gpddats.shape[0]:
        return gpddats

    gpddats_notland = gpddats[isnot_land].copy()
    if gpddats_notland.shape[0] == 0:
        return gpddats_notland

    land_geometries = gpddats.geometry[~isnot_land]
    land_geometries = unary_union(land_geometries.tolist())

    # Not all polygons are valid: filter to valid gpddats_notland
    gpddats_notland_valid = gpddats_notland[gpddats_notland["geometry"].is_valid]

    gpddats_notland_valid["geometry"] = gpddats_notland_valid.geometry.apply(
        lambda g: g.difference(land_geometries)
    )

    return gpddats_notland_valid


def filter_pols(gpddats: gpd.GeoDataFrame, pol_shapely: Polygon) -> gpd.GeoDataFrame:
    """ filter pols that do not intersects pol_shapely """
    gpddats_cp = gpddats[~(gpddats.geometry.isna() | gpddats.geometry.is_empty)].copy()

    return (
        gpddats_cp[gpddats_cp.geometry.apply(lambda g: g.intersects(pol_shapely))]
        .reset_index()
        .copy()
    )

class CustomJSONEncoder(json.JSONEncoder):

    def default(self, obj_to_encode):
        """Pandas and Numpy have some specific types that we want to ensure
        are coerced to Python types, for JSON generation purposes. This attempts
        to do so where applicable.
        """
        # Pandas dataframes have a to_json() method, so we'll check for that and
        # return it if so.
        if hasattr(obj_to_encode, "to_json"):
            return obj_to_encode.to_json()
        # Numpy objects report themselves oddly in error logs, but this generic
        # type mostly captures what we're after.
        if isinstance(obj_to_encode, np.generic):
            return obj_to_encode.item()
        # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        if isinstance(obj_to_encode, Polygon):
            return mapping(obj_to_encode)
        if isinstance(obj_to_encode, pd.Timestamp):
            return obj_to_encode.isoformat()
        if isinstance(obj_to_encode, datetime):
            return obj_to_encode.isoformat()
        # torch or tensorflow -> list, pretty straightforward.
        if hasattr(obj_to_encode, "numpy"):
            return obj_to_encode.numpy().tolist()
        # If none of the above apply, we'll default back to the standard JSON encoding
        # routines and let it work normally.
        return super().default(obj_to_encode)

REQUESTER_PAYS_DEFAULT = True

def get_filesystem(path: Union[str, Path], requester_pays:Optional[bool]=None):
    if requester_pays is None:
        requester_pays = REQUESTER_PAYS_DEFAULT
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0], requester_pays=requester_pays)
    else:
        # use local filesystem
        return fsspec.filesystem("file")


def write_geojson_to_gcp(gs_path: str, geojson_val: gpd.GeoDataFrame) -> None:
    fs = get_filesystem(gs_path)
    if geojson_val.shape[0] == 0:
        warnings.warn(f"Dataframe is empty. Saving in {gs_path}")
        with fs.open(gs_path, "w") as fh:
            json.dump(eval(geojson_val.to_json()), fh)
    else:
        with fs.open(gs_path, "wb") as fh:
            geojson_val.to_file(fh, driver="GeoJSON")


def check_gdal_requester_pays_gcp_available() -> bool:
    """
    Requester pays for GCP is available from GDAL 3.4

    rasterio issue: https://github.com/rasterio/rasterio/issues/1948
    Commit GDAL: https://github.com/OSGeo/gdal/pull/3883/commits/8d551953dea9290b21bd9747ec9ed22c81ca0409

    Returns:
        True if version >= 3.4

    """
    try:
        version_gdal = rasterio.__gdal_version__
        vnumber, vsubnumber = version_gdal.split(".")[:2]
        vnumber, vsubnumber = int(vnumber), int(vsubnumber)
        if (vnumber*100+vsubnumber) >= (3*100+4):
            return True
    except Exception as e:
        pass

    return False


REQUESTER_PAYS_AVAILABLE = check_gdal_requester_pays_gcp_available()


@contextmanager
def rasterio_open_read(tifffile:str, requester_pays:Optional[bool]=None) -> rasterio.DatasetReader:
    if requester_pays is None:
        requester_pays = REQUESTER_PAYS_DEFAULT

    if requester_pays and tifffile.startswith("gs"):
        if REQUESTER_PAYS_AVAILABLE:
            assert "GS_USER_PROJECT" in os.environ, \
                "'GS_USER_PROJECT' env variable not found and requester_pays=True set a project name to read rasters from the bucket" \
                "(i.e. -> export GS_USER_PROJECT='myprojectname')"

            with rasterio.open(tifffile) as src:
                yield src
        else:
            fs = fsspec.filesystem("gs", requester_pays=True)
            with fs.open(tifffile, "rb") as fh:
                with rasterio.io.MemoryFile(fh.read()) as mem:
                    yield mem.open()

    else:
        with rasterio.open(tifffile) as src:
            yield src


def read_geojson_from_gcp(gs_path: str) -> gpd.GeoDataFrame:
    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "rb") as fh:
        return gpd.read_file(fh)


def write_pickle_to_gcp(gs_path: str, dict_val: dict) -> None:

    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "wb") as fh:
        pickle.dump(dict_val, fh)


def read_pickle_from_gcp(gs_path:str) -> dict:
    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "rb") as fh:
        my_dictionary = pickle.load(fh)

    return my_dictionary


def write_json_to_gcp(gs_path: str, dict_val: dict) -> None:
    fs = get_filesystem(gs_path)

    with fs.open(gs_path, "w") as fh:
        json.dump(dict_val, fh, cls=CustomJSONEncoder)


def read_json_from_gcp(gs_path: str) ->Dict:
    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "r") as fh:
        my_dictionary = json.load(fh)

    return my_dictionary

