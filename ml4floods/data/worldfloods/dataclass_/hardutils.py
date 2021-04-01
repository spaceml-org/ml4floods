"""
Examples:

"""
import json
from dataclasses import dataclass
from typing import Dict, List

import geojson
import numpy as np
import rasterio

from ml4floods.data.utils import open_file_from_bucket, parse_gcp_path


def store_tiff_to_bytes(dc: dataclass) -> dataclass:
    """Create the full path from the info within the dataclass"""
    full_path = "gs://" + dc.bucket_name + "/" + dc.filename
    dc.full_path = full_path
    return dc


def create_dataclass_fullpath(dc: dataclass) -> dataclass:
    """Create the full path from the info within the dataclass"""
    full_path = "gs://" + dc.bucket_name + "/" + dc.filename
    dc.full_path = full_path
    return dc


def grab_dict(dc: dataclass, meta_dict: Dict) -> dataclass:
    """input metadata from the tiff"""
    dc.meta_data = meta_dict
    return dc


def open_source_meta(dc: dataclass, **kwargs) -> dataclass:
    """Open tiff given a dataclass"""

    full_path = dc.full_path.replace("S2", "S2metadata")
    full_path = full_path.rsplit(".", 1)[0] + ".geojson"

    dc.meta_data = open_file_from_bucket(full_path)
    return dc


def open_source_tiff(dc: dataclass, **kwargs) -> List:
    """Open tiff given a dataclass"""
    with rasterio.open(dc.full_path) as f:
        dc.source_tiff = f.read(**kwargs).tolist()  # .tobytes()
        #         dc.source_meta = f.meta.copy()
        return dc


def open_source_tiff_meta(dc: dataclass, **kwargs) -> np.ndarray:
    """Open tiff given a dataclass"""
    with rasterio.open(dc.full_path) as f:
        dc.source_tiff_meta = f.meta  # .tobytes()
        #         dc.source_meta = f.meta.copy()
        return dc


# OPTIONAL STUFF
def open_with_rasterio(dc: dataclass, **kwargs) -> np.ndarray:
    """Open tiff given a dataclass"""
    with rasterio.open(dc.full_path) as f:
        return f.read(**kwargs)


def parse_gcp_files_dataclass(dc):
    """Parse names for essential paths for easier calls"""
    bucket_id, file_path, file_name = parse_gcp_path(dc.full_path)
    dc.bucket_id = bucket_id
    dc.file_path = file_path
    dc.file_name = file_name
    return dc