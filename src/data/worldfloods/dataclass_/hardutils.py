"""
Examples:

"""
from dataclasses import dataclass
from typing import Dict

import numpy as np
import rasterio


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


def open_source_tiff(dc: dataclass, **kwargs) -> np.ndarray:
    """Open tiff given a dataclass"""
    with rasterio.open(dc.full_path) as f:
        dc.source_tiff = f.read(**kwargs).tolist()  # .tobytes()
        #         dc.source_meta = f.meta.copy()
        return dc


# OPTIONAL STUFF
def open_with_rasterio(dc: dataclass, **kwargs) -> np.ndarray:
    """Open tiff given a dataclass"""
    with rasterio.open(dc.full_path) as f:
        return f.read(**kwargs)