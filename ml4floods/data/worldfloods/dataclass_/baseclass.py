from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import numpy as np


@dataclass
class WorldFloodsS2Image:
    """Simple WFImage dataclass"""

    # ESSENTIAL METADATA
    filename: str
    bucket_name: str
    full_path: str = field(default="Not Specified")

    # PAYLOAD
    #     source_tiff: bytes = field(default = "Not Specified")
    meta_data: Dict = field(default="Not Specified")

    # BREADCRUMBS
    load_date: str = field(default=str(datetime.now()))
    viewed_by: list = field(default_factory=list, compare=False, repr=False)
    source_system: str = field(default="Not Specified")


@dataclass
class WorldFloodsS2ImageSaved:
    """Simple WFImage dataclass"""

    # ESSENTIAL METADATA
    full_path: str
    file_name: str = field(default="Not Specified")
    bucket_name: str = field(default="Not Specified")
    file_path: str = field(default="Not Specified")

    # PAYLOAD
    source_tiff: List = field(default="Not Specified")
    source_tiff_meta: Dict = field(default="Not Specified")
    meta_data: Dict = field(default="Not Specified")

    # BREADCRUMBS
    load_date: str = field(default=str(datetime.now()))
    viewed_by: list = field(default_factory=list, compare=False, repr=False)
    source_system: str = field(default="Not Specified")

    def __repr__(self):
        return f"Filepath: {self.full_path}\nLoad Date: {self.load_date}"

    def __str__(self):
        return f"Filepath: {self.full_path}\nLoad Date: {self.load_date}"


@dataclass
class WorldFloodsL8Image:
    """Simple WFImage dataclass"""

    # ESSENTIAL METADATA
    filename: str
    bucket_name: str
    full_path: str = field(default="Not Specified")

    # PAYLOAD
    #     source_tiff: bytea = field(default = "Not Specified")
    meta_data: Dict = field(default="Not Specified")

    # BREADCRUMBS
    load_date: str = field(default=datetime.now())
    viewed_by: list = field(default_factory=list, compare=False, repr=False)
    source_system: str = field(default="Not Specified")


@dataclass
class WorldFloodsCloudProb:
    """Simple WFImage dataclass"""

    # ESSENTIAL METADATA
    filename: str
    bucket_name: str
    full_path: str = field(default="Not Specified")

    # PAYLOAD
    #     source_tiff: bytea = field(default = "Not Specified")
    meta_data: Dict = field(default="Not Specified")

    # BREADCRUMBS
    load_date: str = field(default=datetime.now())
    viewed_by: list = field(default_factory=list, compare=False, repr=False)
    source_system: str = field(default="Not Specified")