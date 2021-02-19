import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.windows
from torch.utils.data import Dataset

from src.data.utils import check_path_exists
from src.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from src.data.worldfloods.prepare_data import prepare_data_func


from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class WorldFloodsImage:
    # ESSENTIAL METADATA
    filename: str
    uri: str = field(default=None)
    filepath: str = field(default=None)
    bucket_id: str = field(default=None)
    product_id: str = field(default=None)

    # BREADCRUMBS
    load_date: str = field(default=datetime.now())
    viewed_by: list = field(default_factory=list, compare=False, repr=False)
    source_system: str = field(default="Not Specified")


class WorldFloodsDataset(Dataset):
    """
    A dataloader for the WorldFloods dataset.

    Attributes
    ----------
    window_size: tuple(int, int)
            size of the tiling window
    image_prefix: str
            the subdirectory name for the images
    gt_prefix: str
            the subdirectory name for the groundtruth

    """

    def __init__(
        self,
        image_files: List[str],
        image_prefix: str = "/image_files/",
        gt_prefix: str = "/gt_files/",
        transforms: Optional[List[Callable]]=None
    ):

        self.image_files = image_files
        self.image_prefix = image_prefix
        self.gt_prefix = gt_prefix
        self.transforms = transforms


        # sort to make sure that the order is deterministic
        # (order of the flow of data points to the ML model)
        # TODO: Do this for the list of filepaths at the end as well
        self.image_files.sort()

    def __getitem__(self, idx):
        
        # get filenames
        x_name = self.image_files[idx]
        y_name = x_name.replace(self.image_prefix, self.gt_prefix, 1)
        
        # Open Image File
        with rasterio.open(x_name) as f:
            x_tif = f.read()
            
            # permute channels because of rasterio
            x_tif = np.transpose(x_tif, (1, 2, 0))
            
        # Open Ground Truth File
        with rasterio.open(y_name) as f:
            y_tif = f.read()
            
            # permute channels because of rasterio
            y_tif = np.transpose(y_tif, (1, 2, 0))
        
        # get rid of nan, convert to float
        x = np.nan_to_num(x_tif).astype(np.float32)
        
        # TODO: Need to check why the 0th index.
        y = np.nan_to_num(y_tif)
        
        # data = {"image": x, "mask": y}

        # Apply transformation
        if self.transforms is not None:
            res = self.transforms(image=x, mask=y)
            # x, y = res["image"], res["mask"]
        
        # get the channels
        # return x, y
        return res