import os
import random
from datetime import datetime
from pathlib import Path
from src.preprocess.tiling import WindowSize
from typing import Callable, Dict, List, Optional, Tuple


import numpy as np
import rasterio
import rasterio.windows
from torch.utils.data import Dataset
import contextlib

from src.data.worldfloods.configs import BANDS_S2
from src.preprocess.utils import get_list_of_window_slices

import threading




class WorldFloodsDataset(Dataset):
    """A prepackaged WorldFloods PyTorch Dataset
    This initializes the dataset given a set a set of image files with a 
    subdirectory for the training and testing data ("image_prefix" and "gt_prefix").
    
    Args:
        image_files (List[str]): the image files to be loaded into the 
            dataset
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        transforms (Callable): the transformations used within the 
            training data module
            
    Attributes:
        image_files (List[str]): the image files to be loaded into the 
            dataset
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        transforms (Callable): the transformations used within the 
            training data module
        bands: List[int]
            0-based list of bands to read from BANDS_S2
    """

    def __init__(
        self,
        image_files: List[str],
        image_prefix: str = "/image_files/",
        gt_prefix: str = "/gt_files/",
        transforms: Optional[List[Callable]] = None,
        bands: List[int] = list(range(len(BANDS_S2))),
        lock_read:bool=False
    ) -> None:

        self.image_files = image_files
        self.image_prefix = image_prefix
        self.gt_prefix = gt_prefix
        self.transforms = transforms
        self.bands_read = bands
        if lock_read:
            self._lock = threading.Lock()
        else:
            self._lock = contextlib.nullcontext()

        # sort to make sure that the order is deterministic
        # (order of the flow of data points to the ML model)
        # TODO: Do this for the list of filepaths at the end as well
        self.image_files.sort()

    def __getitem__(self, idx: int) -> Dict:
        """Index to select an image
        
        Args:
            idx (int): index
        
        Returns:
            a dictionary with the image and mask keys
            {"image", "mask"}
        """

        # get filenames
        image_name = self.image_files[idx]

        y_name = image_name.replace(self.image_prefix, self.gt_prefix, 1)

        image_tif = rasterio_read(image_name, self._lock,
                                  channels=[c + 1 for c in self.bands_read])

        mask_tif = rasterio_read(y_name, self._lock)

        # get rid of nan, convert to float
        image = np.nan_to_num(image_tif).astype(np.float32)

        # The 0-index comes from reading all the bands with f.read()
        mask = np.nan_to_num(mask_tif)

        # Apply transformation
        if self.transforms is not None:
            data = self.transforms(image=image, mask=mask)
        else:
            data = {"image": image, "mask": mask}
        # return x, y
        return data

    def __len__(self) -> int:
        return len(self.image_files)


class WorldFloodsDatasetTiled(Dataset):
    """A prepackaged WorldFloods PyTorch Dataset
    This initializes the dataset given a set a set of image files with a 
    subdirectory for the training and testing data ("image_prefix" and "gt_prefix").
    This also does the tiling under the hood given the windowsize.
    
    Args:
<<<<<<< HEAD
        image_files (List[str]): list of specific image files
            e.g., path/to/file/prefix/filename
        image_prefix (str): prefix for images
        gt_prefix (str): prefix for groundtruth
        window_size (Tuple[int,int]): tuple for window size for sliing
            height,width
        transforms (List[Callable]): the transformations to be done later
            for each tile.

    Attributes
    ----------
    window_size: tuple(int, int)
            size of the tiling window
    image_prefix: str
            the subdirectory name for the images
    gt_prefix: str
            the subdirectory name for the groundtruth
    bands: List[int]
            0-based list of bands to read from BANDS_S2

=======
        image_files (List[str]): the image files to be loaded into the 
            dataset
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        window_size (Tuple[int,int]): the window sizes (height, width) to be
            used for the tiling
        transforms (Callable): the transformations used within the 
            training data module
            
    Attributes:
        image_files (List[str]): the image files to be loaded into the 
            dataset
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        window_size (namedtuple): a tuple with the height and width
            arguments
        transforms (Callable): the transformations used within the 
            training data module
        accumulated_list_of_windows_test (List[namedtuple]): a list of
            namedtuples each consisting of a filename and a rasterio.window
>>>>>>> main
    """

    def __init__(
        self,
        image_files: List[str],
        image_prefix: str = "/image_files/",
        gt_prefix: str = "/gt_files/",
        window_size: Tuple[int, int] = (64, 64),
        transforms: Optional[Callable] = None,
        bands: List[int] = list(range(len(BANDS_S2))),
        lock_read: bool = False
    ) -> None:

        self.image_files = image_files
        self.image_prefix = image_prefix
        self.gt_prefix = gt_prefix
        self.transforms = transforms
        self.window_size = WindowSize(height=window_size[0], width=window_size[1])
        self.channels_read = bands

        if lock_read:
            # Useful when reading from bucket
            self._lock = threading.Lock()
        else:
            self._lock = contextlib.nullcontext

        # sort to make sure that the order is deterministic
        # (order of the flow of data points to the ML model)
        # TODO: Do this for the list of filepaths at the end as well
        self.image_files.sort()
        # get the image slices
        self.accumulated_list_of_windows_test = get_list_of_window_slices(
            self.image_files, window_size=self.window_size
        )

    def __getitem__(self, idx: int) -> Dict:
        """Index to select an image tile
        
        Args:
            idx (int): index
        
        Returns:
            a dictionary with the keys to the image tiles and mask tiles
            {"image", "mask"}
        """
        # get filenames from named tuple
        sub_window = self.accumulated_list_of_windows_test[idx]

        # get filename
        image_name = sub_window.file_name

        # replace string for image_prefix
        image_name = image_name.replace(self.gt_prefix, self.image_prefix, 1)

        # replace string for gt_prefix
        y_name = image_name.replace(self.image_prefix, self.gt_prefix, 1)

        # Open Image File
        image_tif = rasterio_read(image_name, self._lock, channels=[c+1 for c in self.channels_read],
                                  kwargs_rasterio={"window": sub_window.window, "boundless":True, "fill_value": 0})
        mask_tif = rasterio_read(y_name, self._lock, channels=None,
                                 kwargs_rasterio={"window": sub_window.window, "boundless":True, "fill_value": 0})

        # get rid of nan, convert to float
        image = np.nan_to_num(image_tif).astype(np.float32)

        mask = np.nan_to_num(mask_tif)

        # Apply transformation
        if self.transforms is not None:
            data = self.transforms(image=image, mask=mask)
        else:
            data = {"image": image, "mask": mask}
        # return x, y
        return data

    def __len__(self) -> int:
        return len(self.accumulated_list_of_windows_test)


def rasterio_read(image_name, lock, channels=None, kwargs_rasterio={}):
    with lock:
        with rasterio.open(image_name) as f:
            im_tif = f.read(channels, **kwargs_rasterio)

    return im_tif
