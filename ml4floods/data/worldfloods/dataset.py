import numbers

from ml4floods.preprocess.tiling import WindowSlices
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import rasterio.windows
import torch
from torch.utils.data import Dataset
import contextlib
from ml4floods.data import utils


from ml4floods.data.worldfloods.configs import BANDS_S2

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
        transforms: Optional[Callable] = None,
        bands: List[int] = list(range(len(BANDS_S2))),
        mndwi_indices: List[int] = None,
        lock_read: bool = False,
    ) -> None:

        self.image_files = image_files
        self.image_prefix = image_prefix
        self.gt_prefix = gt_prefix
        self.transforms = transforms
        self.bands_read = bands
        self.mndwi_indices = mndwi_indices
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

        image_tif = rasterio_read(
            image_name, self._lock, channels=[c + 1 for c in self.bands_read]
        )

        mask_tif = rasterio_read(y_name, self._lock)

        # get rid of nan, convert to float
        image = np.nan_to_num(image_tif).astype(np.float32)
        
        if self.mndwi_indices is not None:
            mndwi = (image[self.mndwi_indices][0] - image[self.mndwi_indices][1]) / (image[self.mndwi_indices][0] + image[self.mndwi_indices][1] + 1e-6)
            image = np.concatenate([image,mndwi[np.newaxis]], axis = 0)

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
        list_of_windows (List[WindowSlices]):  a list of
            namedtuples each consisting of a filename and a rasterio.window
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        transforms (Callable): the transformations used within the
            training data module

    Attributes:
        list_of_windows (List[WindowSlices]):  a list of
            namedtuples each consisting of a filename and a rasterio.window
        image_prefix (str): the input folder sub_directory
        gt_prefix (str): the target folder sub directory
        transforms (Callable): the transformations used within the
            training data module
        bands: List[int]
            0-based list of bands to read from BANDS_S2
    """

    def __init__(
        self,
        list_of_windows: List[WindowSlices],
        image_prefix: str = "/image_files/",
        gt_prefix: str = "/gt_files/",
        transforms: Optional[Callable] = None,
        bands: List[int] = list(range(len(BANDS_S2))),
        mndwi_indices: List[int] = None,
        lock_read: bool = False,
    ) -> None:

        self.image_prefix = image_prefix
        self.gt_prefix = gt_prefix
        self.transforms = transforms
        self.channels_read = bands
        self.mndwi_indices = mndwi_indices

        if lock_read:
            # Useful when reading from bucket
            self._lock = threading.Lock()
        else:
            self._lock = contextlib.nullcontext()

        self.list_of_windows = list_of_windows

    def get_label(self, idx: int) -> np.ndarray:
        """
        Method to read only the label. This function is useful for filtering the patches of the Dataset
        Args:
            idx:

        Returns:


        """
        sub_window = self.list_of_windows[idx]
        y_name = sub_window.file_name.replace(self.image_prefix, self.gt_prefix, 1)
        return rasterio_read(
            y_name,
            self._lock,
            channels=None,
            kwargs_rasterio={
                "window": sub_window.window,
                "boundless": True,
                "fill_value": 0,
            },
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
        sub_window = self.list_of_windows[idx]

        # get filename
        image_name = sub_window.file_name

        # replace string for image_prefix
        image_name = image_name.replace(self.gt_prefix, self.image_prefix, 1)

        # replace string for gt_prefix
        y_name = image_name.replace(self.image_prefix, self.gt_prefix, 1)

        # Open Image File
        image_tif = rasterio_read(
            image_name,
            self._lock,
            channels=[c + 1 for c in self.channels_read],
            kwargs_rasterio={
                "window": sub_window.window,
                "boundless": True,
                "fill_value": 0,
            },
        )
        mask_tif = rasterio_read(
            y_name,
            self._lock,
            channels=None,
            kwargs_rasterio={
                "window": sub_window.window,
                "boundless": True,
                "fill_value": 0,
            },
        )

        # get rid of nan, convert to float
        image = np.nan_to_num(image_tif).astype(np.float32)
        
        if self.mndwi_indices is not None:
            mndwi = (image[self.mndwi_indices][0] - image[self.mndwi_indices][1]) / (image[self.mndwi_indices][0] + image[self.mndwi_indices][1] + 1e-6)
            image = np.concatenate([image,mndwi[np.newaxis]], axis = 0)
        mask = np.nan_to_num(mask_tif).astype(int)

        # Apply transformation
        if self.transforms is not None:
            data = self.transforms(image=image, mask=mask)
        else:
            data = {"image": image, "mask": mask}
        # return x, y
        return data

    def __len__(self) -> int:
        return len(self.list_of_windows)


def rasterio_read(
    image_name: str, lock, channels: List[int] = None, kwargs_rasterio: Dict = {}
) -> np.ndarray:
    with lock:
        with utils.rasterio_open_read(image_name) as f:
            im_tif = f.read(channels, **kwargs_rasterio)

    return im_tif


def load_input(tiff_input:str, channels:Union[List[int],List[str]],
               window:Optional[rasterio.windows.Window]=None) -> Tuple[torch.Tensor, rasterio.transform.Affine]:
    """
    Reads from a tiff the specified channel and window.

    Args:
        tiff_input: path to geotiff file
        window: rasterio.Window object to read (None to read all)
        channels: 0-based channels to read or names of the bands (we will use `.descriptions` to find the band
            names of the `tiff_input`).

    Returns:
        3-D tensor (len(channels), H, W), Affine transform to geo-reference the array read.

    """
    with utils.rasterio_open_read(tiff_input) as rst:
        if isinstance(channels[0], numbers.Number):
            indexes = (np.array(channels) + 1).tolist()
        else:
            channels_tiff = list(rst.descriptions)
            indexes = [channels_tiff.index(c) + 1 for c in channels]

        inputs = rst.read(indexes, window = window)

        # Shifted transform based on the given window (used for plotting)
        transform = rst.transform if window is None else rasterio.windows.transform(window, rst.transform)
        torch_inputs = torch.tensor(np.nan_to_num(inputs).astype(np.float32))
    return torch_inputs, transform