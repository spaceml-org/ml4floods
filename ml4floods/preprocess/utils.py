import math
import os
import sys
from typing import List

import rasterio

from ml4floods.preprocess.tiling import WindowSize, WindowSlices, get_window_tiles


def get_list_of_window_slices(
    file_names: List[str], window_size: WindowSize
) -> List[WindowSlices]:
    """Function to return the list of window slices for the all the
    input images and the given window size.

    Args:
        file_names (List[str]): List of filenames that are to be sliced.
        window_size (WindowSize): Window size of the tiles.

    Returns:
        List[WindowSlices]: List of window slices for the each tile
        corresponding to each input image.
    """

    accumulated_list_of_windows = []
    for ifilename in file_names:

        with rasterio.open(ifilename) as dataset:
            # get list of windows
            list_of_windows = get_window_tiles(
                dataset, height=window_size.height, width=window_size.width
            )
            # create a list of filenames
            list_of_windows = [
                WindowSlices(file_name=ifilename, window=iwindow)
                for iwindow in list_of_windows
            ]

        accumulated_list_of_windows += list_of_windows

    return accumulated_list_of_windows
