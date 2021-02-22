import math
import os
import sys
from typing import List

import rasterio
from src.preprocess.tiling import WindowSlices, WindowSize, get_window_tiles

from pyprojroot import here

SRC_DIR = here(project_files=[".here"])


def get_list_of_window_slices(
    file_names: List[str], window_size: WindowSize
) -> List[WindowSlices]:

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
