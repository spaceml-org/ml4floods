from collections import namedtuple
from itertools import product
from pathlib import Path
from typing import Callable, List, Optional, Dict

import rasterio
from rasterio import windows
from rasterio.io import DatasetReader
import json

WindowSize = namedtuple("WindowSize", ["height", "width"])
WindowSlices = namedtuple("WindowSlices", ["file_name", "window"])


def load_windows(filename:str) -> List[WindowSlices]:
    with open(filename, "r") as fh:
        list_of_windows = [Dict_to_WindowSlices(dictio) for dictio in json.load(fh)["slices"]]
    return list_of_windows

def save_windows(list_windows:List[WindowSlices], filename:str) -> None:
    list_save = [WindowSlices_to_Dict(ws) for ws in list_windows]
    with open(filename, "w") as fh:
        json.dump({"slices": list_save}, fh)


def WindowSlices_to_Dict(ws: WindowSlices) -> Dict:
    return {
        "file_name" : ws.file_name,
        "window": {
            "col_off" : ws.window.col_off,
            "row_off": ws.window.row_off,
            "width": ws.window.width,
            "height": ws.window.height,
        }
    }

def Dict_to_WindowSlices(ds: Dict) -> WindowSlices:
    return WindowSlices(file_name=ds["file_name"],
                        window=windows.Window(col_off=ds["window"]["col_off"],
                                              row_off=ds["window"]["row_off"],
                                              width=ds["window"]["width"],
                                              height=ds["window"]["height"]))



def yield_window_tiles(
    ds: rasterio.io.DatasetReader, height: int = 128, width: int = 128, **kwargs
) -> rasterio.windows.Window:
    """a generator for rasterio specific slices given a rasterio dataset

    Args:
        ds (rasterio.io.DatasetReader): a rasterio dataset object
        height (int): the height for the slice
        width (int): the width for the slice

    Yields:
        window (rasterio.windows.Window): slicing
    """
    # extract the row height from the dataset
    n_columns, n_rows = ds.meta["width"], ds.meta["height"]

    # create the offsets
    offsets = product(range(0, n_columns, width), range(0, n_rows, height))

    for col_offset, row_offset in offsets:
        window = windows.Window(
            col_off=col_offset, row_off=row_offset, width=width, height=height, **kwargs
        )
        yield window


def get_window_tiles(
    ds: rasterio.io.DatasetReader, height: int = 128, width: int = 128, **kwargs
) -> List[rasterio.windows.Window]:
    """a generator for rasterio specific slices given a rasterio dataset

    Args:
        ds (rasterio.io.DatasetReader): a rasterio dataset object
        height (int): the height for the slice
        width (int): the width for the slice

    Yields:
        window (rasterio.windows.Window): slicing
    """
    # extract the row height from the dataset
    n_columns, n_rows = ds.meta["width"], ds.meta["height"]

    # create the offsets
    offsets = product(range(0, n_columns, width), range(0, n_rows, height))
    list_of_windows = []
    for col_offset, row_offset in offsets:
        iwindow = windows.Window(
            col_off=col_offset, row_off=row_offset, width=width, height=height, **kwargs
        )
        list_of_windows.append(iwindow)

    return list_of_windows


def save_tiles(
    file_name: str,
    dest_dir: str,
    bands: List[str],
    window_size: WindowSize,
    verbose: bool = False,
    n_samples: Optional[int] = None,
) -> None:
    """does tiling from a predefined window size and saves them to a directory

    Args:
        file_name (str): the filename to be opened by rasterio
        dest_dir (str): the directory where the data will be saved
        bands (List[str]): a list of bands to be accessed and saved
        window_size (WindowSize): a namedtuple with the height and width
        verbose (bool): flag to allow one to view the files saved

    Returns:
        None
    """
    with rasterio.open(file_name) as dataset:

        itile = 0

        # copy the metadata
        window_meta = dataset.meta.copy()

        for window_tile in get_window_tiles(
            dataset, window_size.height, window_size.width
        ):

            # copy the meta data
            window_meta["width"] = window_tile.width
            window_meta["height"] = window_tile.height
            window_meta["channels"] = bands
            window_meta["transform"] = windows.transform(window_tile, dataset.transform)

            # open the dataset with only the window selected
            sub_image = dataset.read(indexes=bands, window=window_tile)

            # create unique filename for the tile
            window_file_name = (
                f"{str(Path(file_name).stem)}_tile_{itile}{str(Path(file_name).suffix)}"
            )

            # filepath for saving
            output_tile_file_name = str(Path(dest_dir).joinpath(window_file_name))

            # open file and also save meta data
            with rasterio.open(output_tile_file_name, "w", **window_meta) as out_f:
                out_f.write(
                    dataset.read(window=window_tile, boundless=True, fill_value=0)
                )

            itile += 1
            if verbose:
                print(f"Saved: {window_file_name}")

            if n_samples is not None and n_samples == itile:
                break
    return None
