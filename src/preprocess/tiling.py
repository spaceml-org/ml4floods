from collections import namedtuple
from itertools import product
from pathlib import Path
from typing import Callable, List, Optional

import rasterio
from rasterio import windows
from rasterio.io import DatasetReader

WindowSize = namedtuple("WindowSize", ["height", "width"])
WindowSlices = namedtuple("WindowSlices", ["file_name", "window"])


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
