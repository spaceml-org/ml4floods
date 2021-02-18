from rasterio.io import DatasetReader
from itertools import product
from rasterio import windows
import rasterio


def get_tiles(
    ds: rasterio.io.DatasetReader, height: int = 128, width: int = 123
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
            col_off=col_offset,
            row_off=row_offset,
            width=width,
            height=height,
        )
        yield window
