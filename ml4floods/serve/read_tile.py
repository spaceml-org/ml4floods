import rasterio
import rasterio.windows
import mercantile
from typing import List, Optional, Tuple, Any
from rasterio import warp
import numpy as np

SIZE = 256
WEB_MERCATOR_CRS = "epsg:3857"

PIXEL_PRECISION = 3
# Required because we cant read in parallel with rasterio
def pad_window(window: rasterio.windows.Window, pad_size) -> rasterio.windows.Window:
    """ Add the provided pad to a rasterio window object """
    return rasterio.windows.Window(window.col_off - pad_size[1],
                                   window.row_off - pad_size[0],
                                   width=window.width + 2 * pad_size[1],
                                   height=window.height + 2 * pad_size[1])


def round_outer_window(window:rasterio.windows.Window)-> rasterio.windows.Window:
    """ Rounds a rasterio.windows.Window object to outer (larger) window """
    return window.round_lengths(op="ceil", pixel_precision=PIXEL_PRECISION).round_offsets(op="floor",
                                                                                          pixel_precision=PIXEL_PRECISION)


def read_tile(tiff_path:str, z:int, x:int, y:int, size_output:Tuple[int,int]=(SIZE, SIZE),
              indexes:Optional[List[int]]=None, dst_nodata:Optional[Any]=None,
              resampling=warp.Resampling.cubic_spline) -> Optional[Tuple[np.ndarray, rasterio.transform.Affine]]:
    """
    Reads tile from tileserver location z, x, y from the raster tiff_path. This function reads from the pyramids
    of the tiff file if available.

    Args:
        tiff_path: path to the tiff file
        z: mercantile zoom level
        x: mercantile x coordinate
        y: mercantile y coordinate
        size_output: Size to read
        indexes: 1-based bands to read. input to rasterio read function of the raster.
        dst_nodata:
        resampling: Resampling to use

    Returns:
        numpy array with shape (len(indexes),) + size_output with the data read and affine transformation
    """

    bounds_wgs = mercantile.xy_bounds(int(x), int(y), int(z))

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, CPL_DEBUG=True), rasterio.open(tiff_path, 'r') as rst:
        x1, y1, x2, y2 = rst.bounds
        rst_bounds = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        if rst.crs != WEB_MERCATOR_CRS:
            bounds_read_crs_dest = warp.transform_bounds({"init": WEB_MERCATOR_CRS}, rst.crs,
                                                         left=bounds_wgs.left,
                                                         top=bounds_wgs.top,
                                                         bottom=bounds_wgs.bottom, right=bounds_wgs.right)
        else:
            bounds_read_crs_dest = (bounds_wgs.left, bounds_wgs.bottom, bounds_wgs.right, bounds_wgs.top)

        if rasterio.coords.disjoint_bounds(bounds_read_crs_dest, rst_bounds):
            return None

        rst_transform = rst.transform
        rst_crs = rst.crs
        rst_nodata = rst.nodata or dst_nodata
        rst_dtype = rst.meta["dtype"]

        # Compute the window to read and read the data
        window = rasterio.windows.from_bounds(*bounds_read_crs_dest, rst.transform)
        if rst.crs != WEB_MERCATOR_CRS:
            window = pad_window(window, (1, 1))  # Add padding for bicubic int or for co-registration
            window = round_outer_window(window)

        if indexes is None:
            output_shape_read = (rst.count, size_output[0] + 1, size_output[1] + 1)
            output_shape = (rst.count,) + size_output
        else:
            output_shape_read = (len(indexes), size_output[0] + 1, size_output[1] + 1)
            output_shape = (len(indexes),)+ size_output


        # With out_shape it reads from the pyramids if rst is a COG GeoTIFF
        src_arr = rst.read(indexes=indexes, window=window, out_shape=output_shape_read, boundless=True,
                           fill_value=rst_nodata)

    # Compute transform of readed data and reproject to WEB_MERCATOR_CRS
    input_output_factor = (window.height / output_shape_read[1], window.width / output_shape_read[2])

    transform_window = rasterio.windows.transform(window, rst_transform)
    transform_src = rasterio.Affine(transform_window.a * input_output_factor[1], transform_window.b, transform_window.c,
                                    transform_window.d, transform_window.e * input_output_factor[0], transform_window.f)

    transform_dst = rasterio.transform.from_bounds(west=bounds_wgs.left,north=bounds_wgs.top,
                                                   east=bounds_wgs.right, south=bounds_wgs.bottom,
                                                   width=size_output[0], height=size_output[1])

    rst_arr = np.empty(shape=output_shape, dtype=rst_dtype)

    rst_arr, rst_transform = warp.reproject(source=src_arr, destination=rst_arr,
                                            src_transform=transform_src, src_crs=rst_crs,
                                            src_nodata=rst_nodata,
                                            dst_transform=transform_dst, dst_crs=WEB_MERCATOR_CRS,
                                            dst_nodata=dst_nodata,
                                            resampling=resampling)
    return rst_arr, rst_transform