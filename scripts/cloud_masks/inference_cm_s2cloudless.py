import argparse
import fsspec
from ml4floods.data import utils, vectorize
from datetime import datetime
from typing import Optional
import rasterio.windows
import numpy as np
import sys
import warnings
import traceback
from skimage.morphology import disk, binary_opening
import geopandas as gpd


def vectorize_output(binary_mask:np.ndarray, crs:str, transform:rasterio.transform.Affine) -> Optional[gpd.GeoDataFrame]:
    """
    Vectorize cloud class

    Args:
        binary_mask: (H, W) array with predictions. Values 0 clear 1 cloud.
        crs:
        transform:

    Returns:
        gpd.GeoDataFrame with vectorized cloud, shadows and thick and thin clouds classes
    """

    geoms_polygons = vectorize.get_polygons(binary_mask,
                                            transform=transform)
    if len(geoms_polygons) > 0:
        return gpd.GeoDataFrame({"geometry": geoms_polygons,
                                 "id": np.arange(0, len(geoms_polygons)),
                                 "class": "CLOUD"},
                                    crs=crs)
    return None


def main(cems_code:str, aoi_code:str):
    tiff_files = fs.glob(f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/*{cems_code}/*{aoi_code}/S2/*.tif")

    files_with_errors = []
    for total, filename in enumerate(tiff_files):
        filename = f"gs://{filename}"
        filename_save_vect = filename.replace("/S2/", "/cms2cloudless_vec/").replace(".tif", ".geojson")

        if fs.exists(filename_save_vect):
            continue

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({total}/{len(tiff_files)}) Processing {filename}")

        try:
            with utils.rasterio_open_read(filename) as rst:
                band_index = rst.descriptions.index("probability") + 1
                pred = rst.read(band_index)
                crs  = rst.crs
                transform = rst.transform

            pred = binary_opening(pred > 50, disk(3))
            data_out = vectorize_output(pred, crs, transform)

            if data_out is not None:
                utils.write_geojson_to_gcp(filename_save_vect, data_out)

        except Exception:
            warnings.warn(f"Failed {filename}")
            traceback.print_exc(file=sys.stdout)
            files_with_errors.append(filename)

    if len(files_with_errors) > 0:
        print(f"Files with errors:\n {files_with_errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Vectorize s2cloudless outputs')
    parser.add_argument('--cems_code', default="",
                        help="EMS Code to filter")
    parser.add_argument('--aoi_code', default="",
                        help="CEMS AoI to download images from. If empty string (default) run in all the images"
                             "from all the AoIs")
    args = parser.parse_args()
    fs = fsspec.filesystem("gs")
    main(args.cems_code, args.aoi_code)
