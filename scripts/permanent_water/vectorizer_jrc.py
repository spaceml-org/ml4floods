import torch
import argparse
from ml4floods.data import utils
from ml4floods.models import postprocess
from datetime import datetime
from typing import Optional
import rasterio.windows
import numpy as np
import sys
import warnings
import traceback
import geopandas as gpd
import os
import pandas as pd


def vectorize_output(permanent_water:np.ndarray, crs:str, transform:rasterio.transform.Affine) -> Optional[gpd.GeoDataFrame]:
    """
    Vectorize cloud class

    Args:
        permanent_water: (H, W) array with predictions.
        https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_3_YearlyHistory?hl=en
        Values 0 nodata 1 not water, 2 seasonal water, 3: permanent water
        crs:
        transform:

    Returns:
        gpd.GeoDataFrame with vectorized cloud, shadows and thick and thin clouds classes
    """

    data_out = []
    start = 0
    for c, class_name in zip([2, 3],["seasonal", "permanent"]):
        geoms_polygons = postprocess.get_water_polygons(permanent_water == c,
                                                        transform=transform)
        if len(geoms_polygons) > 0:
            data_out.append(gpd.GeoDataFrame({"geometry": geoms_polygons,
                                              "id": np.arange(start, start + len(geoms_polygons)),
                                              "class": class_name},
                                             crs=crs))
            start += len(geoms_polygons)

    if len(data_out) == 1:
        return data_out[0]
    elif len(data_out) > 1:
        return pd.concat(data_out, ignore_index=True)
    return None


def main(folder_image:str, overwrite:bool=False):
    folder_image = folder_image.replace("\\", "/")

    fs = utils.get_filesystem(folder_image)
    if folder_image.endswith(".tif"):
        permanent_water_files = [folder_image]
    else:
        if not folder_image.endswith("/"):
            folder_image += "/"
        permanent_water_files = fs.glob(f"{folder_image}*.tif")
        if folder_image.startswith("gs://"):
            permanent_water_files = [f"gs://{s2}" for s2 in permanent_water_files]

        assert len(permanent_water_files) > 0, f"No Tiff files found in {folder_image}*.tif"


    files_with_errors = []
    for total, filename in enumerate(permanent_water_files):
        dir_save = os.path.dirname(os.path.dirname(filename))
        name_folder = os.path.basename(os.path.dirname(filename))+"_vec"
        filename_save_vect = os.path.join(dir_save, name_folder, os.path.splitext(os.path.basename(filename))[0]+".gejson")

        if (not overwrite) and fs.exists(filename_save_vect):
            continue

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({total}/{len(permanent_water_files)}) Processing {filename}")

        try:
            with utils.rasterio_open_read(filename) as rst:
                permanent_water_data = rst.read(1)
                crs  = rst.crs
                transform = rst.transform

            data_out = vectorize_output(permanent_water_data, crs, transform)

            if data_out is not None:
                utils.write_geojson_to_gcp(filename_save_vect, data_out)

        except Exception:
            warnings.warn(f"Failed {filename}")
            traceback.print_exc(file=sys.stdout)
            files_with_errors.append(filename)

    if len(files_with_errors) > 0:
        print(f"Files with errors:\n {files_with_errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Vectorize Permanent Water JRC')
    parser.add_argument("--image", required=True, help="Path to folder with tif files or tif file with permanent water")
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help="Overwrite the prediction if exists")

    args = parser.parse_args()

    main(folder_image=args.image, overwrite=args.overwrite)




