import argparse
import fsspec
from ml4floods.data import cmkappazeta, save_cog
from ml4floods.models import postprocess
import geopandas as gpd
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List
import rasterio
import rasterio.windows
import numpy as np
import sys
import warnings
import traceback


def load_input(tiff_input:str, channels:List[int],
               window:Optional[rasterio.windows.Window]=None) -> Tuple[np.ndarray, rasterio.transform.Affine, str]:
    """

    Args:
        tiff_input:
        window: rasterio.Window object to read (None to read all)
        channels: 0-based channels to read

    Returns:
        3-D tensor (len(channels), H, W), rasterio.transform and crs

    """
    with rasterio.open(tiff_input, "r") as rst:
        inputs = rst.read((np.array(channels) + 1).tolist(), window=window)
        # Shifted transform based on the given window (used for plotting)
        transform = rst.transform if window is None else rasterio.windows.transform(window, rst.transform)
        inputs = np.nan_to_num(inputs).astype(np.float32)
        crs = rst.crs

    return inputs, transform, crs


def vectorize_output(prediction, crs, transform):
    data_out = []
    start = 0
    for c in [2, 3, 4]:
        if c == 3:
            binary_mask = (prediction == 3) | (prediction == 4)
            class_name = "THICK AND THIN CLOUDS"
        else:
            binary_mask = prediction == c
            class_name = cmkappazeta.CLASSES_KAPPAZETA[c]
        geoms_polygons = postprocess.get_water_polygons(prediction == binary_mask,
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


def main(cems_code:str, aoi_code:str):
    tiff_files = fs.glob(f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/*{cems_code}/*{aoi_code}/S2/*.tif")

    model = cmkappazeta.Unet()
    model.construct()
    model.load_weights(filelocal="./weights_l1c_kappazeta.hdf5")
    files_with_errors = []
    for total, filename in enumerate(tiff_files):
        filename = f"gs://{filename}"
        filename_save = filename.replace("/S2/", "/cmkappazeta/")
        filename_save_vect = filename.replace("/S2/", "/cmkappazeta_vec/").replace(".tif", ".geojson")

        if fs.exists(filename_save) and fs.exists(filename_save_vect):
            continue

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({total}/{len(tiff_files)}) Processing {filename}")

        try:
            inputs, transform, crs = load_input(filename,
                                                channels=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])

            pred = model.predict(inputs)
            pred[pred == 5] = 0

            data_out = vectorize_output(pred, crs, transform)
            if data_out is not None:
                data_out.to_file(filename_save_vect, driver="GeoJSON")

            # Save data as COG GeoTIFF
            profile = {"crs": crs,
                       "transform": transform,
                       "compression": "lzw",
                       "RESAMPLING": "NEAREST",
                       "nodata": 0}

            save_cog.save_cog(pred[np.newaxis], filename_save, profile=profile)

        except Exception:
            warnings.warn(f"Failed {filename}")
            traceback.print_exc(file=sys.stdout)
            files_with_errors.append(filename)

    if len(files_with_errors) > 0:
        print(f"Files with errors:\n {files_with_errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run inference on all S2 images in Staging')
    parser.add_argument('--cems_code', default="",
                        help="EMS Codes to filter")
    parser.add_argument('--aoi_code', default="",
                        help="CEMS AoI to download images from. If empty string (default) download the images"
                             "from all the AoIs")
    args = parser.parse_args()
    fs = fsspec.filesystem("gs")
    main(args.cems_code, args.aoi_code)
