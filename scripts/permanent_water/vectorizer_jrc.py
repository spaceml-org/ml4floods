import argparse
from ml4floods.data import utils
from ml4floods.models import postprocess
from datetime import datetime
import sys
import warnings
import traceback
import os


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
        filename_save_vect = os.path.join(dir_save, name_folder, os.path.splitext(os.path.basename(filename))[0]+".geojson")

        if (not overwrite) and fs.exists(filename_save_vect):
            continue

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({total}/{len(permanent_water_files)}) Processing {filename}")

        try:
            with utils.rasterio_open_read(filename) as rst:
                permanent_water_data = rst.read(1)
                crs  = rst.crs
                transform = rst.transform

            data_out = postprocess.vectorize_jrc_permanent_water_layer(permanent_water_data, crs, transform)

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




