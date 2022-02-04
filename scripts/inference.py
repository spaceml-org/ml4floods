from ml4floods.models.config_setup import get_default_config, get_filesystem
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.data.worldfloods import dataset
from ml4floods.models import postprocess
import torch
import rasterio
from ml4floods.data import save_cog, utils
import numpy as np
import os
from datetime import datetime
import argparse
import geopandas as gpd
import pandas as pd
import warnings
import sys
import traceback
from ml4floods.models.postprocess import get_pred_mask_v2
from typing import Tuple, Callable, List

def load_inference_function(model_path:str, device_name:str) -> Tuple[Callable[[torch.Tensor], torch.Tensor], List[int]]:

    if model_path.endswith("/"):
        experiment_name = os.path.basename(model_path[:-1])
        model_folder = os.path.dirname(model_path[:-1])
    else:
        experiment_name = os.path.basename(model_path)
        model_folder = os.path.dirname(model_path[:-1])

    config_fp = os.path.join(model_path, "config.json").replace("\\", "/")
    config = get_default_config(config_fp)

    # The max_tile_size param controls the max size of patches that are fed to the NN. If you're in a memory constrained environment set this value to 128
    config["model_params"]["max_tile_size"] = 1024

    config["model_params"]['model_folder'] = model_folder
    config["model_params"]['test'] = True
    model = get_model(config.model_params, experiment_name)
    model.to(device_name)
    channels = get_channel_configuration_bands(config.data_params.channel_configuration)
    inference_function = get_model_inference_function(model, config, apply_normalization=True,
                                                      activation=None)

    if config.model_params.get("model_version", "v1") == "v2":
        # Add post-processing of binary mask
        def predict(s2tensor:torch.Tensor) -> torch.Tensor:
            """
            Args:
                s2tensor: (C, H, W) tensor

            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land, 2: water, 3: cloud}
            """
            with torch.no_grad():
                pred = inference_function(s2tensor.unsqueeze(0))[0] # (2, H, W)
            return get_pred_mask_v2(s2tensor, pred, channels_input=channels)

    else:
        def predict(s2tensor: torch.Tensor) -> torch.Tensor:
            """
            Args:
                s2tensor: (C, H, W) tensor

            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land, 2: water, 3: cloud}
            """
            with torch.no_grad():
                pred = inference_function(s2tensor.unsqueeze(0))[0] # (3, H, W)
                mask_invalids = torch.all(s2tensor == 0, dim=0) # (H, W)
                prediction = torch.argmax(pred, dim=0).long() + 1 # (H, W)
                prediction[mask_invalids] = 0

            return prediction

    return predict, channels


@torch.no_grad()
def main(model_path:str, s2folder_file:str, device_name:str, output_folder:str):

    # This takes into account that this could be run on windows
    output_folder = output_folder.replace("\\", "/")
    s2folder_file = s2folder_file.replace("\\", "/")
    model_path = model_path.replace("\\", "/")

    if model_path.endswith("/"):
        experiment_name = os.path.basename(model_path[:-1])
    else:
        experiment_name = os.path.basename(model_path)

    inference_function, channels = load_inference_function(model_path, device_name)

    # Get S2 files to run predictions
    fs = get_filesystem(s2folder_file)
    if s2folder_file.endswith(".tif"):
        s2files = [s2folder_file]
    else:
        if not s2folder_file.endswith("/"):
            s2folder_file+="/"
        s2files = fs.glob(f"{s2folder_file}/*.tif")
        if s2folder_file.startswith("gs://"):
            s2files = [f"gs://{s2}" for s2 in s2files]

        assert len(s2files) > 0, f"No Tiff files found in {s2folder_file}/*.tif"

    if output_folder.endswith("/"):
        output_folder_vec = output_folder[:-1] + "_vec"
    else:
        output_folder_vec = output_folder + "_vec"

    files_with_errors = []

    fs_dest = get_filesystem(output_folder)

    for total, filename in enumerate(s2files):
        filename_save = os.path.join(output_folder, os.path.basename(filename))
        filename_save_vect = os.path.join(output_folder_vec, f"{os.path.splitext(os.path.basename(filename))[0]}.geojson")

        exists_tiff = fs_dest.exists(filename_save)
        if exists_tiff and fs_dest.exists(filename_save_vect):
            continue

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({total}/{len(s2files)}) Processing {filename}")
        try:
            if exists_tiff:
                with rasterio.open(filename_save) as rst:
                    prediction = rst.read(1)
                    crs  = rst.crs
                    transform = rst.transform
            else:
                torch_inputs, transform = dataset.load_input(filename,
                                                             window=None, channels=channels)

                with rasterio.open(filename) as src:
                    crs = src.crs

                prediction = inference_function(torch_inputs).cpu().numpy()

            # Save data as vectors
            data_out = vectorize_outputv1(prediction, crs, transform)
            if data_out is not None:
                utils.write_geojson_to_gcp(filename_save_vect, data_out)

            # Save data as COG GeoTIFF
            profile = {"crs": crs,
                       "transform": transform,
                       "compression": "lzw",
                       "RESAMPLING": "NEAREST",
                       "nodata": 0}

            save_cog.save_cog(prediction[np.newaxis], filename_save, profile=profile,
                              tags={"model": experiment_name})
        except Exception:
            warnings.warn(f"Failed")
            traceback.print_exc(file=sys.stdout)
            files_with_errors.append(filename)

    if len(files_with_errors) > 0:
        print(f"Files with errors:\n {files_with_errors}")


def vectorize_outputv1(prediction, crs, transform):
    data_out = []
    start = 0
    class_name = {2: "water", 3: "cloud"}
    for c in [2, 3]:
        geoms_polygons = postprocess.get_water_polygons(prediction == c, transform=transform)
        if len(geoms_polygons) > 0:
            data_out.append(gpd.GeoDataFrame({"geometry": geoms_polygons,
                                              "id": np.arange(start, start + len(geoms_polygons)),
                                              "class": class_name[c]},
                                             crs=crs))
        start += len(geoms_polygons)

    if len(data_out) == 1:
        return data_out[0]
    elif len(data_out) > 1:
        return pd.concat(data_out, ignore_index=True)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run inference on S2 images')
    parser.add_argument("--s2", required=True, help="Path to folder with tif files or tif file to make prediction")
    parser.add_argument("--model_path",
                        help="Path to experiment folder. Inside this folder there should be a config.json file and  a model weights file model.pt",
                        required=True)
    parser.add_argument("--output_folder",
                        help="Path to save the files. The name of the prediction will be the same as the S2 image."
                             "If not provided it will be saved in dirname(s2)/basename(model_path)/",
                        required=False)
    parser.add_argument('--device_name', default="cuda", help="Device name")

    args = parser.parse_args()

    # Compute folder name to save the predictions if not provided
    if not args.output_folder:
        if args.model_path.endswith("/"):
            en = os.path.basename(args.model_path[:-1])
        else:
            en = os.path.basename(args.model_path)
        if args.s2.endswith(".tif"):
            base_output_folder = os.path.dirname(os.path.dirname(args.s2))
        elif args.s2.endswith("/"):
            base_output_folder = os.path.dirname(args.s2[:-1])
        else:
            base_output_folder = os.path.dirname(args.s2)

        output_folder = os.path.join(base_output_folder, en)
        print(f"Predictions will be saved in folder: {output_folder}")
    else:
        output_folder = args.output_folder

    main(model_path=args.model_path, s2folder_file=args.s2,device_name=args.device_name,
         output_folder=output_folder)




