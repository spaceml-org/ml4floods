from ml4floods.models.config_setup import get_default_config, get_filesystem
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.models.utils.configuration import AttrDict
from ml4floods.data.worldfloods import dataset
from ml4floods.data import create_gt
from ml4floods.models import postprocess
import torch
import rasterio
from ml4floods.data import save_cog, utils
import numpy as np
import os
from datetime import datetime
import geopandas as gpd
import pandas as pd
import warnings
import sys
import traceback
from ml4floods.models.postprocess import get_pred_mask_v2
from typing import Tuple, Callable, Any, Optional
from ml4floods.data.worldfloods.configs import BANDS_S2, BANDS_L8
from skimage.morphology import binary_dilation, disk


def load_inference_function(model_path: str, device_name: str, max_tile_size: int = 1024,
                            th_water: float = .5,
                            th_brightness: float = create_gt.BRIGHTNESS_THRESHOLD,
                            collection_name:str="S2",
                            disable_pbar:bool=True,
                            distinguish_flood_traces:bool=False) -> Tuple[
    Callable[[torch.Tensor], Tuple[torch.Tensor,torch.Tensor]], AttrDict]:
    if model_path.endswith("/"):
        experiment_name = os.path.basename(model_path[:-1])
        model_folder = os.path.dirname(model_path[:-1])
    else:
        experiment_name = os.path.basename(model_path)
        model_folder = os.path.dirname(model_path)

    config_fp = os.path.join(model_path, "config.json").replace("\\", "/")
    config = get_default_config(config_fp)

    # The max_tile_size param controls the max size of patches that are fed to the NN. If you're in a memory constrained environment set this value to 128
    config["model_params"]["max_tile_size"] = max_tile_size

    config["model_params"]['model_folder'] = model_folder
    config["model_params"]['test'] = True
    model = get_model(config.model_params, experiment_name)
    model.to(device_name)
    inference_function = get_model_inference_function(model, config, apply_normalization=True,
                                                      activation=None,disable_pbar=disable_pbar)

    if config.model_params.get("model_version", "v1") == "v2":

        channels = get_channel_configuration_bands(config.data_params.channel_configuration,
                                                   collection_name=collection_name)
        if distinguish_flood_traces:
            if collection_name == "S2":
                band_names_current_image = [BANDS_S2[iband] for iband in channels]
                mndwi_indexes_current_image = [band_names_current_image.index(b) for b in ["B3", "B11"]]
            elif collection_name == "Landsat":
                band_names_current_image = [BANDS_L8[iband] for iband in channels]
                # TODO ->  if not all(b in band_names_current_image for b in ["B3","B6"])
                mndwi_indexes_current_image = [band_names_current_image.index(b) for b in ["B3", "B6"]]

        # Add post-processing of binary mask
        def predict(s2l89tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                s2l89tensor: (C, H, W) tensor

            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land, 2: water, 3: think cloud, 4: flood trace}
            """
            with torch.no_grad():
                pred = inference_function(s2l89tensor.unsqueeze(0))[0]  # (2, H, W)
                land_water_cloud =  get_pred_mask_v2(s2l89tensor, pred, channels_input=channels,
                                                     th_water=th_water, th_brightness=th_brightness,
                                                     collection_name=collection_name)

                # Set invalids in continuous pred to -1
                invalids = land_water_cloud == 0
                pred[0][invalids] = -1
                pred[1][invalids] = -1

                if distinguish_flood_traces:
                    s2l89mndwibands = s2l89tensor[mndwi_indexes_current_image, ...].float()

                    # Green − SWIR1)/(Green + SWIR1)
                    mndwi = (s2l89mndwibands[0] - s2l89mndwibands[1]) / (s2l89mndwibands[0] + s2l89mndwibands[1] + 1e-6)

                    land_water_cloud[(land_water_cloud == 2) & (mndwi < 0)] = 4

            return land_water_cloud, pred

    else:
        def predict(s2l89tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                s2l89tensor: (C, H, W) tensor

            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land, 2: water, 3: cloud, 4: flood trace}
            """
            with torch.no_grad():
                pred = inference_function(s2l89tensor.unsqueeze(0))[0]  # (3, H, W)
                invalids = torch.all(s2l89tensor == 0, dim=0)  # (H, W)
                land_water_cloud = torch.argmax(pred, dim=0).type(torch.uint8) + 1  # (H, W)
                land_water_cloud[invalids] = 0

                # Set invalids in continuous pred to -1
                pred[0][invalids] = -1
                pred[1][invalids] = -1
                pred[2][invalids] = -1

                if distinguish_flood_traces:
                    s2l89mndwibands = s2l89tensor[mndwi_indexes_current_image, ...].float()
                    # Green − SWIR1)/(Green + SWIR1)
                    mndwi = (s2l89mndwibands[0] - s2l89mndwibands[1]) / (s2l89mndwibands[0] + s2l89mndwibands[1] + 1e-6)
                    land_water_cloud[(land_water_cloud == 2) & (mndwi < 0)] = 4

            return land_water_cloud, pred

    return predict, config


@torch.no_grad()
def main(model_path: str, s2folder_file: str, device_name: str,
         output_folder: Optional[str]=None, max_tile_size: int = 1_024,
         th_brightness: float = create_gt.BRIGHTNESS_THRESHOLD, th_water: float = .5, overwrite: bool = False,
         collection_name: str = "S2",distinguish_flood_traces:bool=False):

    # This takes into account that this could be run on windows
    s2folder_file = s2folder_file.replace("\\", "/")
    model_path = model_path.replace("\\", "/")

    if model_path.endswith("/"):
        experiment_name = os.path.basename(model_path[:-1])
    else:
        experiment_name = os.path.basename(model_path)

    inference_function, config = load_inference_function(model_path, device_name, max_tile_size=max_tile_size,
                                                         th_water=th_water, th_brightness=th_brightness,
                                                         collection_name=collection_name,
                                                         distinguish_flood_traces=distinguish_flood_traces)

    # Get S2 files to run predictions
    fs = get_filesystem(s2folder_file)
    if s2folder_file.endswith(".tif"):
        s2files = [s2folder_file]
    else:
        if not s2folder_file.endswith("/"):
            s2folder_file += "/"
        s2files = fs.glob(f"{s2folder_file}*.tif")
        if s2folder_file.startswith("gs://"):
            s2files = [f"gs://{s2}" for s2 in s2files]

        assert len(s2files) > 0, f"No Tiff files found in {s2folder_file}*.tif"

    files_with_errors = []

    if output_folder:
        fs_dest = get_filesystem(output_folder)
        if output_folder.endswith("/"):
            output_folder_iter_vec = output_folder[:-1] + "_vec"
            output_folder_iter_cont = output_folder[:-1] + "_cont"
        else:
            output_folder_iter_vec = output_folder + "_vec"
            output_folder_iter_cont = output_folder + "_cont"
    else:
        fs_dest = fs

    for total, filename in enumerate(s2files):
        # Compute folder name to save the predictions if not provided
        if not output_folder:
            base_output_folder = os.path.dirname(os.path.dirname(filename))
            output_folder_iter = os.path.join(base_output_folder, experiment_name, collection_name).replace("\\", "/")
            output_folder_iter_vec = os.path.join(base_output_folder, experiment_name + "_vec", collection_name).replace("\\", "/")
            output_folder_iter_cont = os.path.join(base_output_folder, experiment_name + "_cont",
                                                  collection_name).replace("\\", "/")
        else:
            output_folder_iter = output_folder


        filename_save = os.path.join(output_folder_iter, os.path.basename(filename))
        filename_save_cont = os.path.join(output_folder_iter_cont, os.path.basename(filename))
        filename_save_vect = os.path.join(output_folder_iter_vec,
                                          f"{os.path.splitext(os.path.basename(filename))[0]}.geojson")

        if not overwrite and all(fs_dest.exists(f) for f in [filename_save, filename_save_cont, filename_save_vect]):
            continue

        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({total + 1}/{len(s2files)}) Processing {collection_name} {filename}")
        if not output_folder:
            print(f"Predictions will be saved in folder: {output_folder_iter}")
        try:
            channels = get_channel_configuration_bands(config.data_params.channel_configuration,
                                                       collection_name=collection_name,as_string=True)
            torch_inputs, transform = dataset.load_input(filename,
                                                         window=None, channels=channels)

            with rasterio.open(filename) as src:
                crs = src.crs

            prediction, pred_cont = inference_function(torch_inputs)
            prediction = prediction.cpu().numpy()
            pred_cont = pred_cont.cpu().numpy()

            # Save data as vectors
            data_out = vectorize_outputv1(prediction, crs, transform)
            if data_out is not None:
                if not filename_save_vect.startswith("gs"):
                    fs_dest.makedirs(os.path.dirname(filename_save_vect), exist_ok=True)
                utils.write_geojson_to_gcp(filename_save_vect, data_out)

            # Save data as COG GeoTIFF
            profile = {"crs": crs,
                       "transform": transform,
                       "compression": "lzw",
                       "RESAMPLING": "NEAREST",
                       "nodata": 0}

            if not filename_save.startswith("gs"):
                fs_dest.makedirs(os.path.dirname(filename_save), exist_ok=True)

            save_cog.save_cog(prediction[np.newaxis], filename_save, profile=profile.copy(),
                              descriptions=["invalid/land/water/cloud/trace"],
                              tags={"invalid":0, "land":1, "water":2, "cloud":3 , "trace":4, "model": experiment_name})

            if not filename_save_cont.startswith("gs"):
                fs_dest.makedirs(os.path.dirname(filename_save_cont), exist_ok=True)

            if pred_cont.shape[0] == 2:
                descriptions = ["clear/cloud", "land/water"]
            else:
                descriptions = ["prob_clear","prob_water", "prob_cloud"]

            profile["nodata"] = -1
            save_cog.save_cog(pred_cont, filename_save_cont, profile=profile.copy(),
                              descriptions=descriptions,
                              tags={"model": experiment_name})

        except Exception:
            warnings.warn(f"Failed")
            traceback.print_exc(file=sys.stdout)
            files_with_errors.append(filename)

    if len(files_with_errors) > 0:
        print(f"Files with errors:\n {files_with_errors}")


def vectorize_outputv1(prediction: np.ndarray, crs: Any, transform: rasterio.Affine,
                       border:int=2) -> Optional[gpd.GeoDataFrame]:
    """

    Args:
        prediction: (H, W) array with 4 posible values  0: "invalid", 2: "water", 3: "cloud", 4: "flood_trace"
        crs:
        transform:
        border:

    Returns:

    """
    data_out = []
    start = 0
    class_name = {0: "area_imaged", 2: "water", 3: "cloud", 4: "flood_trace"}
    # Dilate invalid mask
    invalid_mask = binary_dilation(prediction == 0, disk(3)).astype(bool)

    # Set borders to zero to avoid border effects when vectorizing
    prediction[:border,:] = 0
    prediction[:, :border] = 0
    prediction[-border:, :] = 0
    prediction[:, -border:] = 0

    prediction[invalid_mask] = 0
    for c, cn in class_name.items():
        if c == 0:
            # To remove stripes in area imaged
            mask = prediction != c
        else:
            mask = prediction == c

        geoms_polygons = postprocess.get_water_polygons(mask, transform=transform)
        if len(geoms_polygons) > 0:
            data_out.append(gpd.GeoDataFrame({"geometry": geoms_polygons,
                                              "id": np.arange(start, start + len(geoms_polygons)),
                                              "class": cn},
                                             crs=crs))
        start += len(geoms_polygons)

    if len(data_out) == 1:
        return data_out[0]
    elif len(data_out) > 1:
        return pd.concat(data_out, ignore_index=True)

    return None