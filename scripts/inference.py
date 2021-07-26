from ml4floods.models.config_setup import get_default_config
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function
from ml4floods.models.model_setup import get_channel_configuration_bands
from ml4floods.data.worldfloods import dataset
from ml4floods.models import postprocess
import torch
import rasterio
from ml4floods.data import save_cog
import numpy as np
import fsspec
from datetime import datetime
import argparse
import geopandas as gpd
import pandas as pd
import warnings
import sys
import traceback


def load_inference_function(experiment_name):

    # TODO handle multioutput model (instead of binary classification model)
    config_fp = f"gs://ml4cc_data_lake/2_PROD/2_Mart/2_MLModelMart/{experiment_name}/config.json"
    config = get_default_config(config_fp)

    # The max_tile_size param controls the max size of patches that are fed to the NN. If you're in a memory constrained environment set this value to 128
    config["model_params"]["max_tile_size"] = 1024

    config["model_params"]['model_folder'] = 'gs://ml4cc_data_lake/2_PROD/2_Mart/2_MLModelMart'
    config["model_params"]['test'] = True
    model = get_model(config.model_params, experiment_name)
    model.to("cuda")
    channels = get_channel_configuration_bands(config.data_params.channel_configuration)

    return get_model_inference_function(model, config,apply_normalization=True), channels



@torch.no_grad()
def get_segmentation_mask(torch_inputs, inference_function):
    outputs = inference_function(torch_inputs.unsqueeze(0))[0]
    prediction = torch.argmax(outputs, dim=0).long()
    mask_invalid = torch.all(torch_inputs == 0, dim=0)
    prediction += 1
    prediction[mask_invalid] = 0  # (H, W) {0: invalid, 1:land, 2: water, 3:cloud}

    # prob_water_mask = outputs[0, 1].cpu().numpy()
    # binary_water_mask = prob_water_mask > .5
    prediction = prediction.unsqueeze(0)
    return np.array(prediction.cpu()).astype(np.uint8)


MODEL_EXPERIMENT_DEFAULT = "WFV1_unet"

@torch.no_grad()
def main(model_experiment, cems_code):
    inference_function, channels = load_inference_function(model_experiment)

    tiff_files = fs.glob(f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/*{cems_code}/*/S2/*.tif")
    tiff_files = [f for f in tiff_files if not fs.exists(f"gs://{f}".replace("/S2/", f"/{model_experiment}/"))]

    files_with_errors = []
    for total, filename in enumerate(tiff_files):
        filename = f"gs://{filename}"
        filename_save = filename.replace("/S2/", f"/{model_experiment}/")
        filename_save_vect = filename.replace("/S2/", f"/{model_experiment}_vec/").replace(".tif", ".geojson")

        if fs.exists(filename_save) and fs.exists(filename_save_vect):
            continue

        try:
            torch_inputs, transform = dataset.load_input(filename,
                                                         window=None, channels=channels)

            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ({total}/{len(tiff_files)}) Processing {filename}")

            with rasterio.open(filename) as src:
                crs = src.crs

            prediction = get_segmentation_mask(torch_inputs, inference_function)

            # Save data as vectors
            data_out = []
            start = 0
            class_name = {2: "water", 3: "cloud"}
            for c in [2, 3]:
                geoms_polygons = postprocess.get_water_polygons(prediction == c, transform=transform)
                data_out.append(gpd.GeoDataFrame({"geometry": geoms_polygons,
                                                  "id": np.arange(start, start + len(geoms_polygons)),
                                                  "class": class_name[c]},
                                                 crs=crs))
                start += len(geoms_polygons)

            data_out = pd.concat(data_out, ignore_index=True)
            if data_out.shape[0] > 0:
                data_out.to_file(filename_save_vect, driver="GeoJSON")

            # Save data as COG GeoTIFF
            profile = {"crs": crs,
                       "transform": transform,
                       "compression": "lzw",
                       "RESAMPLING": "NEAREST",
                       "nodata": 0}

            save_cog.save_cog(prediction, filename_save, profile=profile,
                              tags={"model": model_experiment})
        except Exception:
            warnings.warn(f"Failed")
            traceback.print_exc(file=sys.stdout)
            files_with_errors.append(filename)

    if len(files_with_errors) > 0:
        print(f"Files with errors:\n {files_with_errors}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run inference on all S2 images in Stagging')
    parser.add_argument('--cems_code', default="",
                        help="EMS Codes to filter")
    parser.add_argument('--model_experiment', default=MODEL_EXPERIMENT_DEFAULT,
                        help="Experiment name to load the weights")
    args = parser.parse_args()
    fs = fsspec.filesystem("gs")
    main(args.model_experiment, args.cems_code)




