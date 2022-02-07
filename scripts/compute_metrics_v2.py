# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:09:23 2022

@author: 1
"""

from ml4floods.models.config_setup import get_default_config
from ml4floods.models import dataset_setup
from ml4floods.models.model_setup import get_model, get_model_inference_function, get_channel_configuration_bands
import numpy as np
from ml4floods.data.utils import write_json_to_gcp as save_json
import os
from ml4floods.models.utils.metrics import compute_metrics_v2

def get_code(x):
    """" Get CEMS code """

    bn = os.path.basename(x)
    if bn.startswith("EMSR"):
        cems_code = bn.split("_")[0]
    else:
        cems_code = os.path.splitext(bn)[0]
    return cems_code


def main(experiment_name, path_to_splits = None, overwrite = False, mask_clouds = True):
    """
    Compute metrics of a given experiment and saves it on the corresponding folder:
    gs://ml4cc_data_lake/{prod_dev}/2_Mart/2_MLModelMart/{experiment_name}

    Args:
        experiment_name: e.g. WF2_unet
        path_to_splits: e.g. /worldfloods/worldfloods_v1_0/
        overwrite: overwrite the files if they exist

    """
    
    ## LOAD CONFIG 
    path_model = '/media/disk/databases/WORLDFLOODS/2_MLModelMart'
    
    config_fp = os.path.join(path_model,f"{experiment_name}/config.json")
    config = get_default_config(config_fp)

    ## DATA PARAMS FOR LOCAL COMPUTATION
    config["model_params"]["max_tile_size"] = 1024
    config.data_params.loader_type = 'local'
    if path_to_splits is not None:
        config.data_params.path_to_splits = path_to_splits  # local folder to download the data
    config.data_params.bucket_id = ""
    config.data_params.train_test_split_file = r''
    config.data_params["download"] = {"train": False, "val": False, "test": False}  # download only test data
    if "filter_windows" in config["data_params"]:
        del config["data_params"]["filter_windows"]
    config["model_params"]['model_folder'] = path_model
    config["model_params"]['test'] = True
    config["model_params"]['train'] = False
    
    ## LOAD MODEL AND INFERENCE FUNCTION 
    model = get_model(config.model_params, experiment_name = config.experiment_name)
    model.eval()
    #model.to("cuda") # comment this line if your machine does not have GPU
    
    inference_function = get_model_inference_function(model, config,apply_normalization=False,activation="sigmoid")

    ### METRICS COMPUTATION #### 
    data_module = dataset_setup.get_dataset(config["data_params"])
    thresholds_water = [0, 1e-3, 1e-2] + np.arange(0.5, .96, .05).tolist() + [.99, .995, .999]

    for dl, dl_name in [(data_module.test_dataloader(), "test"), (data_module.val_dataloader(), "val")]:
        metrics_file = os.path.join(path_model,f"{experiment_name}",f"{dl_name}.json")
        if not overwrite and os.path.exists(metrics_file):
            print(f"File {metrics_file} exists. Continue")
            continue

        mets = compute_metrics_v2(
            dl,
            inference_function, threshold=0.5,
            thresholds_water=thresholds_water,
            plot=False, mask_clouds=mask_clouds)

        if hasattr(dl.dataset, "image_files"):
            mets["cems_code"] = [get_code(f) for f in dl.dataset.image_files]
        else:
            mets["cems_code"] = [get_code(f.file_name) for f in dl.dataset.list_of_windows]

        save_json(metrics_file,mets)

if __name__ == '__main__':

    experiments_dev = ['WF2_unet']
    overwrite = False
    path_2_splits = '/media/disk/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0_BRIGHTNESS'

    for e in experiments_dev:

        main(e, path_to_splits = path_2_splits, overwrite = overwrite)
