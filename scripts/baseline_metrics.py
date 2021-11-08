# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:17:29 2021

@author: 1
"""

from ml4floods.models.config_setup import get_default_config
from ml4floods.models import dataset_setup
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function
import torch
import numpy as np
from ml4floods.models.utils import metrics
from ml4floods.data.utils import  write_json_to_gcp as save_json
# from ml4floods.models.config_setup import save_json
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"X:\home\kike\json_creds\ml4cc-general-access_request_pays.json"

def get_code(x):
    """" Get CEMS code """

    bn = os.path.basename(x)
    if bn.startswith("EMSR"):
        cems_code = bn.split("_")[0]
    else:
        cems_code = os.path.splitext(bn)[0]
    return cems_code


from ml4floods.models.architectures import ndwi

def inference_function(x):
    return -ndwi.extract_mndwi(x)

def baseline_metrics(experiment_name, overwrite = False, mask_clouds = True):
    ## SETUP ##
    
    path_model = r'X:\home\kike\Projectes\ml4floods\2_MLModelMart'
    config_fp = os.path.join(path_model,f"{experiment_name}\config.json").replace("\\","/")

    if not os.path.exists(config_fp):
        print(f'{config_fp} no exists')
        return
    config = get_default_config(config_fp)

    config.data_params.loader_type = 'local'
    config.data_params.bucket_id = ""
    config.data_params.train_test_split_file = r'C:\Users\1\Documents\Projectes\Floods\Exploratori\splits\split_v2_0_unused_NEW_local.json'
    config["model_params"]["max_tile_size"] = 1024
    config.data_params["download"] = {"train": False, "val": False, "test": False}  # download only test data

    if "filter_windows" in config["data_params"]:
        del config["data_params"]["filter_windows"]

    #config["model_params"]['model_folder'] = rf'X:\home\kike\Projectes\ml4floods\2_MLModelMart'
    config["model_params"]['model_folder'] = path_model
    config["model_params"]['test'] = True
    config["model_params"]['train'] = False
    
    ## DATA MODULE AND INF FUNCTION ## 
    
    data_module = dataset_setup.get_dataset(config["data_params"])
    # thresholds_water = [0, 1e-3, 1e-2] + np.arange(0.5, .96, .05).tolist() + [.99, .995, .999]
    thresholds_water = np.arange(-1, 1, .05)
    
    for dl, dl_name in [(data_module.test_dataloader(), "test"), (data_module.val_dataloader(), "val")]:
        #metrics_file = rf"X:\home\kike\Projectes\ml4floods\model_metrics_new_test\{experiment_name}\{dl_name}.json"
        metrics_file = os.path.join(path_model,'MNDWI',f"{dl_name}_masked_clouds.json")
        if not overwrite and os.path.exists(metrics_file):
            print(f"File {metrics_file} exists. Continue")
            continue

        mets = metrics.compute_metrics(
            dl,
            inference_function,
            thresholds_water=thresholds_water,
            threshold=0,
            plot=False, mask_clouds=mask_clouds)

        if hasattr(dl.dataset, "image_files"):
            mets["cems_code"] = [get_code(f) for f in dl.dataset.image_files]
        else:
            mets["cems_code"] = [get_code(f.file_name) for f in dl.dataset.list_of_windows]

        #write_json_to_gcp(metrics_file, mets)
        save_json(metrics_file,mets)

if __name__ == '__main__':

    overwrite = True
    mask_clouds = True
    experiment_name = 'WFV1_unet'
    baseline_metrics(experiment_name, overwrite, mask_clouds)
