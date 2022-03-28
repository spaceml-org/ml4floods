# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:09:23 2022

@author: 1

Compute metrics for ml4floods package:
compute_metrics_v2 uses water probability band to compute the metrics
"""
import numpy as np
import os

from ml4floods.models.config_setup import get_default_config
from ml4floods.models import dataset_setup
from ml4floods.models.model_setup import get_model, get_model_inference_function, get_channel_configuration_bands
from ml4floods.data.utils import write_json_to_gcp as save_json
from ml4floods.models.utils.metrics import compute_metrics_v2
from typing import Tuple, Callable, List, Optional
import torch 

def get_code(x):
    """" Get CEMS code """

    bn = os.path.basename(x)
    if bn.startswith("EMSR"):
        cems_code = bn.split("_")[0]
    else:
        cems_code = os.path.splitext(bn)[0]
    return cems_code

def load_inference_function(model_path:str, device_name:str,max_tile_size:int=1024) -> Tuple[Callable[[torch.Tensor], torch.Tensor], List[int]]:

    if model_path.endswith("/"):
        experiment_name = os.path.basename(model_path[:-1])
        model_folder = os.path.dirname(model_path[:-1])
    else:
        experiment_name = os.path.basename(model_path)
        model_folder = os.path.dirname(model_path[:-1])

    config_fp = os.path.join(model_path, "config.json").replace("\\", "/")
    config = get_default_config(config_fp)

    # The max_tile_size param controls the max size of patches that are fed to the NN. If you're in a memory constrained environment set this value to 128
    config["model_params"]["max_tile_size"] = max_tile_size

    config["model_params"]['model_folder'] = model_folder
    config["model_params"]['test'] = True
    model = get_model(config.model_params, experiment_name)
    model.to(device_name)
    channels = get_channel_configuration_bands(config.data_params.channel_configuration)
    if config.model_params.get("model_version", "v1") == "v2":
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                          activation='sigmoid')
    else:
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                  activation='softmax')
        


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
                #pred = inference_function(s2tensor.unsqueeze(0))[0] # (2, H, W)
                pred = inference_function(s2tensor) # (2, H, W)
                #pred = inference_function(s2tensor)[0] # (2, H, W) NO RESPETA EL BATCH
            #return get_pred_mask_v2(s2tensor, pred, channels_input=channels)
            return pred

    else:
        def predict(s2tensor: torch.Tensor) -> torch.Tensor:
            """
            Args:
                s2tensor: (C, H, W) tensor
            Returns:
                (H, W) mask with interpretation {0: invalids, 1: land, 2: water, 3: cloud}
            """
            with torch.no_grad():
                #pred = inference_function(s2tensor.unsqueeze(0))[0] # (3, H, W)
                pred = inference_function(s2tensor) # (3, H, W)
                #pred = inference_function(s2tensor)[0] # (2, H, W) NO RESPETA EL BATCH
                # mask_invalids = torch.all(s2tensor == 0, dim=1).squeeze() # (H, W)
                # prediction = torch.argmax(pred, dim=0).type(torch.uint8) + 1 # (H, W)
                #prediction = (pred[1] > .5).type(torch.uint8) + 1 # (H, W)
                #prediction[mask_invalids] = 0
                
            #return prediction
            return pred

    return predict, channels



def main(experiment_name, experiment_path="gs://ml4cc_data_lake/0_DEV/2_Mart/2_MLModelMart",
                       path_to_splits=None, overwrite=False, device:Optional[torch.device]=None):
    """
    Compute metrics of a given experiment and saves it on the corresponding folder:
    gs://ml4cc_data_lake/{prod_dev}/2_Mart/2_MLModelMart/{experiment_name}

    Args:
        experiment_name: e.g. WF2_unet
        path_to_splits: e.g. /worldfloods/worldfloods_v1_0/
        overwrite: overwrite the files if they exist
    """
    ## LOAD CONFIG 
    
    model_path = os.path.join(experiment_path,experiment_name)
    
    config_fp = os.path.join(model_path,"config.json")
    config = get_default_config(config_fp)

    inference_function, channels = load_inference_function(model_path, device)
    

    if path_to_splits is not None:
        config.data_params.path_to_splits = path_to_splits  # local folder to download the data
    config.data_params.bucket_id = "ml4cc_data_lake"
    config.data_params.train_test_split_file = f"2_PROD/2_Mart/worldfloods_v1_0/train_test_split.json"
    config.data_params.train_test_split_file = ""
    if "filter_windows" in config["data_params"]:
        del config["data_params"]["filter_windows"]
    
    ### METRICS COMPUTATION #### 
    data_module = dataset_setup.get_dataset(config["data_params"])
    thresholds_water = [0, 1e-3, 1e-2] + np.arange(0.5, .96, .05).tolist() + [.99, .995, .999]

    for dl, dl_name in [(data_module.test_dataloader(), "test"), (data_module.val_dataloader(), "val")]:
    # for dl, dl_name in [ (data_module.val_dataloader(), "val")]:        
        metrics_file = os.path.join(model_path,f"{dl_name}.json")
        if not overwrite and os.path.exists(metrics_file):
            print(f"File {metrics_file} exists. Continue")
            continue

        mets = compute_metrics_v2(
            dl,
            inference_function, threshold=0.5,
            thresholds_water=thresholds_water,
            plot=False, mask_clouds=True)

        if hasattr(dl.dataset, "image_files"):
            mets["cems_code"] = [get_code(f) for f in dl.dataset.image_files]
        else:
            mets["cems_code"] = [get_code(f.file_name) for f in dl.dataset.list_of_windows]

        save_json(metrics_file,mets)
        
        
        
if __name__ == '__main__':
    from tqdm import tqdm
    import traceback
    import sys

    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

    experiments_dev = ["WFV1_unet", "WF1_unet_all", "WF1_unet_bgr", "WF1_unet_rgbiswir", "WF1_unet_rgbi",
                       "WF1_hrnet_rgbbs32", "WF1_hrnet_allbs32", "WF2_unet",
                       "WF2_hrnetsmall_rgb", "WFV1_scnn20", "WF1_scnn_bgr",
                       "WF1_simplecnn_rgbi", "WF1_simplecnn_rgbiswir", "WF1_unetFilWME",
                       "WF1_scnnFiltW"]
    
    path_2_splits = "/worldfloods/worldfloods_v1_0/"
    experiment_path = "gs://ml4cc_data_lake/0_DEV/2_Mart/2_MLModelMart"
    
    experiments_dev = ['WF1_unet_full_norm','WF2_unet_full_norm']
    path_2_splits = r'X:\media\disk\databases\WORLDFLOODS\2_Mart\worldfloods_extra_v2_0_BRIGHTNESS'
    experiment_path = r'X:\home\kike\Projectes\ml4floods\2_MLModelMart'

    for e in tqdm(experiments_dev):
        try:
            main(e, experiment_path=experiment_path, path_to_splits=path_2_splits, device=device)
        except Exception:
            print(f"Error in experiment {e}")
            traceback.print_exc(file=sys.stdout)
