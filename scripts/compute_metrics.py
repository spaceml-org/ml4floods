from ml4floods.models.config_setup import get_default_config
from ml4floods.models import dataset_setup
from ml4floods.models.model_setup import get_model
from ml4floods.models.model_setup import get_model_inference_function
import torch
import numpy as np
from ml4floods.models.utils import metrics
from ml4floods.data.utils import  write_json_to_gcp, GCPPath
from typing import Optional
import os
import json
import fsspec

def get_code(x):
    """" Get CEMS code """

    bn = os.path.basename(x)
    if bn.startswith("EMSR"):
        cems_code = bn.split("_")[0]
    else:
        cems_code = os.path.splitext(bn)[0]
    return cems_code

def load_inf_function_as_v1(config, model):
    if config["model_params"].get("model_version", "v1") == "v2":

        inf_func = get_model_inference_function(model, config, apply_normalization=False,
                                                activation="sigmoid")
        print("v2 inference function converted to v1")

        def inference_function(x):
            dual_head_output = inf_func(x)
            B, C, H, W = dual_head_output.shape
            out = torch.zeros((B, 3, H, W), dtype=dual_head_output.dtype)
            out[:, 2] = dual_head_output[:, 0]
            out[:, 1] = (1 - dual_head_output[:, 0]) * dual_head_output[:, 1]
            out[:, 0] = (1 - dual_head_output[:, 0]) * (1 - dual_head_output[:, 1])
            return out
    else:
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                          activation="softmax")

    return inference_function


def compute_metrics_v1(experiment_name, experiment_path="gs://ml4cc_data_lake/0_DEV/2_Mart/2_MLModelMart",
                       path_to_splits=None, overwrite=False, device:Optional[torch.device]=None):
    """
    Compute metrics of a given experiment and saves it on the corresponding folder:
    gs://ml4cc_data_lake/{prod_dev}/2_Mart/2_MLModelMart/{experiment_name}

    Args:
        experiment_name: e.g. WFV1_unet
        experiment_path: path to experiments
        path_to_splits: e.g. /worldfloods/worldfloods_v1_0/
        overwrite: overwrite the files if they exist
        device: device to run the metrics

    """
    fs = fsspec.filesystem(experiment_path)

    if not overwrite and all(fs.exists(os.path.join(experiment_path, f"{experiment_name}/{dl_name}.json").replace("\\","/")) for dl_name in ["test","val"]):
        print(f"All files exists for experiment {experiment_path}/{experiment_name}")
        return

    config_fp = os.path.join(experiment_path, f"{experiment_name}/config.json").replace("\\", "/")
    config = get_default_config(config_fp)

    config["model_params"]["max_tile_size"] = 256

    config.data_params.loader_type = 'local'
    if path_to_splits is not None:
        config.data_params.path_to_splits = path_to_splits  # local folder to download the data
    config.data_params.bucket_id = "ml4cc_data_lake"
    config.data_params.train_test_split_file = f"2_PROD/2_Mart/worldfloods_v1_0/train_test_split.json"

    config.data_params["download"] = {"train": False, "val": False, "test": True}  # download only test data

    if "filter_windows" in config["data_params"]:
        del config["data_params"]["filter_windows"]

    config["model_params"]['model_folder'] = experiment_path
    config["model_params"]['test'] = True
    model = get_model(config.model_params, experiment_name)

    model.eval()
    if device is not None:
        model.to(device)
    inference_function = load_inf_function_as_v1(config, model)

    data_module = dataset_setup.get_dataset(config["data_params"])
    thresholds_water = [0, 1e-3, 1e-2] + np.arange(0.5, .96, .05).tolist() + [.99, .995, .999]

    for dl, dl_name in [(data_module.test_dataloader(), "test"), (data_module.val_dataloader(), "val")]:
        metrics_file = os.path.join(experiment_path, f"{experiment_name}/{dl_name}.json").replace("\\", "/")
        if not overwrite and GCPPath(metrics_file).check_if_file_exists():
            print(f"File {metrics_file} exists. Continue")
            continue

        mets = metrics.compute_metrics(
            dl,
            inference_function,
            thresholds_water=thresholds_water,
            plot=False)

        if hasattr(dl.dataset, "image_files"):
            mets["cems_code"] = [get_code(f) for f in dl.dataset.image_files]
        else:
            mets["cems_code"] = [get_code(f.file_name) for f in dl.dataset.list_of_windows]

        write_json_to_gcp(metrics_file, mets)
        print(f"Per Class IOU", json.dumps(mets['iou'], indent=4, sort_keys=True))


if __name__ == '__main__':
    from tqdm import tqdm
    import traceback
    import sys

    device = torch.device("cuda:1")

    experiments_dev = ["WFV1_unet", "WF1_unet_all", "WF1_unet_bgr", "WF1_unet_rgbiswir", "WF1_unet_rgbi",
                       "WF1_hrnet_rgbbs32", "WF1_hrnet_allbs32", "WF2_unet",
                       "WF2_hrnetsmall_rgb", "WFV1_scnn20", "WF1_scnn_bgr",
                       "WF1_simplecnn_rgbi", "WF1_simplecnn_rgbiswir", "WF1_unetFilWME",
                       "WF1_scnnFiltW"]

    path_2_splits = "/worldfloods/worldfloods_v1_0/"

    experiment_path = "gs://ml4cc_data_lake/0_DEV/2_Mart/2_MLModelMart"

    for e in tqdm(experiments_dev):
        try:
            compute_metrics_v1(e, experiment_path=experiment_path, path_to_splits=path_2_splits, device=device)
        except Exception:
            print(f"Error in experiment {e}")
            traceback.print_exc(file=sys.stdout)


