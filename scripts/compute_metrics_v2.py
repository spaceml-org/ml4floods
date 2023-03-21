import numpy as np
import os

from ml4floods.models.config_setup import get_default_config
from ml4floods.models import dataset_setup
from ml4floods.models.model_setup import get_model, get_model_inference_function, get_channel_configuration_bands
from ml4floods.data.utils import write_json_to_gcp as save_json
from ml4floods.data.utils import get_filesystem
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


def load_inference_function(model_path:str,
                            device:Optional[torch.device]=None, max_tile_size:int=1024) -> Tuple[Callable[[torch.Tensor], torch.Tensor], List[int]]:
    """
    Loads the function to make predictions. This function loads v1 models (multi-class) and v2 models (multioutput-binary).
    Args:
        model_path:
        device:
        max_tile_size: Size to tile the GeoTIFFs

    Returns:
        Function to make predictions
    """

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
    model.to(device)
    channels = get_channel_configuration_bands(config.data_params.channel_configuration)
    if config.model_params.get("model_version", "v1") == "v2":
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                          activation='sigmoid')
    else:
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                          activation='softmax')

    return inference_function, channels



def main(experiment_path:str, path_to_splits=None, train_test_split_file = None, overwrite=False, device:Optional[torch.device]=None,
         max_tile_size:int=1_024):
    """
    Compute metrics of a given experiment and saves it on the experiment_path folder

    Args:
        experiment_path: /path/to/folder/with/config.json file
        path_to_splits: e.g. /worldfloods/worldfloods_v1_0/
        overwrite:
        device:
        max_tile_size: Size to tile the GeoTIFFs
    """

    config_fp = os.path.join(experiment_path, "config.json").replace("\\","/")
    config = get_default_config(config_fp)

    inference_function, channels = load_inference_function(experiment_path, device, max_tile_size=max_tile_size)

    if path_to_splits is not None:
        config.data_params.path_to_splits = path_to_splits  # local folder where data is located
    if train_test_split_file is not None:
        config.data_params.train_test_split_file = train_test_split_file
        metrics_name = os.path.basename(train_test_split_file).split('.json')[0]
    else:
        config.data_params.train_test_split_file = ""
        metrics_name = ""

    if "filter_windows" in config["data_params"]:
        del config["data_params"]["filter_windows"]
    
    ### METRICS COMPUTATION #### 
    data_module = dataset_setup.get_dataset(config["data_params"])

    for dl, dl_name in [(data_module.test_dataloader(), "test"), (data_module.val_dataloader(), "val")]:     
        metrics_file = os.path.join(experiment_path, f"{dl_name}_{metrics_name}.json").replace("\\","/")
        fs = get_filesystem(metrics_file)
        if not overwrite and fs.exists(metrics_file):
            print(f"File {metrics_file} exists. Continue")
            continue

        mets = compute_metrics_v2(
            dl,
            inference_function, threshold_water=0.5,
            plot=False,
            mask_clouds=True)

        if hasattr(dl.dataset, "image_files"):
            mets["cems_code"] = [get_code(f) for f in dl.dataset.image_files]
        else:
            mets["cems_code"] = [get_code(f.file_name) for f in dl.dataset.list_of_windows]

        save_json(metrics_file, mets)
        
        
if __name__ == '__main__':
    from tqdm import tqdm
    import traceback
    import sys
    import argparse

    parser = argparse.ArgumentParser('Run metrics on test and val subsets for the provided models')
    parser.add_argument("--experiment_path", default="",
                        help="""
                        Path with config.json and model.pt files to load the model.
                        If not provided it will glob the --experiment_folder to compute metrics of all models
                        """)
    parser.add_argument("--experiment_folder", default="",help="""
                        Folder with folders with models. Each of the model folder is expected to have a config.json and 
                        model.pt files to load the model.
                        If --experiment_path provided will ignore this argument 
                        """)
    parser.add_argument("--max_tile_size", help="Size to tile the GeoTIFFs", type=int, default=1_024)
    parser.add_argument("--path_to_splits", required=True, help="path to test and val folders")
    parser.add_argument("--train_test_split_file", default="", help="split file with test and validation tiffs")
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    device = torch.device(args.device)

    if args.experiment_path == "":
        # glob experiment folder
        fs = get_filesystem(args.experiment_folder)
        if args.experiment_folder.startswith("gs"):
            prefix = "gs://"
        else:
            prefix = ""
        experiment_paths = [f"{prefix}{os.path.dirname(f)}" for f in fs.glob(os.path.join(args.experiment_folder,"*","config.json").replace("\\","/"))]
        assert len(experiment_paths) > 0, "No models found in "+os.path.join(args.experiment_folder,"*","config.json").replace("\\","/")
    else:
        experiment_paths = [args.experiment_path]

    for ep in tqdm(experiment_paths):
        try:
            main(experiment_path=ep, path_to_splits=args.path_to_splits, train_test_split_file = args.train_test_split_file, device=device)
        except Exception:
            print(f"Error in experiment {ep}")
            traceback.print_exc(file=sys.stdout)
