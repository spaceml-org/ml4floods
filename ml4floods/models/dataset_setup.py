from ml4floods.preprocess.worldfloods import normalize as wf_normalization
import ml4floods.preprocess.transformations as transformations
from ml4floods.preprocess.tiling import WindowSlices, load_windows, save_windows
from glob import glob
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from ml4floods.data.worldfloods.dataset import WorldFloodsDatasetTiled
from ml4floods.data.worldfloods.lightning import WorldFloodsDataModule
from ml4floods.models.utils.configuration import AttrDict
import pytorch_lightning as pl
import os
import json
from typing import Dict, List, Callable, Tuple, Optional
from ml4floods.preprocess.worldfloods import prepare_patches
from ml4floods.models.config_setup import load_json, get_filesystem
import warnings
import numpy as np


def filenames_train_test_split(bucket_name:Optional[str], train_test_split_file:str) -> Dict[str, Dict[str, List[str]]]:
    """
    Read train test split json file from remote if needed.

    Args:
        bucket_name:
        train_test_split_file:

    Returns:

    """
    if bucket_name or train_test_split_file.startswith("gs://"):
        if not train_test_split_file.startswith("gs://"):
            train_test_split_file = f"gs://{bucket_name}/{train_test_split_file}"
        filenames_train_test = load_json(train_test_split_file)

        # Preprend the bucket name to the files
        for splitname, split in filenames_train_test.items():
            for foldername, listoftiffs in split.items():
                for idx,tiffname in enumerate(listoftiffs):
                    if not tiffname.startswith("gs://"):
                        tiffname = f"gs://{bucket_name}/{tiffname}"
                        # assert fs.exists(tiffname), f"File {tiffname} does not exists"
                        listoftiffs[idx] = tiffname
    else:
        with open(train_test_split_file, "r") as fh:
            filenames_train_test = json.load(fh)

    return filenames_train_test


def validate_worldfloods_data(path_to_splits:str,
                              split_folders:List[str] = ["train", "val", "test"],
                              folders_to_test:List[str] = ["S2", "gt"],
                              verbose:bool=True):
    """
    Check the structure of the data follows the expected convention
    Args:
        path_to_splits: path where train/test/val splits are.
        split_folders: Split folders to check
        folders_to_test: products to test (i.e. this could be S2, PERMANENTWATERJRC, floodmaps,...)
        verbose: prints success

    Raises:
        FileNotFoundError if something goes wrong

    """
    if not os.path.exists(path_to_splits):
        raise FileNotFoundError(f" Path to splits folder not found {path_to_splits}")

    for isplit in split_folders:
        folder_split = os.path.join(path_to_splits, isplit)
        if not os.path.exists(folder_split):
            raise FileNotFoundError(f"Splits folder not found {folder_split}")

        filenames = None
        for foldername in folders_to_test:
            folder_path = os.path.join(folder_split, foldername)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f" Folder not found {folder_path}")
            filenames_folder = [os.path.basename(os.path.splitext(f)[0]) for f in sorted(glob(os.path.join(folder_path, "*")))]
            if len(filenames_folder) == 0:
                raise FileNotFoundError(f"Folder {folder_path} does not have any files on it")
            if filenames is None:
                filenames = filenames_folder
            else:
                if filenames != filenames_folder:
                    raise FileNotFoundError(f"Different files {filenames} {filenames_folder}")
    if verbose:
        print("Data downloaded follows the expected format")


def process_filename_train_test(train_test_split_file:Optional[str]="gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/train_test_split.json",
                                input_folder:str="S2",
                                target_folder:str="gt",
                                bucket_id:Optional[str]=None, path_to_splits:Optional[str]=None,
                                download:Optional[Dict[str,bool]]=None) -> Dict[str,Dict[str, List[str]]]:
    """
    The train_test_split_file contains which files go to the train/val/test splits. This function validate that
    the content is as expected and that all files referred there exists. Additionally it downloads the data to loca
    if specified.

    Args:
        train_test_split_file:
        input_folder:
        target_folder:
        bucket_id:
        path_to_splits:
        download: e.g. `{"train": True, "val": False, "test": True}` to download train and val data.

    Returns:

    """

    if download is None:
        download = {"train": False, "val": False, "test": False}

    if train_test_split_file:
        filenames_train_test = filenames_train_test_split(bucket_id, train_test_split_file)
    else:
        assert (path_to_splits is not None) and os.path.exists(path_to_splits), \
            f"train_test_split_file not provided and path_to_splits folder {path_to_splits} does not exist"

        print(f"train_test_split_file not provided. We will use the content in the folder {path_to_splits}")
        filenames_train_test = {'train': {target_folder:[], input_folder:[]},
                                'test': {target_folder:[],input_folder:[]},
                                'val': {target_folder:[],input_folder:[]}}

    # loop through the naming splits
    for isplit in ["train", "test", "val"]:
        for foldername in [input_folder, target_folder]:

            # glob files in path_to_splits dir if there're not files in the given split
            if len(filenames_train_test[isplit][foldername]) == 0:
                # get the subdirectory
                assert (path_to_splits is not None) and os.path.exists(path_to_splits), \
                    f"path_to_splits {path_to_splits} doesn't exists or not provided and there're no files in split {isplit} folder {foldername}"

                path_2_glob = os.path.join(path_to_splits, isplit, foldername, "*.tif")
                filenames_train_test[isplit][foldername] = glob(path_2_glob)
                assert len(filenames_train_test[isplit][foldername]) > 0, f"No files found in {path_2_glob}"

        assert len(filenames_train_test[isplit][input_folder]) == len(filenames_train_test[isplit][target_folder]), \
            f"Different number of files in {input_folder} and {target_folder} for split {isplit}: {len(filenames_train_test[isplit][input_folder])} {len(filenames_train_test[isplit][target_folder])}"

        # check correspondence input output files (assert files exists)
        for idx, filename in enumerate(filenames_train_test[isplit][input_folder]):
            fs = get_filesystem(filename)
            assert fs.exists(filename), f"File input: {filename} does not exists"

            filename_target = filenames_train_test[isplit][target_folder][idx]
            assert fs.exists(filename_target), f"File target: {filename_target} does not exists"

            # Download if needed and replace filenames_train_test with the downloaded version
            if filename.startswith("gs://") and download[isplit]:
                assert (path_to_splits is not None) and os.path.exists(path_to_splits), \
                    f"path_to_splits {path_to_splits} doesn't exists or not provided ad requested to download the data"

                for input_target_folder in [input_folder, target_folder]:
                    folder_local = os.path.join(path_to_splits, isplit, input_target_folder)
                    os.makedirs(folder_local, exist_ok=True)
                    basename = os.path.basename(filename)
                    file_src = filenames_train_test[isplit][input_target_folder][idx]
                    file_dest = os.path.join(folder_local, basename)
                    if not os.path.isfile(file_dest):
                        fs.get_file(file_src, file_dest)
                        print(f"Downloaded ({idx}/{len(filenames_train_test[isplit][input_target_folder])}) {file_src}")
                    filenames_train_test[isplit][input_target_folder][idx] = file_dest

    return filenames_train_test


def get_dataset(data_config) -> pl.LightningDataModule:
    """
    Function to set up dataloaders for model training
    """
    
    # 1. Setup transformations for dataset
    
    train_transform, test_transform = get_transformations(data_config)

    # ======================================================
    # Obtain train/val/test files
    # ======================================================
    download = data_config.get("download")
    if download is None:
        download = {"train": data_config.get("loader_type","local")=="local",
                    "val": data_config.get("loader_type","local")=="local",
                    "test": data_config.get("loader_type","local")=="local"}

    filenames_train_test = process_filename_train_test(data_config.get("train_test_split_file"),
                                                       input_folder=data_config.input_folder,
                                                       target_folder=data_config.target_folder,
                                                       bucket_id=data_config.get("bucket_id"),
                                                       path_to_splits=data_config.get("path_to_splits"),
                                                       download=download)

    filter_windows_attr = data_config.get("filter_windows", None)
    if filter_windows_attr is not None and filter_windows_attr.get("apply", False):
        filter_windows_config = filter_windows_fun(data_config.filter_windows.version, data_config.train_test_split_file,
                                                   threshold_clouds=data_config.filter_windows.threshold_clouds,
                                                   local_destination_dir=data_config.path_to_splits)
    else:
        filter_windows_config = None

    # CREATE DATAMODULE
    datamodule = WorldFloodsDataModule(
        filenames_train_test=filenames_train_test,
        input_folder=data_config.input_folder,
        target_folder=data_config.target_folder,
        train_transformations=train_transform,
        test_transformations=test_transform,
        bands=CHANNELS_CONFIGURATIONS[data_config.channel_configuration],
        add_mndwi_input = data_config.add_mndwi_input,
        num_workers=data_config.num_workers,
        window_size=data_config.window_size,
        batch_size=data_config.batch_size,
        filter_windows= filter_windows_config
    )
    datamodule.setup()

    print("train", datamodule.train_dataset.__len__(), " tiles")
    print("val", datamodule.val_dataset.__len__(), " tiles")
    print("test", datamodule.test_dataset.__len__(), " tiles")
    
    return datamodule

def filter_windows_fun(data_version:str, train_test_split_file:str, local_destination_dir:Optional[str]=None, threshold_clouds=.5) -> Callable:
    """
    Returns a function to filter the windows in the  WorldFloodsDatasetTiled dataset. This is used for pre-filtering
    the training images to discard patches with high cloud content.

    Args:
        data_version: "v1" or "v2"
        local_destination_dir: local destination to save the json file with the windows to use
        threshold_clouds: threshold to use to filter the window (window will be filtered if it has more than threshold_clouds
        clouds)

    Returns:
        function to filter windows

    """
    if local_destination_dir is not None:
        if train_test_split_file:
            split_name = "_"+os.path.basename(train_test_split_file)
        else:
            split_name =".json"
        #windows_file = os.path.join(local_destination_dir, f"windows_{data_version}.json")
        windows_file = os.path.join(local_destination_dir, f"windows{split_name}")
    else:
        windows_file = None

    def filter_windows(tiledDataset: WorldFloodsDatasetTiled) -> List[WindowSlices]:
        if windows_file and os.path.exists(windows_file):
            selected_windows = load_windows(windows_file)
        else:
            if data_version == "v1":
                selected_windows =  prepare_patches.filter_windows_v1(tiledDataset,
                                                                      threshold_clouds=threshold_clouds)
            elif data_version == "v2":
                selected_windows = prepare_patches.filter_windows_v2(tiledDataset,
                                                                     threshold_clouds=threshold_clouds)
            else:
                raise NotImplementedError(f"Unknown ground truth version {data_version} expected v1 or v2")

            if windows_file:
                save_windows(selected_windows, windows_file)

        return selected_windows

    return filter_windows


def get_transformations(data_config) -> Tuple[Callable, Callable]:
    """
    Function to generate transformations object to pass to dataloader
    TODO: Build from config instead of using default values
    """

    train_transform = [
        transformations.InversePermuteChannels(),
        transformations.RandomRotate90(always_apply=True, p=0.5),
        transformations.Flip(always_apply=True, p=0.5)]
    
    if "train_transformation" not in data_config:
        warnings.warn("Train transformation not found in data config. Assume normalize is True")
        data_config["train_transformation"] = AttrDict({"normalize": True})

    channel_mean = None
    if data_config.train_transformation.normalize:
        channel_mean, channel_std = wf_normalization.get_normalisation(data_config.channel_configuration)
        if data_config.add_mndwi_input:
            channel_mean = np.concatenate([channel_mean,np.zeros((1,1,1))],axis = -1)
            channel_std = np.concatenate([channel_std,np.ones((1,1,1))],axis = -1)
            
        train_transform.append(transformations.Normalize(
            mean=channel_mean,
            std=channel_std,
            max_pixel_value=1))

    train_transform.extend([
        transformations.PermuteChannels(),
        transformations.ToTensor(),
    ])

    train_transform = transformations.Compose(train_transform)

    if "test_transformation" not in data_config:
        warnings.warn("Test transformation not found in data config. Assume normalize is True")
        data_config["test_transformation"] = AttrDict({"normalize": True})
    
    if data_config.test_transformation.normalize:
        if channel_mean is None:
            channel_mean, channel_std = wf_normalization.get_normalisation(data_config.channel_configuration)

        test_transform = [
        transformations.InversePermuteChannels(), 
        transformations.Normalize(
            mean=channel_mean, 
            std=channel_std, 
            max_pixel_value=1),
        transformations.PermuteChannels(),
        transformations.ToTensor(),
        ]
        test_transform = transformations.Compose(test_transform)
    else:
        test_transform = transformations.ToTensor()
    
    return train_transform, test_transform
