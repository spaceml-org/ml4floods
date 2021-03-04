from src.preprocess.worldfloods import normalize as wf_normalization
import src.preprocess.transformations as transformations
from src.preprocess.tiling import WindowSize, save_tiles
from src.data import utils
from src.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
import os
from pyprojroot import here
# spyder up to find the root
root = here(project_files=[".here"])
import io
import json
from typing import Dict, List, Callable, Tuple, Optional


def filenames_train_test_split(bucket_name:Optional[str], train_test_split_file:str) -> Dict[str, Dict[str, List[str]]]:
    """
    Read train test split file from remote

    Args:
        bucket_name:
        train_test_split_file:

    Returns:

    """
    if bucket_name != "" and bucket_name is not None:
        from google.cloud import storage
        client = storage.Client()
        with io.BytesIO() as file_obj:
            client.download_blob_to_file(f"gs://{bucket_name}/{train_test_split_file}", file_obj)
            file_obj.seek(0)
            return json.load(file_obj)
    else:
        with open(train_test_split_file, "r") as fh:
            return json.load(fh)


def get_dataset(data_config):
    """
    Function to set up dataloaders for model training
    option 1: Local
    option 2: Local Tiles (buggy)
    option 3: GCP Bucket (buggy)
    """
    
    # 1. Setup transformations for dataset
    
    train_transform, test_transform = get_transformations(data_config)

    bands = CHANNELS_CONFIGURATIONS[data_config.channel_configuration]
    window_size = WindowSize(height=data_config.window_size[0], width=data_config.window_size[1])

    # ======================================================
    # LOCAL PREPARATION
    # ======================================================
    local_destination_dir = data_config.path_to_splits
    filenames_train_test = filenames_train_test_split(data_config.bucket_id, data_config.train_test_split_file)
    
    # ======================================================
    # LOCAL DATASET SETUP
    # ======================================================
    if data_config.loader_type == 'local':
        from src.data.worldfloods.lightning import WorldFloodsDataModule
        print('Using local dataset for this run')
        
        # Read Files from bucket and copy them in local_destination_dir
        download_tiffs_from_bucket(data_config.bucket_id,
                                   [data_config.input_folder, data_config.target_folder],
                                   filenames_train_test, local_destination_dir)
                        
        # CREATE DATAMODULE
        dataset = WorldFloodsDataModule(
            input_folder=data_config.input_folder,
            target_folder=data_config.target_folder,
            train_transformations=train_transform,
            test_transformations=test_transform,
            data_dir=local_destination_dir,
            bands=bands,
            window_size=data_config.window_size,
            batch_size=data_config.batch_size
        )
        dataset.setup()
            
    # ======================================================
    # LOCAL TILED DATASET SETUP
    # ======================================================
    # elif data_config.loader_type == 'local_tiles':
    #     print('Using local pre-tiled dataset for this run')
    #
    #     # Read Files from bucket
    #     for split in filenames_train_test.keys():
    #         for k in filenames_train_test[split].keys():
    #             create_folder(f"{local_destination_dir}_tiles/{split}/{k}")
    #             cur_bands = bands if data_config.input_folder == k else [1,]
    #             for fp in filenames_train_test[split][k]:
    #                 raw_fp = fp.split(f"{split}/{k}/")[1].split('.tif')[0]
    #                 if not os.path.isfile(f"{local_destination_dir}_tiles/{split}/{k}/{raw_fp}_tile_0.tif"):
    #                     save_tiles(f"gs://{data_config.bucket_id}/{fp}", f"{local_destination_dir}_tiles/{split}/{k}/", cur_bands, window_size)
    #                     print(f'Loaded {fp}')
    #                 else:
    #                     print(f'Tiles for {fp} already exist')
    #
    #     # CREATE DATASET
    #     dataset = WorldFloodsDataModule(
    #         input_folder=data_config.input_folder,
    #         target_folder=data_config.target_folder,
    #         train_transformations=train_transform,
    #         test_transformations=test_transform,
    #         data_dir=destination_dir,
    #         bands=bands,
    #         window_size=data_config.window_size,
    #         batch_size=data_config.batch_size)
    #     dataset.setup()
        
    
    # ======================================================
    # GCP BUCKET DATASET SETUP
    # ======================================================    
    elif data_config.loader_type == 'bucket':
        print('Using remote bucket storate dataset for this run')
        from src.data.worldfloods.lightning import WorldFloodsGCPDataModule
        
        dataset = WorldFloodsGCPDataModule(
            bucket_id=data_config.bucket_id,
            filenames_train_test=filenames_train_test,
            input_folder=data_config.input_folder,
            target_folder=data_config.target_folder,
            window_size=window_size,
            bands=bands,
            train_transformations=train_transform,
            test_transformations=test_transform,
            batch_size=data_config.batch_size
        )
        dataset.setup()
        
    else:
        raise Exception(f"No dataset implemented for loader_type: {data_config.loader_type}")
        
    
    print("train", dataset.train_dataset.__len__(), " tiles")
    print("val", dataset.val_dataset.__len__(), " tiles")
    print("test", dataset.test_dataset.__len__(), " tiles")
    
    return dataset


def download_tiffs_from_bucket(bucket_id, input_target_folders, filenames, local_destination_dir, verbose=False):
    for split in filenames.keys():
        for input_target_folder in input_target_folders:
            folder_local = os.path.join(local_destination_dir, split, input_target_folder)
            os.makedirs(folder_local, exist_ok=True)
            for fp in filenames[split][input_target_folder]:
                basename = os.path.basename(fp)
                file_dest = os.path.join(folder_local, basename)
                if not os.path.isfile(file_dest):
                    save_file(bucket_id, fp, file_dest)
                    if verbose:
                        print(f"Loaded {fp}")
                else:
                    if verbose:
                        print(f"{file_dest} already exists")


def save_file(bucket_id, remote_blob_name, local_file):
    from google.cloud import storage
    # get blob
    blob = storage.Client().get_bucket(bucket_id).get_blob(remote_blob_name)
    blob.download_to_filename(local_file)


def get_transformations(data_config) -> Tuple[Callable, Callable]:
    """
    Function to generate transformations object to pass to dataloader
    TODO: Build from config instead of using default values
    """
    channel_mean, channel_std = wf_normalization.get_normalisation(data_config.channel_configuration)

    train_transform = transformations.Compose([
        transformations.InversePermuteChannels(),
        transformations.RandomRotate90(always_apply=True, p=0.5),
        transformations.Flip(always_apply=True, p=0.5),
        transformations.Normalize(
            mean=channel_mean, 
            std=channel_std, 
            max_pixel_value=1),
        transformations.PermuteChannels(), 
        transformations.ToTensor(),
    ])

    test_transform = transformations.Compose([ 
        transformations.InversePermuteChannels(), 
        transformations.Normalize(
            mean=channel_mean, 
            std=channel_std, 
            max_pixel_value=1),
        transformations.PermuteChannels(),
        transformations.ToTensor(),
    ])
    
    return train_transform, test_transform