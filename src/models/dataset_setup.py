from src.preprocess.worldfloods import normalize as wf_normalization
import src.preprocess.transformations as transformations
from src.preprocess.tiling import WindowSize, save_tiles
from src.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
import os
from pathlib import Path
from pyprojroot import here
# spyder up to find the root
root = here(project_files=[".here"])


def get_dataset(data_config):
    """
    Function to set up dataloaders for model training
    option 1: Local
    option 2: Local Tiles (buggy)
    option 3: GCP Bucket (buggy)
    """
    
    # 1. Setup transformations for dataset
    
    train_transform, test_transform = get_transformations(data_config)

    bands = CHANNELS_CONFIGURATIONS[data_config.bands]
    window_size = WindowSize(height=data_config.window_size[0], width=data_config.window_size[1])
    
    # ======================================================
    # LOCAL PREPARATION
    # ======================================================
    if 'local' in data_config.loader_type:
        from src.data.worldfloods.lightning import WorldFloodsDataModule
        from src.data.utils import create_folder, get_files_in_bucket_directory, save_file_from_bucket
        image_count = data_config.get('image_count', -1)
        
        destination_dir = Path(root).joinpath('datasets', f"{data_config.path_to_splits}")
        filenames = {
            "train": {},
            "val": {},
            "test": {}
        }
        # Read Filenames from bucket
        for split in filenames.keys():
            filenames[split][data_config.input_folder] = get_files_in_bucket_directory(data_config.bucket_id, f"{data_config.path_to_splits}/{split}/{data_config.input_folder}", suffix=".tif")[:image_count]
            filenames[split][data_config.target_folder] = get_files_in_bucket_directory(data_config.bucket_id, f"{data_config.path_to_splits}/{split}/{data_config.target_folder}", suffix=".tif")[:image_count]

    
    # ======================================================
    # LOCAL DATASET SETUP
    # ======================================================
    if data_config.loader_type == 'local':
        print('Using local dataset for this run')
        
        # Read Files from bucket
        for split in filenames.keys():
            for k in filenames[split].keys():
                create_folder(f"{destination_dir}/{split}/{k}")
                for fp in filenames[split][k]:
                    raw_fp = fp.split(f"{split}/{k}/")[1]
                    if not os.path.isfile(f"{destination_dir}/{split}/{k}/{raw_fp}"):
                        save_file_from_bucket(data_config.bucket_id, fp, f"{destination_dir}/{split}/{k}/")
                        print(f"Loaded {fp}")
                    else:
                        print(f"{destination_dir}/{split}/{k}/{fp} already exists")
                        
        # CREATE DATASET                
        dataset = WorldFloodsDataModule(
            input_folder=data_config.input_folder,
            target_folder=data_config.target_folder,
            train_transformations=train_transform,
            test_transformations=test_transform,
            data_dir=destination_dir,
            window_size=data_config.window_size,
            batch_size=data_config.batch_size
        )
        dataset.setup()
            
    # ======================================================
    # LOCAL TILED DATASET SETUP
    # ======================================================
    elif data_config.loader_type == 'local_tiles':
        print('Using local pre-tiled dataset for this run')
        
        # Read Files from bucket
        for split in filenames.keys():
            for k in filenames[split].keys():
                create_folder(f"{destination_dir}_tiles/{split}/{k}")
                cur_bands = bands if data_config.input_folder == k else [1,]
                for fp in filenames[split][k]:
                    raw_fp = fp.split(f"{split}/{k}/")[1].split('.tif')[0]
                    if not os.path.isfile(f"{destination_dir}_tiles/{split}/{k}/{raw_fp}_tile_0.tif"):     
                        save_tiles(f"gs://{data_config.bucket_id}/{fp}", f"{destination_dir}_tiles/{split}/{k}/", cur_bands, window_size)
                        print(f'Loaded {fp}')
                    else:
                        print(f'Tiles for {fp} already exist')
           
        # CREATE DATASET
        dataset = WorldFloodsDataModule(
            input_folder=data_config.input_folder,
            target_folder=data_config.target_folder,
            train_transformations=train_transform,
            test_transformations=test_transform,
            data_dir=destination_dir,
            window_size=data_config.window_size,
            batch_size=data_config.batch_size)
        dataset.setup()
        
    
    # ======================================================
    # GCP BUCKET DATASET SETUP
    # ======================================================    
    elif data_config.loader_type == 'bucket':
        print('Using remote bucket storate dataset for this run')
        from src.data.worldfloods.lightning import WorldFloodsGCPDataModule
        
        dataset = WorldFloodsGCPDataModule(
            bucket_id=data_config.bucket_id,
            path_to_splits=data_config.path_to_splits,
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


def get_transformations(data_config):
    """
    Function to generate transformations object to pass to dataloader
    TODO: Build from config instead of using default values
    """
    channel_mean, channel_std = wf_normalization.get_normalisation(data_config.train_transformation.use_channels)

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