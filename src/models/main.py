import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

def setup_config(args):
    # ======================================================
    # WORLD FLOODS FLOOD EXTENT SEGMENTATION CONFIG SETUP 
    # ======================================================  
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    from src.models.utils.configuration import AttrDict
    
    # 1. Load config json from argparse input
    with open(args.config)as json_file:
        config = json.load(json_file)
    
    # 2. Add additional fields to config using worldfloods constants etc
    from src.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
    
    config['train'] = args.train
    config['test'] = args.test
    config['deploy'] = args.deploy
    
    config['wandb_entity'] = args.wandb_entity
    config['wandb_project'] = args.wandb_project
    
    config['model_params']['hyperparameters']['num_channels'] = len(CHANNELS_CONFIGURATIONS[config['model_params']['hyperparameters']['channel_configuration']])
    
    config = AttrDict.from_nested_dicts(config)

    print('Loaded Config for experiment: ', config.experiment_name)
    pp.pprint(config)
    
    # 3. return config to training
    return config


def train(config):
    # ======================================================
    # EXPERIMENT SETUP
    # ====================================================== 
    from pytorch_lightning import seed_everything
    # Seed
    seed_everything(config.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # DATASET SETUP
    print("======================================================")
    print("SETTING UP DATASET")
    print("======================================================")
    from src.models.dataset_setup import get_dataset
    dataset = get_dataset(config.data_params)
    
    
    # MODEL SETUP 
    print("======================================================")
    print("SETTING UP MODEL")
    print("======================================================")
    from src.models.model_setup import get_model
    model = get_model(config.model_params)
    
    
    # LOGGING SETUP 
    print("======================================================")
    print("SETTING UP LOGGERS")
    print("======================================================")
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
        project=config.wandb_project, 
        entity=config.wandb_entity,
        save_dir=f"{config.model_params.model_folder}/{config.experiment_name}"
    )
    
    
    # CHECKPOINTING SETUP
    print("======================================================")
    print("SETTING UP CHECKPOINTING")
    print("======================================================")
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    
    checkpoint_callback = ModelCheckpoint(
        filepath=f"{config.model_params.model_folder}/{config.experiment_name}/checkpoint",
        save_top_k=True,
        verbose=True,
        monitor='dice_loss',
        mode='min',
        prefix=''
    )
    
    early_stop_callback = EarlyStopping(
        monitor='dice_loss',
        patience=10,
        strict=False,
        verbose=False,
        mode='min'
    )
    
    callbacks = [checkpoint_callback, early_stop_callback]
    

    # TRAINING SETUP 
    print("======================================================")
    print("START TRAINING")
    print("======================================================")
    from pytorch_lightning import Trainer
    trainer = Trainer(
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=f"{config.model_params.model_folder}/{config.experiment_name}",
        accumulate_grad_batches=1,
        gradient_clip_val=0.0,
        auto_lr_find=False,
        benchmark=False,
        distributed_backend=None,
        max_epochs=config.model_params.hyperparameters.max_epochs,
        check_val_every_n_epoch=config.model_params.hyperparameters.val_every,
        log_gpu_memory=None,
        resume_from_checkpoint=None
    )
    
    trainer.fit(model, dataset)
    
    # ======================================================
    # SAVING SETUP 
    # ======================================================
    print("======================================================")
    print("FINISHED TRAINING, SAVING MODEL")
    print("======================================================")
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    wandb.save("*.pt")
    
    return 1
    

def test(opt):
    """
    Test a model:
    
    1. Load a model architecture
    
    2. Load model weights from storage into model
    
    3. Load a test dataset
    
    4. Run inference over test set and compute metrics
    
    
    5. Save metrics to:
        a) local storage
        b) gcp bucket storage
        c) wandb dashboard
        
    6. Serve metrics to Visualisation Dashboards
    """
    trainer.test(model, datamodule=mnist)
    return 0


def deploy(opt):
    """
    Deploy a model to serve predictions to Visualisation Dashboard
    
    1. Load a model architecture
    
    2. Load model weights from storage into model
    
    3. Load a dataset to deploy
    
    4. Run inference over dataset
    
    5. Serve predictions to Visualisation Dashboards
    """
    return 0


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    from pathlib import Path
    from pyprojroot import here
    # spyder up to find the root
    root = here(project_files=[".here"])
    # append to path
    sys.path.append(str(here()))
    
    parser = argparse.ArgumentParser('Area Ratios Segmentation Classifiers')
    parser.add_argument('--config', default='configurations/worldfloods_template.json')
    parser.add_argument('--cuda', '-c', default='0')
    # Mode: train, test or deploy
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--deploy', default=False, action='store_true')
    # WandB fields TODO
    parser.add_argument('--wandb_entity', default='sambuddinc')
    parser.add_argument('--wandb_project', default='worldfloods-demo')
    
    args = parser.parse_args()
    
    # Set device ids visible to CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    # Setup config
    config = setup_config(args)

    # Run training
    if config.train:
        train(config)

    # Run testing
    if config.test:
        test(config)

    # Run deployment ready inference
    if config.deploy:
        deploy(config)