import torch
import json


def train(config):
    # ======================================================
    # EXPERIMENT SETUP
    # ====================================================== 
    from pytorch_lightning import seed_everything
    # Seed
    seed_everything(config.seed)
    
    
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
    config.model_params.test = False
    config.model_params.train = True
    model = get_model(config.model_params)
    
    # LOGGING SETUP 
    print("======================================================")
    print("SETTING UP LOGGERS")
    print("======================================================")
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
        name=config.experiment_name,
        project=config.wandb_project, 
        entity=config.wandb_entity,
#         save_dir=f"{config.model_params.model_folder}/{config.experiment_name}"
    )
    
    # CHECKPOINTING SETUP
    print("======================================================")
    print("SETTING UP CHECKPOINTING")
    print("======================================================")
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    experiment_path = f"{config.model_params.model_folder}/{config.experiment_name}"

    checkpoint_callback = ModelCheckpoint(
        filepath=f"{experiment_path}/checkpoint",
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
        gpus=config.gpus,
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
    from pytorch_lightning.utilities.cloud_io import atomic_save
    atomic_save(model.state_dict(), f"{experiment_path}/model.pt")
    torch.save(model.state_dict(), os.path.join(wandb_logger.save_dir, 'model.pt'))
    wandb.save(os.path.join(wandb_logger.save_dir, 'model.pt'))
    wandb.finish()

    # Save cofig file in experiment_path
    config_file_path = f"{experiment_path}/config.json"

    if config_file_path.startswith("gs://"):
        from google.cloud import storage
        splitted_path = config_file_path.replace("gs://", "").split("/")
        bucket_name = splitted_path[0]
        blob_name = "/".join(splitted_path[1:])
        bucket = storage.Client().get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            data=json.dumps(config),
            content_type='application/json'
        )
    else:
        with open(config_file_path, "w") as fh:
            json.dump(config, fh)
    
    return 1
    

def test(config):
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
    config.model_params.test = True
    config.model_params.train = False
    config.model_params.model_path = f"{config.model_params.model_folder}/latest-run/files/model.pt"
    model = get_model(config.model_params)
    
    
#     trainer.test(model, datamodule=mnist)
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
    import os
    import sys
    from pathlib import Path
    from pyprojroot import here
    # spyder up to find the root
    root = here(project_files=[".here"])
    # append to path
    sys.path.append(str(here()))
    
    from src.models.config_setup import setup_config
    
    parser = argparse.ArgumentParser('Area Ratios Segmentation Classifiers')
    parser.add_argument('--config', default='configurations/worldfloods_template.json')
    parser.add_argument('--gpus', default='0', type=str)
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
    if args.train:
        train(config)

    # Run testing
    if args.test:
        test(config)

    # Run deployment ready inference
    if args.deploy:
        deploy(config)