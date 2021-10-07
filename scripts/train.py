import torch
from ml4floods.models.config_setup import save_json


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
    from ml4floods.models.dataset_setup import get_dataset
    dataset = get_dataset(config.data_params)
    
    
    # MODEL SETUP 
    print("======================================================")
    print("SETTING UP MODEL")
    print("======================================================")
    from ml4floods.models.model_setup import get_model
    from ml4floods.models import worldfloods_model
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

    checkpoint_path = f"{experiment_path}/checkpoint"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=True,
        verbose=True,
        monitor=config.model_params.hyperparameters.metric_monitor,
        mode=worldfloods_model.METRIC_MODE[config.model_params.hyperparameters.metric_monitor]
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config.model_params.hyperparameters.metric_monitor,
        patience=config.model_params.hyperparameters.get("early_stopping_patience", 4),
        strict=False,
        verbose=False,
        mode=worldfloods_model.METRIC_MODE[config.model_params.hyperparameters.metric_monitor]
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
        gpus=config.gpus if config.gpus != '' else None,
        max_epochs=config.model_params.hyperparameters.max_epochs,
        check_val_every_n_epoch=config.model_params.hyperparameters.val_every,
        log_gpu_memory=None,
        resume_from_checkpoint=checkpoint_path if config.resume_from_checkpoint else None
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

    save_json(config, config_file_path)
    
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
    print('Not yet implemented - please see notebook tutorials instead')
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
    print('Not yet implemented - please see notebook tutorials instead')
    return 0


if __name__ == "__main__":
    import argparse
    import os
    
    from ml4floods.models.config_setup import setup_config
    
    parser = argparse.ArgumentParser('Train WorldFloods model')
    parser.add_argument('--config', default='configurations/worldfloods_template.json')
    parser.add_argument('--gpus', default='', type=str)
    # Mode: train, test or deploy
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--resume_from_checkpoint', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--deploy', default=False, action='store_true')
    # WandB fields
    parser.add_argument('--wandb_entity', default='ml4floods')
    parser.add_argument('--wandb_project', default='worldfloods')
    
    args = parser.parse_args()
    
    # Set device ids visible to CUDA
#     os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'FALSE'
    
    # Setup config
    config = setup_config(args)
    
    config['wandb_entity'] = args.wandb_entity
    config['wandb_project'] = args.wandb_project

    # Run training
    if args.train:
        train(config)

    # Run testing
    if args.test:
        raise NotImplementedError("Test mode not implemented")
        # test(config)

    # Run deployment ready inference
    if args.deploy:
        raise NotImplementedError("Deploy mode not implemented")
