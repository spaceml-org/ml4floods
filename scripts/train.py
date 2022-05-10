import torch
import wandb
import argparse
import os

from ml4floods.data.utils import get_filesystem
from ml4floods.models.config_setup import save_json
from pytorch_lightning import seed_everything
from ml4floods.models.dataset_setup import get_dataset
from ml4floods.models.utils.metrics import compute_metrics_v2
from ml4floods.models.model_setup import get_model, get_model_inference_function
from ml4floods.models import worldfloods_model
from ml4floods.models.config_setup import setup_config
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
import numpy as np


def get_code(x):
    """" Get CEMS code """

    bn = os.path.basename(x)
    if bn.startswith("EMSR"):
        cems_code = bn.split("_")[0]
    else:
        cems_code = os.path.splitext(bn)[0]
    return cems_code


def train(config):
    # ======================================================
    # EXPERIMENT SETUP
    # ======================================================
    # Seed
    seed_everything(config.seed)
    
    # DATASET SETUP
    print("======================================================")
    print("SETTING UP DATASET")
    print("======================================================")
    data_module = get_dataset(config.data_params)

    # MODEL SETUP 
    print("======================================================")
    print("SETTING UP MODEL")
    print("======================================================")
    config.model_params.test = False
    config.model_params.train = True
    model = get_model(config.model_params)

    # LOGGING SETUP 
    print("======================================================")
    print("SETTING UP LOGGERS")
    print("======================================================")
    wandb_logger = WandbLogger(
        name=config.experiment_name,
        project=config.wandb_project, 
        entity=config.wandb_entity,
    )
    
    # CHECKPOINTING SETUP
    print("======================================================")
    print("SETTING UP CHECKPOINTING")
    print("======================================================")
    experiment_path = os.path.join(config.model_params.model_folder,config.experiment_name).replace("\\","/")

    checkpoint_path = os.path.join(experiment_path, "checkpoint").replace("\\","/")
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
    trainer = Trainer(
        fast_dev_run=False,
        logger=wandb_logger,
        callbacks=callbacks,
        auto_select_gpus=True,
        default_root_dir=os.path.join(config.model_params.model_folder, config.experiment_name).replace("\\","/"),
        accumulate_grad_batches=1,
        gradient_clip_val=0.0,
        auto_lr_find=False,
        benchmark=False,
        gpus=config.gpus,
        max_epochs=config.model_params.hyperparameters.max_epochs,
        check_val_every_n_epoch=config.model_params.hyperparameters.val_every,
        log_gpu_memory=None,
        resume_from_checkpoint=checkpoint_path if config.resume_from_checkpoint else None
    )
    
    trainer.fit(model, data_module)
    
    # ======================================================
    # SAVING SETUP 
    # ======================================================
    print("======================================================")
    print("FINISHED TRAINING, SAVING MODEL")
    print("======================================================")
    fs = get_filesystem(experiment_path)
    path_save_model = os.path.join(experiment_path, "model.pt").replace("\\","/")

    # More details can be found here: https://github.com/pytorch/pytorch/issues/42239
    with fs.open(path_save_model, "wb") as fh:
        torch.save(model.state_dict(), fh, _use_new_zipfile_serialization=False)

    wandb.save(path_save_model)
    wandb.finish()

    # Save cofig file in experiment_path
    config_file_path = os.path.join(experiment_path, "config.json").replace("\\","/")
    save_json(config_file_path, config)

    config["model_params"]["max_tile_size"] = 512

    # Compute metrics in test and val datasets
    if config.model_params.get("model_version", "v1") == "v2":
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                          activation='sigmoid')
    else:
        inference_function = get_model_inference_function(model, config, apply_normalization=False,
                                                          activation='softmax')


    for dl, dl_name in [(data_module.test_dataloader(), "test"), (data_module.val_dataloader(), "val")]:
        metrics_file = os.path.join(experiment_path, f"{dl_name}.json").replace("\\","/")
        if fs.exists(metrics_file):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train WorldFloods model')
    parser.add_argument('--config', default='configurations/worldfloods_template.json')
    parser.add_argument('--gpus', default=1, type=int)
    # Mode: train, test or deploy
    parser.add_argument('--resume_from_checkpoint', default=False, action='store_true')
    # WandB fields
    parser.add_argument('--wandb_entity', default='ipl_uv')
    parser.add_argument('--wandb_project', default='ml4floods-scripts')
    parser.add_argument("--experiment_name", default="")
    parser.add_argument(
        "--n_runs",
        action='store', type=int, default=1,
        help='Number of runs with different seed',
    )
    
    args = parser.parse_args()
    
    # Set device ids visible to CUDA
#     os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'FALSE'
    
    # Setup config
    config = setup_config(args)
    
    config['wandb_entity'] = args.wandb_entity
    config['wandb_project'] = args.wandb_project

    # Use custom experiment name
    if args.experiment_name != "":
        config.experiment_name = args.experiment_name

    if args.n_runs == 1:
        config["seed"] = 42
        train(config)

    else:
        seeds = np.random.randint(0, 2 ** 14, args.n_runs)

        # train several times with different seed
        for _i, s in enumerate(seeds):
            config["seed"] = s
            config["experiment_name"] = f"{config.experiment_name}_{_i:02d}"
            # Run training
            train(config)
