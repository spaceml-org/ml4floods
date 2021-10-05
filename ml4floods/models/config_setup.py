import json
from typing import Dict
from ml4floods.models.utils.configuration import AttrDict
from typing import Union
from pathlib import Path
import fsspec


def get_filesystem(path: Union[str, Path]):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0],requester_pays = True)
    else:
        # use local filesystem
        return fsspec.filesystem("file",requester_pays = True)


def load_json(filename) ->Dict:
    """Loads a json file possibly from the gcp bucket if name start with gs:// """

    fs = get_filesystem(filename,requester_pays = True)
    with fs.open(filename, "rb") as fh:
        return json.load(fh)


def setup_config(args) -> AttrDict:
    """
    Loads a config file from disk/bucket. Check channel configuration is consistent and set up the number of channels
    to load the models

    Args:
        args: args to populate the config

    Returns:
        config: config object

    """
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    
    # 1. Load config json from argparse input
    config = load_json(args.config)
    
    # 2. Add additional fields to config using worldfloods constants etc
    from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS

    config['resume_from_checkpoint'] = args.resume_from_checkpoint
    config['train'] = args.train
    config['gpus'] = args.gpus
    config['test'] = args.test
    config['deploy'] = args.deploy

    # TODO check channel_configuration is the same in all the parts. Populate this to transforms!
    assert config['model_params']['hyperparameters']['channel_configuration'] ==  config['data_params']['channel_configuration'],\
         f"Set the same channel configuration: {config['model_params']['hyperparameters']['channel_configuration']} {config['data_params']['bands']}"
    
    config['model_params']['hyperparameters']['num_channels'] = len(CHANNELS_CONFIGURATIONS[config['model_params']['hyperparameters']['channel_configuration']])
    
    config = AttrDict.from_nested_dicts(config)

    print('Loaded Config for experiment: ', config.experiment_name)
    pp.pprint(config)
    
    # 3. return config to training
    return config


def get_default_config(config_fp) -> AttrDict:
    """
    Loads a config json file (e.g. from src/models/configurations/worldfloods_template.json) as a config object
    with defaults on gpus

    Args:
        config_fp: path to config file

    Returns:
        config: config object
    """
    import argparse
    parser = argparse.ArgumentParser('WorldFloods')
    parser.add_argument('--config', default=config_fp)
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--resume_from_checkpoint', default=False, action='store_true')
    # Mode: train, test or deploy
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--deploy', default=False, action='store_true')

    args, _ = parser.parse_known_args()
    
    config = setup_config(args)
    return config


def save_json(config:AttrDict, config_file_path:str) -> None:
    """
    Saves a config file posibly in the google bucket

    Args:
        config: config dict to save
        config_file_path: location to save it

    Returns:

    """
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