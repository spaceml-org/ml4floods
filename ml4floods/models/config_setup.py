from ml4floods.models.utils.configuration import AttrDict
from ml4floods.data import utils


get_filesystem = utils.get_filesystem
load_json = utils.read_json_from_gcp
save_json = utils.write_json_to_gcp


def setup_config(args) -> AttrDict:
    """
    Loads a config file from disk/bucket. Check channel configuration is consistent and set up the number of channels
    to load the models

    Args:
        args: args to populate the config

    Returns:
        config: config object

    """
    
    # 1. Load config json from argparse input
    config = load_json(args.config)
    
    # 2. Add additional fields to config using worldfloods constants etc
    from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS

    config['resume_from_checkpoint'] = args.resume_from_checkpoint
    config['gpus'] = args.gpus

    # TODO check channel_configuration is the same in all the parts. Populate this to transforms!
    assert config['model_params']['hyperparameters']['channel_configuration'] ==  config['data_params']['channel_configuration'],\
         f"Set the same channel configuration: {config['model_params']['hyperparameters']['channel_configuration']} {config['data_params']['bands']}"
    
    config['model_params']['hyperparameters']['num_channels'] = len(CHANNELS_CONFIGURATIONS[config['model_params']['hyperparameters']['channel_configuration']])
    config['data_params']['add_mndwi_input'] = config['data_params'].get('add_mndwi_input', False)
    if config['data_params']['add_mndwi_input']:
        config['model_params']['hyperparameters']['num_channels'] += 1

    config = AttrDict.from_nested_dicts(config)

    # print(f'Loaded Config for experiment: {config.experiment_name}')
    
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
