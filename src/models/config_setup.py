import json


class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed like attributes
        (as well as normally).
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dicts(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dicts(data[key])
                             for key in data})

def setup_config(args):
    # ======================================================
    # WORLD FLOODS FLOOD EXTENT SEGMENTATION CONFIG SETUP 
    # ======================================================  
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    
    # 1. Load config json from argparse input
    with open(args.config)as json_file:
        config = json.load(json_file)
    
    # 2. Add additional fields to config using worldfloods constants etc
    from src.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
    
    config['train'] = args.train
    config['gpus'] = args.gpus
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