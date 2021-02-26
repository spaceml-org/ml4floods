from src.models.worldfloods_model import WorldFloodsModel

def get_model(model_config):
    """
    Function to setup WorldFloodsModel
    """
    if model_config.get("test", False):
        
        return WorldFloodsModel.load_from_checkpoint(model_config.model_path, model_params=model_config)
    
    elif model_config.get("train", False):
        
        return WorldFloodsModel(model_config)
    
    else:
        raise Exception("No model type set in config e.g model_params.test == True")