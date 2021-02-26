from src.models.worldfloods_model import WorldFloodsModel

def get_model(model_config):
    """
    Function to setup WorldFloodsModel
    """
    return WorldFloodsModel(model_config)