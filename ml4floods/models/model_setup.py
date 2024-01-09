import torch
import itertools
from ml4floods.models.worldfloods_model import WorldFloodsModel, ML4FloodsModel, load_weights
from ml4floods.models.utils.configuration import AttrDict
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS, SENTINEL2_NORMALIZATION, CHANNELS_CONFIGURATIONS_LANDSAT, BANDS_S2, BANDS_L8
import numpy as np
import os
from tqdm import tqdm

from typing import (Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Tuple, Union)


# U-Net inputs must be divisible by 8
SUBSAMPLE_MODULE = {
    "unet": 8,
    "unet_dropout": 8
}

def get_model(model_params:AttrDict,
              experiment_name:Optional[str]=None, normalized_data:bool=True) -> Union[ML4FloodsModel, WorldFloodsModel]:
    """
    Creates a model from a model_params Dict. (i.e. config.model_params).
    if model_params is in test or deploy model it will load the weights in the model_params.model_folder

    Args:
        model_params:
        experiment_name:
        normalized_data:

    Returns:
        pytorch lightning model

    """
    if model_params.get("test", False) or model_params.get("deploy", False):
        assert experiment_name is not None, f"Expermient name must be set on test or deploy mode"

        if model_params.get("model_version", "v1") == "v2":
            model = ML4FloodsModel(model_params)
        else:
            model = WorldFloodsModel(model_params, normalized_data=normalized_data)

        path_to_models = os.path.join(model_params.model_folder, experiment_name, "model.pt").replace("\\", "/")
        model.load_state_dict(load_weights(path_to_models))
        print(f"Loaded model weights: {path_to_models}")
        return model

    elif model_params.get("train", False):

        if model_params.get("model_version", "v1") == "v2":
            return ML4FloodsModel(model_params)

        return WorldFloodsModel(model_params)

    else:
        raise Exception("No model type set in config e.g model_params.test == True")

        
        
def get_channel_configuration_bands(channel_configuration:str, collection_name:str="S2",
                                    as_string:bool=False) -> Union[List[int], List[str]]:
    """
    Returns 0-based list of channels of a given configuration name
    """
    if collection_name == "S2":
        channel_indexes = CHANNELS_CONFIGURATIONS[channel_configuration]
        if as_string:
            return [BANDS_S2[idx] for idx in channel_indexes]
        return channel_indexes
    elif collection_name =="Landsat":
        channel_indexes = CHANNELS_CONFIGURATIONS_LANDSAT[channel_configuration]
        if as_string:
            return [BANDS_S2[idx] for idx in channel_indexes]
        return channel_indexes
    else:
        raise NotImplementedError(f"Collection {collection_name} not implemented")

        
        
def get_model_inference_function(model: torch.nn.Module, config: AttrDict,
                                 apply_normalization:bool=True, eval_mode:bool=True,
                                 activation:Optional[str]=None,
                                 disable_pbar:bool = True,
                                 device:Optional[torch.device]=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Loads a model inference function for an specific configuration. It loads the model, the weights and ensure that
    prediction does not break bc of memory errors when predicting large tiles.

    Args:
        model :LightingModule or torch.nn.Module
        config: AttrDict with fields:
            - config.model_params.hyperparameters.model_type
            - config.model_params.hyperparameters.channel_configuration
            - config.model_params.model_version
            - config.model_params.max_tile_size
        apply_normalization:
        eval_mode: set for predicting model.eval()
        activation: activation function to apply on inference time (softmax|sigmoid). If None it will gess it from
            the model_version
        device: if None uses model.device

    Returns: callable function
    """
    print("Getting model inference function")
    model_type = config.model_params.hyperparameters.model_type
    module_shape = SUBSAMPLE_MODULE.get(model_type, 1)

    if apply_normalization:
        channel_configuration_bands = get_channel_configuration_bands(config.model_params.hyperparameters.channel_configuration)

        mean_batch = SENTINEL2_NORMALIZATION[channel_configuration_bands, 0]
        if config.data_params.add_mndwi_input:
            mean_batch = np.concatenate([mean_batch,np.array([0])],axis = 0)
        mean_batch = torch.tensor(mean_batch[None, :, None, None])  # (1, num_channels, 1, 1)

        std_batch = SENTINEL2_NORMALIZATION[channel_configuration_bands, 1]
        if config.data_params.add_mndwi_input:
            std_batch = np.concatenate([std_batch,np.array([1])],axis = 0)
        std_batch = torch.tensor(std_batch[None, :, None, None])  # (1, num_channels, 1, 1)

        def normalize(batch_image):
            assert batch_image.ndim == 4, "Expected 4d tensor"
            return (batch_image - mean_batch) / (std_batch + 1e-6)
    else:
        normalize = None

    # Guess activation function
    if activation is None:
        if model_type.endswith("ndwi"):
            activation = "None"
        elif config.model_params.get("model_version", "v1") == "v2":
            activation = "sigmoid"
        else:
            activation = "softmax"

    if activation == "None":
        activation_fun = lambda ot: ot
    elif activation == "softmax":
        activation_fun = lambda ot: torch.softmax(ot, dim=1)
    elif activation == "sigmoid":
        activation_fun = lambda ot: torch.sigmoid(ot)
    else:
        raise NotImplementedError(f"Activation function {activation} not implemented")

    device = device if device is not None else model.device

    return get_pred_function(model, device,
                             module_shape=module_shape,
                             max_tile_size=config.model_params.max_tile_size,
                             activation_fun=activation_fun,
                             disable_pbar=disable_pbar,
                             normalization=normalize, eval_mode=eval_mode)


def get_pred_function(model: torch.nn.Module, device:torch.device, module_shape: int=1, max_tile_size: int=128,
                      normalization: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                      activation_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                      disable_pbar:bool = False,
                      eval_mode:bool=True) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Given a model it returns a callable function to make inferences that:
    1) Normalize the input tensor if provided a callable normalization fun
    2) Tile the input if it's bigger than max_tile_size to avoid memory errors (see pred_by_tile fun)
    3) Checks the input to the network is divisible by module_shape and pads if needed
    (to avoid errors in U-Net like models)
    4) Applies activation function to the outputs

    Args:
        model:
        device:
        module_shape:
        max_tile_size:
        normalization:
        activation_fun:
        eval_mode:

    Returns:
        Function to make inferences:  (B, C, H, W) -> (B, C, H, W)

    """
    if eval_mode:
        model.eval()
    else:
        model.train()
    
    if normalization is None:
        normalization = lambda ti: ti

    if activation_fun is None:
        activation_fun = lambda ot: ot
    
    # Pad the input to be divisible by module_shape (otherwise U-Net model fails)
    if module_shape > 1:
        pred_fun = padded_predict(lambda ti: activation_fun(model(ti.to(device))),
                                  module_shape=module_shape)
    else:
        pred_fun = lambda ti: activation_fun(model(ti.to(device)))

    def pred_fun_final(ti):
        with torch.no_grad():
            ti_norm = normalization(ti)
            if any((s > max_tile_size for s in ti.shape[2:])):
                return predbytiles(pred_fun,
                                   input_batch=ti_norm,
                                   tile_size=max_tile_size,
                                   disable_pbar=disable_pbar)
            
            return pred_fun(ti_norm)

    return pred_fun_final


def padded_predict(predfunction: Callable, module_shape: int) -> Callable:
    """
    This function is needed for U-Net like models that require the shape to be multiple of 8 (otherwise there is an
    error in concat layer between the tensors of the upsampling and downsampling paths).

    Args:
        predfunction:
        module_shape:

    Returns:
        Function that pads the input if it is not multiple of module_shape

    """
    def predict(x: torch.Tensor):
        """

        Args:
            x:

        Returns:

        """
        shape_tensor = np.array(list(x.shape))[2:].astype(np.int64)
        shape_new_tensor = np.ceil(shape_tensor.astype(np.float32) / module_shape).astype(np.int64) * module_shape

        if np.all(shape_tensor == shape_new_tensor):
            return predfunction(x)

        pad_to_add = shape_new_tensor - shape_tensor
        refl_pad_layer = torch.nn.ReflectionPad2d((0, pad_to_add[1], 0, pad_to_add[0]))

        refl_pad_result = refl_pad_layer(x)
        pred_padded = predfunction(refl_pad_result)
        slice_ = (slice(None),
                  slice(None),
                  slice(0, shape_new_tensor[0]-pad_to_add[0]),
                  slice(0, shape_new_tensor[1]-pad_to_add[1]))

        return pred_padded[slice_]

    return predict


def predbytiles(pred_function: Callable[[torch.Tensor], torch.Tensor], input_batch: torch.Tensor,
                tile_size=1280, pad_size=32, device=torch.device("cpu"),
                disable_pbar:bool=False) -> torch.Tensor:
    """
    Apply a pred_function (usually a torch model) by tiling the input_batch array.
    The purpose is to run `pred_function(input_batch)` avoiding memory errors.
    It tiles and stiches the pateches with padding using the strategy of: https://arxiv.org/abs/1805.12219

    Args:
        pred_function: pred_function to call
        input_batch: torch.Tensor in BCHW format
        tile_size: Size of the tiles.
        pad_size: each tile is padded before calling the pred_function.
        device: Device to save the predictions

    Returns:
        torch.Tensor in BCHW format (same B, H and W as input_batch)

    """
    pred_continuous_tf = None
    assert input_batch.dim() == 4, "Expected batch of images"

    indexes_list = list(itertools.product(range(0, input_batch.shape[0]),
                                     range(0, input_batch.shape[2], tile_size),
                                     range(0, input_batch.shape[3], tile_size)))
    
    for b, i, j in tqdm(indexes_list, desc="Predicting by tiles",
                        disable=disable_pbar or (len(indexes_list) == 1)):

        slice_current = (slice(i, min(i + tile_size, input_batch.shape[2])),
                         slice(j, min(j + tile_size, input_batch.shape[3])))
        slice_pad = (slice(max(i - pad_size, 0), min(i + tile_size + pad_size, input_batch.shape[2])),
                     slice(max(j - pad_size, 0), min(j + tile_size + pad_size, input_batch.shape[3])))

        slice_save_i = slice(slice_current[0].start - slice_pad[0].start,
                             None if (slice_current[0].stop - slice_pad[0].stop) == 0 else slice_current[0].stop -
                                                                                           slice_pad[0].stop)
        slice_save_j = slice(slice_current[1].start - slice_pad[1].start,
                             None if (slice_current[1].stop - slice_pad[1].stop) == 0 else slice_current[1].stop -
                                                                                           slice_pad[1].stop)

        slice_save = (slice_save_i, slice_save_j)

        slice_prepend = (slice(b, b + 1), slice(None))
        slice_current = slice_prepend + slice_current
        slice_pad = slice_prepend + slice_pad
        slice_save = slice_prepend + slice_save

        vals_to_predict = input_batch[slice_pad]
        cnn_out = pred_function(vals_to_predict).to(device)

        assert cnn_out.dim() == 4, "Expected 4-band prediction (after softmax)"

        if pred_continuous_tf is None:
            pred_continuous_tf = torch.zeros((input_batch.shape[0], cnn_out.shape[1],
                                              input_batch.shape[2], input_batch.shape[3]),
                                             device=device)

        pred_continuous_tf[slice_current] = cnn_out[slice_save]

    return pred_continuous_tf