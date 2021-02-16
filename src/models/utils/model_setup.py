from glob import glob
import os
import itertools
import torch
import torch.nn
import numpy as np
from src.models.architectures.baselines import SimpleLinear, SimpleCNN
from src.models.architectures.unets import UNet
from src.models.utils import configuration
import time
import random
from typing import List, Union, Optional, Tuple, Callable, Dict, NamedTuple, Iterable


BANDS_S2 = ["B1", "B2", "B3", "B4", "B5",
            "B6", "B7", "B8", "B8A", "B9",
            "B10", "B11", "B12"]


CHANNELS_CONFIGURATIONS = {
    'all': list(range(len(BANDS_S2))),
    'rgb': [1, 2, 3],
    'rgbi': [1, 2, 3, 7],
    'sub_20': [1, 2, 3, 4, 5, 6, 7, 8],
    'hyperscout2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}

# TODO this is specific of worldfloods!
SENTINEL2_NORMALIZATION = np.array([
    [3787.0604973, 2634.44474043],
    [3758.07467509, 2794.09579088],
    [3238.08247208, 2549.4940614],
    [3418.90147615, 2811.78109878],
    [3450.23315812, 2776.93269704],
    [4030.94700446, 2632.13814197],
    [4164.17468251, 2657.43035126],
    [3981.96268494, 2500.47885249],
    [4226.74862547, 2589.29159887],
    [1868.29658114, 1820.90184704],
    [399.3878948,  761.3640411],
    [2391.66101119, 1500.02533014],
    [1790.32497137, 1241.9817628]], dtype=np.float32)


# U-Net inputs must be divisible by 8
SUBSAMPLE_MODULE = {
    "unet": 8
}


def load_model_weights(model: torch.nn.Module, model_file: str):
    # TODO: Just load the final_weights
    print('Using latest model: {}'.format(model_file))
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    return model_file


def load_model_architecture(model_name: str, num_class: int, num_channels: int) -> torch.nn.Module:
    if model_name == "linear":
        model = SimpleLinear(num_channels, num_class)
    elif model_name == "unet":
        model = UNet(num_channels, num_class)
    elif model_name == "simplecnn":
        model = SimpleCNN(num_channels, num_class)
    else:
        raise ModuleNotFoundError("model {} not found".format(model_name))

    print('Model  : {}'.format(model_name))
    return model


def handle_device(device='cuda:0'):
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available. use --device cpu')
        for c in range(torch.cuda.device_count()):
            print("Using device %s" % torch.cuda.get_device_name(c))
    return torch.device(device)


def set_random_seed(seed=None):
    """
    Sets the random seed

    Args:
        seed:

    Returns:

    """
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        
def model_inference_fun(opt: configuration.AttrDict) -> Callable:
    """
    Loads a model inference function for an specific configuration. It loads the model, the weights and ensure that
    prediction does not break bc of memory errors when predicting large tiles.

    Args:
        opt:

    Returns: callable function
    """
    device = handle_device(opt.device)

    model = load_model_architecture(opt.model, opt.num_class, opt.num_channels)

    if device.type.startswith('cuda'):
        model = model.to(device)

    model_weights = os.path.join(opt.model_folder, opt.model + "_final_weights.pt")

    assert os.path.exists(model_weights), f"Model weights file: {model_weights} not found"

    load_model_weights(model, model_weights)

    module_shape = SUBSAMPLE_MODULE[opt.model] if opt.model in SUBSAMPLE_MODULE else 1

    # This does not work because it expects 3 dim images (without the batch dim)
    # norm = Normalize(mean=SENTINEL2_NORMALIZATION[CHANNELS_CONFIGURATIONS[channel_configuration_name], 0], 
    #                  std=SENTINEL2_NORMALIZATION[CHANNELS_CONFIGURATIONS[channel_configuration_name], 1])
    
    channel_configuration_bands = CHANNELS_CONFIGURATIONS[opt.channel_configuration]
    mean_batch = SENTINEL2_NORMALIZATION[channel_configuration_bands, 0]
    mean_batch = torch.tensor(mean_batch[None, :, None, None])  # (1, num_channels, 1, 1)

    std_batch = SENTINEL2_NORMALIZATION[channel_configuration_bands, 1]
    std_batch = torch.tensor(std_batch[None, :, None, None])  # (1, num_channels, 1, 1)

    def normalize(batch_image):
        assert batch_image.ndim == 4, "Expected 4d tensor"
        return (batch_image - mean_batch) / (std_batch + 1e-6)

    return get_pred_function(model,
                             module_shape=module_shape, max_tile_size=opt.max_tile_size,
                             activation_fun=lambda ot: torch.softmax(ot, dim=1),
                             normalization=normalize)

        
def get_pred_function(model: torch.nn.Module, module_shape=1, max_tile_size=1280,
                      normalization: Optional[Callable] = None, activation_fun: Optional[Callable] = None) -> Callable:
    """
    Given a model it returns a callable function to make inferences that:
    1) Normalize the input tensor if provided a callable normalization fun
    2) Tile the input if it's bigger than max_tile_size to avoid memory errors (see pred_by_tile fun)
    3) Checks the input to the network is divisible by module_shape and padd if needed
    (to avoid errors in U-Net like models)
    4) Apply activation function to the outputs

    Args:
        model:
        module_shape:
        max_tile_size:
        normalization:
        activation_fun:

    Returns:
        Function to make inferences

    """

    device = model.device
    model.eval()
    
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
                                   tile_size=max_tile_size)
            
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

        :param x: BCHW tensor
        :return: BCHW tensor with the same B, H and W as x
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


def predbytiles(pred_function: Callable, input_batch: torch.Tensor,
                tile_size=1280, pad_size=32, device=torch.device("cpu")) -> torch.Tensor:
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

    for b, i, j in itertools.product(range(0, input_batch.shape[0], tile_size),
                                     range(0, input_batch.shape[2], tile_size),
                                     range(0, input_batch.shape[3], tile_size)):

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
        cnn_out = pred_function(vals_to_predict)

        assert cnn_out.dim() == 4, "Expected 4-band prediction (after softmax)"

        if pred_continuous_tf is None:
            pred_continuous_tf = torch.zeros((input_batch.shape[0], cnn_out.shape[1],
                                              input_batch.shape[2], input_batch.shape[3]),
                                             device=device)

        pred_continuous_tf[slice_current] = cnn_out[slice_save]

    return pred_continuous_tf
