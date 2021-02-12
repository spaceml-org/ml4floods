from glob import glob
import os
from natsort import natsorted
import itertools
import torch
import numpy as np
from src.models.architectures.baselines import SimpleLinear, SimpleCNN
from src.models.architectures.unets import UNet
import time
import random


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


def load_model_weights(model, model_folder, mode_train=False):
    # TODO: Just load the final_weights
    files = natsorted(glob(os.path.join(model_folder, '*_weights.pt')))
    if len(files) == 0:
        if mode_train:
            return None
        else:
            raise FileNotFoundError(f"Model weights not found in folder {model_folder}")
    # print('Model files       : {}'.format(files))
    model_file = files[-1]
    print('Using latest model: {}'.format(model_file))
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    return model_file


def create_model(model_name, num_class, num_channels):
    if model_name is None:
        print('Expecting a --model argument')
        quit()
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
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        
def model_inference_fun(opt):
    device = handle_device(opt.device)

    model = create_model(opt.model, opt.num_class, opt.num_channels)

    if device.type.startswith('cuda'):
        model = model.to(device)

    load_model_weights(model, opt.model_folder, mode_train=False)

    module_shape = SUBSAMPLE_MODULE[opt.model] if opt.model in SUBSAMPLE_MODULE else 1

    # This does not work because it expects 3 dim images (without the batch dim)
    # norm = Normalize(mean=SENTINEL2_NORMALIZATION[CHANNELS_CONFIGURATIONS[channel_configuration_name], 0], 
    #                  std=SENTINEL2_NORMALIZATION[CHANNELS_CONFIGURATIONS[channel_configuration_name], 1])
    
    channel_configuration_bands = CHANNELS_CONFIGURATIONS[opt.channel_configuration]
    mean_batch = SENTINEL2_NORMALIZATION[channel_configuration_bands, 0]
    mean_batch = torch.tensor(mean_batch[None,:,None,None]) # (1, num_channels)

    std_batch = SENTINEL2_NORMALIZATION[channel_configuration_bands, 1]
    std_batch = torch.tensor(std_batch[None,:,None,None]) # (1, num_channels)

    def normalize(batch_image):
        assert batch_image.ndim == 4, "Expected 4d tensor"
        return (batch_image - mean_batch) / (std_batch + 1e-6)

    return get_pred_function(model, device=device,
                             module_shape=module_shape, max_tile_size=opt.max_tile_size,
                             normalization=normalize)

        
def get_pred_function(model, device=torch.device("cuda:0"), module_shape=1, max_tile_size=1280, 
                      normalization=None):
    
    if device.type.startswith('cuda'):
        model = model.to(device)
    model.eval()
    
    if normalization is None:
        normalization = lambda ti: ti
    
    # Pad the input to be divisible by module_shape (otherwise U-Net model fails)
    if module_shape > 1:
        pred_fun = padded_predict(lambda ti: torch.softmax(model(ti.to(device)), dim=1),
                                  module_shape=module_shape)
    else:
        pred_fun = lambda ti: torch.softmax(model(ti.to(device)), dim=1)

    def pred_fun_final(ti):
        with torch.no_grad():
            if any((s > max_tile_size for s in ti.shape[2:])):
                return predbytiles(pred_fun,
                                   input_batch=normalization(ti),
                                   tile_size=max_tile_size)
            
            return pred_fun(normalization(ti))

    return pred_fun_final


def padded_predict(predfunction, module_shape):
    def predict(x):
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


def predbytiles(pred_function, input_batch, tile_size=1280, pad_size=32, device=torch.device("cpu")):
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