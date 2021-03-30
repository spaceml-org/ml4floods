from typing import Optional, Tuple, Union

import numpy as np
import rasterio
import torch

from ml4floods.models.utils import model_setup


def load_datasets(opt):
    layer_names = ["EMSR333_02PORTOPALO_DEL_MONIT01_v1_observed_event_a", "EMSR347_07ZOMBA_DEL_v2_observed_event_a"]
    windows = [(slice(256,256+256),slice(0,256)), (slice(256,256+256),slice(0,256))]
    channels = [model_setup.CHANNELS_CONFIGURATIONS[opt.data_params.bands], model_setup.CHANNELS_CONFIGURATIONS[opt.data_params.bands]]

    ds = DummyWorldFloodsDataset(layer_names, windows, channels)

    dl = torch.utils.data.DataLoader(ds, batch_size=1)
    return dl

@torch.no_grad()
def read_inference_pair(layer_name:str, window:Optional[Union[rasterio.windows.Window, Tuple[slice,slice]]], 
                        return_ground_truth: bool=False, channels:bool=None, 
                        return_permanent_water=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, rasterio.Affine]:
    """
    Read a pair of layers from the worldfloods bucket and return them as Tensors to pass to a model, return the transform for plotting with lat/long
    
    Args:
        layer_name: filename for layer in worldfloods bucket
        window: window of layer to use
        return_ground_truth: flag to indicate if paired gt layer should be returned
        channels: list of channels to read from the image
        return_permanent_water: Read permanent water layer raster
    
    Returns:
        (torch_inputs, torch_targets, transform): inputs Tensor, gt Tensor, transform for plotting with lat/long
    """
    tiff_inputs = f"gs://ml4floods/worldfloods/tiffimages/S2/{layer_name}.tif"
    tiff_targets = f"gs://ml4floods/worldfloods/tiffimages/gt/{layer_name}.tif"

    with rasterio.open(tiff_inputs, "r") as rst:
        inputs = rst.read((np.array(channels) + 1).tolist(), window=window)
        # Shifted transform based on the given window (used for plotting)
        transform = rst.transform if window is None else rasterio.windows.transform(window, rst.transform)
        torch_inputs = torch.Tensor(inputs.astype(np.float32)).unsqueeze(0)
    
    if return_permanent_water:
        tiff_permanent_water = f"gs://ml4floods/worldfloods/tiffimages/PERMANENTWATERJRC/{layer_name}.tif"
        with rasterio.open(tiff_permanent_water, "r") as rst:
            permanent_water = rst.read(1, window=window)  
            torch_permanent_water = torch.tensor(permanent_water)
    else:
        torch_permanent_water = torch.zeros_like(torch_inputs)
        
    if return_ground_truth:
        with rasterio.open(tiff_targets, "r") as rst:
            targets = rst.read(1, window=window)
        
        torch_targets = torch.tensor(targets).unsqueeze(0)
    else:
        torch_targets = torch.zeros_like(torch_inputs)
    
    return torch_inputs, torch_targets, torch_permanent_water, transform

class DummyWorldFloodsDataset(torch.utils.data.Dataset):
    def __init__(self, layer_names, windows, channels):
        self.inputs = []
        self.targets = []
        self.permanent_water = []
        self.plot_transforms = []
        
        for i in range(len(layer_names)):
            torch_inputs, torch_targets, torch_permanent_water, transform = read_inference_pair(layer_names[i], windows[i], return_ground_truth=True, channels=channels[i])
            
            self.inputs.append(torch_inputs)
            self.targets.append(torch_targets)
            self.permanent_water.append(torch_permanent_water)
            self.plot_transforms.append(transform)
                    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        this_dict = {
            'input': self.inputs[idx],
            'target': self.targets[idx],
            'permanent_water': self.permanent_water[idx],
            'plot_transforms': self.plot_transforms[idx]
        }
        return self.inputs[idx].squeeze(), self.targets[idx].squeeze().long()