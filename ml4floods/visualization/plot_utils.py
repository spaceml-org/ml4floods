import rasterio
from rasterio import plot as rasterioplt
import rasterio.windows
from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np
from typing import Union, Optional
from ml4floods.data.worldfloods.configs import BANDS_S2
from ml4floods.data.worldfloods import configs
import matplotlib.pyplot as plt


# SEABORN SETTINGS
import seaborn as sns
sns.set_context(context='talk',font_scale=0.7)


COLORS_WORLDFLOODS = np.array(configs.COLORS_WORLDFLOODS)

COLORS_WORLDFLOODS_V1_1 = np.array([[0, 0, 0], # invalid
                               [139, 64, 0], # land
                               [0, 0, 139], # water
                               [220, 220, 220]], # cloud
                              dtype=np.float32) / 255

INTERPRETATION_WORLDFLOODS = ["invalid", "land", "water", "cloud"]

COLORS_WORLDFLOODS_PERMANENT = np.array([[0, 0, 0], # 0: invalid
                                         [139, 64, 0], # 1: land
                                         [237, 0, 0], # 2: flood_water
                                         [220, 220, 220], # 3: cloud
                                         [0, 0, 139], # 4: permanent_water
                                         [60, 85, 92]], # 5: seasonal_water
                                        dtype=np.float32) / 255

INTERPRETATION_WORLDFLOODS_PERMANENT = ["invalid", "land", "flood water", "cloud", "permanent water", "seasonal water"]


def get_cmap_norm_colors(color_array, interpretation_array):
    cmap_categorical = colors.ListedColormap(color_array)
    norm_categorical = colors.Normalize(vmin=-.5,
                                        vmax=color_array.shape[0]-.5)
    patches = []
    for c, interp in zip(color_array, interpretation_array):
        patches.append(mpatches.Patch(color=c, label=interp))
    
    return cmap_categorical, norm_categorical, patches


def plot_tiff_image(file_path: str, **kwargs):

    # open image with rasterio
    with rasterio.open(file_path) as src:
        
        # plot image
        fig = rasterioplt.show(src.read(), transform=src.transform, **kwargs)
    
    return fig

def get_image_transform(input:Union[str, np.ndarray], **kwargs):
    if "transform" in kwargs:
        transform = kwargs["transform"]
    else:
        transform = None

    if "bands" in kwargs:
        bands = kwargs["bands"]
        bands_rasterio = [b+1 for b in bands]
    else:
        bands = None
        bands_rasterio = None

    if "window" in kwargs:
        window = kwargs["window"]
    else:
        window = None

    if isinstance(input, str):
        with rasterio.open(input) as rst:
            output = rst.read(bands_rasterio, window=window)
            transform = rst.transform if window is None else rasterio.windows.transform(window, rst.transform)
    else:
        window_slices = window.toslices()
        output = input[bands]
        output = output[:, window_slices]
        if transform is not None:
            transform = transform if window is None else rasterio.windows.transform(window, transform)

    return output, transform


def plot_s2_rbg_image(input: Union[str, np.ndarray], **kwargs):
    kwargs["bands"] = (3, 2, 1)
    image, transform = get_image_transform(input, **kwargs)

    rgb = np.clip(image/3000.,0,1)

    rasterioplt.show(rgb, transform=transform, **kwargs)


def plot_s2_swirnirred_image(input: Union[str, np.ndarray], **kwargs):
    kwargs["bands"] = [BANDS_S2.index(b) for b in ["B11", "B8", "B4"]]
    image, transform = get_image_transform(input, **kwargs)
    rgb = np.clip(image / 3000., 0, 1)

    rasterioplt.show(rgb, transform=transform, **kwargs)


def plots_preds_v1(prediction: Union[str, np.ndarray], **kwargs):
    kwargs["bands"] = [0]
    prediction, transform = get_image_transform(prediction, **kwargs)
    prediction_show = prediction[0] + 1
    cmap_preds, norm_preds, patches_preds = get_cmap_norm_colors(configs.COLORS_WORLDFLOODS,
                                                                 INTERPRETATION_WORLDFLOODS)

    rasterioplt.show(prediction_show, transform=transform, cmap=cmap_preds, norm=norm_preds,
                     interpolation='nearest',**kwargs)

    if kwargs.get("legend", True):
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            ax = plt.gca()
        ax.legend(handles=patches_preds,
                  loc='upper right')


def plot_gt_v1(target: Union[str, np.ndarray], **kwargs):
    kwargs["bands"] = [0]
    target, transform = get_image_transform(target, **kwargs)
    target = target[0]
    cmap_preds, norm_preds, patches_preds = get_cmap_norm_colors(configs.COLORS_WORLDFLOODS,
                                                                 INTERPRETATION_WORLDFLOODS)

    rasterioplt.show(target, transform=transform, cmap=cmap_preds, norm=norm_preds,
                     interpolation='nearest', **kwargs)

    if kwargs.get("legend",True):
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            ax = plt.gca()
        ax.legend(handles=patches_preds,
                  loc='upper right')


def gt_v1_with_permanent_water(gt: np.ndarray, permanent_water: np.ndarray) -> np.ndarray:
    """ Permanent water taken from: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_2_YearlyHistory"""
    gt[(gt == 2) & (permanent_water == 3)] = 4  # set as permanent_water
    gt[(gt == 2) & (permanent_water == 2)] = 5  # set as seasonal water

    return gt


def plot_gt_v1_with_permanent(target: Union[str, np.ndarray], permanent: Optional[Union[str, np.ndarray]]=None,
                              **kwargs):
    kwargs["bands"] = [0]
    target, transform = get_image_transform(target, **kwargs)
    target= target[0]
    if permanent is not None:
        permanent, _ = get_image_transform(permanent, **kwargs)
        permanent = permanent[0]
        target = gt_v1_with_permanent_water(target, permanent)

    cmap_gt, norm_gt, patches_gt = get_cmap_norm_colors(COLORS_WORLDFLOODS_PERMANENT, INTERPRETATION_WORLDFLOODS_PERMANENT)

    rasterioplt.show(target, transform=transform, cmap=cmap_gt, norm=norm_gt,
                     interpolation='nearest', **kwargs)

    if kwargs.get("legend", True):
        if "ax" in kwargs:
            ax = kwargs["ax"]
        else:
            ax = plt.gca()
        ax.legend(handles=patches_gt,
                  loc='upper right')


def plot_s2_cloud_prob_image(file_path: str, **kwargs):
    # open image with rasterio
    with rasterio.open(file_path) as src:
        # assert the image has 15 channels
        assert src.meta["count"] == 15
        
        # plot image
        rasterioplt.show(src.read(15), transform=src.transform, **kwargs)
    
    return None