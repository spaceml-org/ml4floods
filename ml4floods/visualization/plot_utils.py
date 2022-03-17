import rasterio
from rasterio import plot as rasterioplt
import rasterio.windows
from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np
from typing import Union, Optional, List, Tuple
from ml4floods.data.worldfloods.configs import BANDS_S2, CHANNELS_CONFIGURATIONS
from ml4floods.data.worldfloods import configs
from ml4floods.data import utils


COLORS_WORLDFLOODS = np.array(configs.COLORS_WORLDFLOODS)

COLORS_WORLDFLOODS_V1_1 = np.array([[0, 0, 0], # invalid
                               [139, 64, 0], # land
                               [0, 0, 139], # water
                               [220, 220, 220]], # cloud
                              dtype=np.float32) / 255

INTERPRETATION_WORLDFLOODS = ["invalid", "land", "water", "cloud"]
INTERPRETATION_INVLANDWATER = ["invalid", "land", "water"]

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


def _read_data(filename:str,
               bands_rasterio:Optional[List[int]]=None,
               window:Optional[rasterio.windows.Window]=None,
               size_read:Optional[int]=None) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Reads data from filename with rasterio possibly from the pyramids if `size_read` is not None.
    Returns a tuple with the data read and the affine transformation.

    Args:
        filename:
        bands_rasterio:
        window:
        size_read:

    Returns:
        (C, H, W) array and affine transformation
    """

    with utils.rasterio_open_read(filename) as rst:
        if size_read is not None:
            if bands_rasterio is None:
                n_bands = rst.count
            else:
                n_bands = len(bands_rasterio)

            if window is None:
                shape = rst.shape
            else:
                shape = window.height, window.width

            if (size_read >= shape[0]) and (size_read >= shape[1]):
                out_shape = (n_bands, )+ shape
                input_output_factor = 1
            elif shape[0] > shape[1]:
                out_shape = (n_bands, size_read, int(round(shape[1]/shape[0] * size_read)))
                input_output_factor = shape[0]  / size_read # > 1
            else:
                out_shape = (n_bands, int(round(shape[0] / shape[1] * size_read)), size_read)
                input_output_factor = shape[1] / size_read # > 1
        else:
            out_shape = None
            input_output_factor = None

        output = rst.read(bands_rasterio, window=window, out_shape=out_shape)
        transform = rst.transform if window is None else rasterio.windows.transform(window, rst.transform)

    if input_output_factor is not None:
        transform = rasterio.Affine(transform.a * input_output_factor, transform.b, transform.c,
                                    transform.d, transform.e * input_output_factor, transform.f)

    return output, transform


def get_image_transform(array_or_file:Union[str, np.ndarray],
                        transform:Optional[rasterio.Affine]=None,
                        bands:List[int]=None,
                        window:Optional[rasterio.windows.Window]=None,
                        size_read:Optional[int]=None) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    Reads certain bands and window from `array_or_file`. If `array_or_file` is a file with `size_read` we can read
    from the pyramids of the data to speed up plotting.

    Args:
        array_or_file: array or file to read the data.
        transform: if `array_or_file` is a `np.array`, this current affine transform of it.
        bands: 0-based bands to read.
        window: `rasterio.windows.Window` to read
        size_read: if `array_or_file` is a string, this will be the max size of height and width. It is used to read
        from the pyramids  of the file.

    Returns:
        (C, H, W) array and affine transformation

    """

    if bands is not None:
        bands_rasterio = [b+1 for b in bands]
    else:
        bands_rasterio = None

    if isinstance(array_or_file, str):
        return _read_data(array_or_file, bands_rasterio, window=window, size_read=size_read)

    if hasattr(array_or_file, "cpu"):
        array_or_file = array_or_file.cpu()

    array_or_file = np.array(array_or_file)
    if window is None:
        window_slices = (slice(None), slice(None), slice(None))
    else:
        window_slices = (slice(None),) + window.toslices()

    output = array_or_file[bands, ...]
    output = output[window_slices]
    if transform is not None:
        transform = transform if window is None else rasterio.windows.transform(window, transform)

    return output, transform


def plot_s2_rbg_image(input: Union[str, np.ndarray], transform:Optional[rasterio.Affine]=None,
                      window:Optional[rasterio.windows.Window]=None,
                      max_clip_val:Optional[float]=3000.,
                      min_clip_val:Optional[float]=0.,
                      channel_configuration:str="all",
                      size_read:Optional[int]=None,
                      **kwargs):
    """
    Plot bands B4, B3, B2 of a Sentinel-2 image. Input could be an array or a str. Values are clipped to 3000
    (it assumes the image has values in [0, 10_000] -> https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2 )

    Tip: Use `size_read` to read from the image pyramid if input is a COG GeoTIFF to speed up the reading.

    Args:
        input: str of array with (C, H, W) configuration.
        transform: geospatial transform if input is an array
        window: window to read from the input
        max_clip_val: value to clip the the input for visualization
        min_clip_val: value to clip the the input for visualization
        channel_configuration: Expected bands of the inputs
        size_read: max size to read. Use this to read from the overviews of the image.
        **kwargs: extra args for rasterio.plot.show

    Returns:
        ax : matplotlib Axes
            Axes with plot.

    """
    band_names_current_image = [BANDS_S2[iband] for iband in CHANNELS_CONFIGURATIONS[channel_configuration]]
    bands = [band_names_current_image.index(b) for b in ["B4", "B3", "B2"]]
    image, transform = get_image_transform(input, transform=transform, bands=bands, window=window,
                                           size_read=size_read)

    if max_clip_val is not None:
        min_clip_val = 0 if min_clip_val is None else min_clip_val
        image = np.clip((image-min_clip_val)/(max_clip_val - min_clip_val), 0, 1)

    return rasterioplt.show(image, transform=transform, **kwargs)


def plot_s2_swirnirred_image(input: Union[str, np.ndarray],
                             transform:Optional[rasterio.Affine]=None,
                             window:Optional[rasterio.windows.Window]=None,
                             max_clip_val: Optional[float] = 3000.,
                             min_clip_val: Optional[float] = 0.,
                             channel_configuration="all",
                             size_read:Optional[int]=None,
                             **kwargs):
    """
    Plot bands B11, B8, B4 of a Sentinel-2 image. Input could be an array or a str. Values are clipped to 3000
    (it assumes the image has values in [0, 10_000] -> https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2 )

    Tip: Use `size_read` to read from the image pyramid if input is a COG GeoTIFF to speed up the reading.

    Args:
        input: str of array with (C, H, W) configuration.
        transform: geospatial transform if input is an array
        window: window to read from the input
        max_clip_val: value to clip the the input for visualization
        min_clip_val: value to clip the the input for visualization
        channel_configuration: Expected bands of the inputs
        size_read: max size to read. Use this to read from the overviews of the image.
        **kwargs: extra args for rasterio.plot.show

    Returns:
        ax : matplotlib Axes
            Axes with plot.

    """
    band_names_current_image = [BANDS_S2[iband] for iband in CHANNELS_CONFIGURATIONS[channel_configuration]]
    bands = [band_names_current_image.index(b) for b in ["B11", "B8", "B4"]]
    image, transform = get_image_transform(input, transform=transform, bands=bands, window=window,
                                           size_read=size_read)

    if max_clip_val is not None:
        min_clip_val = 0 if min_clip_val is None else min_clip_val
        image = np.clip((image-min_clip_val)/(max_clip_val - min_clip_val), 0, 1)

    return rasterioplt.show(image, transform=transform, **kwargs)


def plots_preds_v1(prediction: Union[str, np.ndarray],transform:Optional[rasterio.Affine]=None,
                   window:Optional[rasterio.windows.Window]=None, legend=True,
                   size_read:Optional[int]=None,
                   **kwargs):
    """
    Prediction expected to be {0: land, 1: water, 2: cloud}
    Args:
        prediction: (1, H, W) tensor or path to tiff tile with codes {0: land, 1: water, 2: cloud}
        transform:
        window:
        legend:
        size_read:
        **kwargs:

    Returns:

    """
    prediction, transform = get_image_transform(prediction,transform=transform, bands=[0], window=window,
                                                size_read=size_read)
    prediction_show = prediction[0] + 1
    cmap_preds, norm_preds, patches_preds = get_cmap_norm_colors(configs.COLORS_WORLDFLOODS,
                                                                 INTERPRETATION_WORLDFLOODS)

    ax = rasterioplt.show(prediction_show, transform=transform, cmap=cmap_preds, norm=norm_preds,
                          interpolation='nearest',**kwargs)

    if legend:
        ax.legend(handles=patches_preds,
                  loc='upper right')

    return ax


def plots_preds_v2(prediction: Union[str, np.ndarray],transform:Optional[rasterio.Affine]=None,
                   window:Optional[rasterio.windows.Window]=None, legend=True,
                   size_read:Optional[int]=None,
                   **kwargs):
    """
    Prediction expected binary (H, W). This function plots only the land/water mask
    Args:
        prediction: (H, W) binary mask
        transform: geotransform
        window: window to read
        legend: plot legend
        size_read:
        **kwargs:

    Returns:

    """
    prediction, transform = get_image_transform(prediction, transform=transform, bands=[0], window=window,
                                                size_read=size_read)
    prediction_show = prediction + 1
    cmap_preds, norm_preds, patches_preds = get_cmap_norm_colors(configs.COLORS_WORLDFLOODS_INVLANDWATER,
                                                                 INTERPRETATION_INVLANDWATER)

    ax = rasterioplt.show(prediction_show, transform=transform, cmap=cmap_preds, norm=norm_preds,
                          interpolation='nearest',**kwargs)

    if legend:
        ax.legend(handles=patches_preds,
                  loc='upper right')

    return ax

def plot_gt_v1(target: Union[str, np.ndarray], transform:Optional[rasterio.Affine]=None,
               window:Optional[rasterio.windows.Window]=None,
               legend=True, size_read:Optional[int]=None, **kwargs):
    """
    ground truth `target` expected to be {0: invalid: 1: land, 2: water, 3: cloud}
    Args:
        target:
        transform:
        window:
        legend:
        size_read:
        **kwargs:

    Returns:

    """

    target, transform = get_image_transform(target,transform=transform, bands=[0], window=window,
                                            size_read=size_read)
    target = target[0]
    cmap_preds, norm_preds, patches_preds = get_cmap_norm_colors(configs.COLORS_WORLDFLOODS,
                                                                 INTERPRETATION_WORLDFLOODS)

    ax = rasterioplt.show(target, transform=transform, cmap=cmap_preds, norm=norm_preds,
                          interpolation='nearest', **kwargs)

    if legend:
        ax.legend(handles=patches_preds,
                  loc='upper right')

    return ax

def plot_gt_v2(target: Union[str, np.ndarray], transform:Optional[rasterio.Affine]=None,
               window:Optional[rasterio.windows.Window]=None,
               legend=True, size_read:Optional[int]=None, **kwargs):
    """
    ground truth `target` expected to be 2 channel image with the following code:
        [{0: invalid: 1: land, 2: cloud}, {0:invalid, 1:land, 2: water}]

    We use the invalid values of the land/water mask

    Args:
        target:
        transform:
        window:
        legend:
        size_read:
        **kwargs:

    Returns:

    """

    target, transform = get_image_transform(target,transform=transform, bands=[0, 1], window=window,
                                            size_read=size_read)
    clear_clouds = target[0]
    land_water = target[1]

    v1gt = land_water.copy() # {0: invalid, 1: land, 2: water}
    v1gt[clear_clouds == 2] = 3

    cmap_preds, norm_preds, patches_preds = get_cmap_norm_colors(configs.COLORS_WORLDFLOODS,
                                                                 INTERPRETATION_WORLDFLOODS)

    ax = rasterioplt.show(v1gt, transform=transform, cmap=cmap_preds, norm=norm_preds,
                          interpolation='nearest', **kwargs)

    if legend:
        ax.legend(handles=patches_preds,
                  loc='upper right')

    return ax


def gt_v1_with_permanent_water(gt: np.ndarray, permanent_water: np.ndarray) -> np.ndarray:
    """ Permanent water taken from: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_3_YearlyHistory"""
    gt[(gt == 2) & (permanent_water == 3)] = 4  # set as permanent_water
    gt[(gt == 2) & (permanent_water == 2)] = 5  # set as seasonal water

    return gt


def plot_gt_v1_with_permanent(target: Union[str, np.ndarray], permanent: Optional[Union[str, np.ndarray]]=None,
                              transform:Optional[rasterio.Affine]=None,
                              window:Optional[rasterio.windows.Window]=None, legend=True,
                              size_read:Optional[int]=None,
                              **kwargs):
    bands = [0]
    target, transform = get_image_transform(target, transform=transform, bands=bands, window=window,
                                            size_read=size_read)
    target= target[0]
    if permanent is not None:
        permanent, _ = get_image_transform(permanent, transform=transform, bands=bands, window=window,
                                           size_read=min(target.shape))
        permanent = permanent[0]
        target = gt_v1_with_permanent_water(target, permanent)

    cmap_gt, norm_gt, patches_gt = get_cmap_norm_colors(COLORS_WORLDFLOODS_PERMANENT, INTERPRETATION_WORLDFLOODS_PERMANENT)

    ax = rasterioplt.show(target,transform=transform, cmap=cmap_gt, norm=norm_gt,
                          interpolation='nearest', **kwargs)

    if legend:
        ax.legend(handles=patches_gt,
                  loc='upper right')

    return ax


def plot_s2_and_confusions(input: Union[str, np.ndarray], positives: np.ndarray ,title:Optional[str] = None,
                           transform:Optional[rasterio.Affine]=None, channel_configuration = 'all', **kwargs):
    """
    Plots a S2 image and overlapping FP, FN and TP with masked clouds, computed from 
    compute_positives function

    """
    band_names_current_image = [BANDS_S2[iband] for iband in CHANNELS_CONFIGURATIONS[channel_configuration]]
    bands = [band_names_current_image.index(b) for b in ["B11", "B8", "B4"]]
    image = get_image_transform(input, transform=transform, bands=bands)[0]
    image = np.clip((image-0)/(3000 - 0), 0, 1)
    image = np.moveaxis(image,0,-1)
    
    #set invalids and clouds to black, FP white, FN orange and TP blue
    image[positives == 4] = colors.to_rgb('black')
    image[positives == 1] = colors.to_rgb('C9')
    image[positives == 2] = colors.to_rgb('orange')
    image[positives == 3] = colors.to_rgb('blue')

    cmap_colors = ['orange','C9', 'blue']
    cmap = colors.ListedColormap(cmap_colors)
    
    return rasterioplt.show(np.moveaxis(image,-1,0), cmap = cmap, transform = transform, title = title, **kwargs)
