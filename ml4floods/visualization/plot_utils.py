import rasterio
from rasterio import plot as rasterioplt
import rasterio.windows
from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np
from typing import Union, Optional, List, Tuple
from ml4floods.data.worldfloods.configs import BANDS_S2, CHANNELS_CONFIGURATIONS
from ml4floods.data.worldfloods import configs
import os


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


def _read_data(input:str,
              bands_rasterio:Optional[List[int]]=None,
              window:Optional[rasterio.windows.Window]=None,
              size_read:Optional[int]=None) -> Tuple[np.ndarray, rasterio.Affine]:

    if size_read is not None:
        if bands_rasterio is None:
            n_bands = 1
        else:
            n_bands = len(bands_rasterio)

        if window is None:
            with rasterio.open(input) as rst:
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

    with rasterio.open(input) as rst:
        output = rst.read(bands_rasterio, window=window, out_shape=out_shape)
        transform = rst.transform if window is None else rasterio.windows.transform(window, rst.transform)

    if input_output_factor is not None:
        transform = rasterio.Affine(transform.a * input_output_factor, transform.b, transform.c,
                                    transform.d, transform.e * input_output_factor, transform.f)

    return output, transform


def plot_s2_cloud_prob_image(file_path: str, size_read:Optional[int]=None, **kwargs):
    # open image with rasterio
    with rasterio.open(file_path) as src:
        # assert the image has 15 channels
        assert src.meta["count"] == 15

    output, transform = _read_data(file_path, bands_rasterio=[15], size_read=size_read)
    output = output[0]
    # plot image
    return rasterioplt.show(output, transform=transform, vmin=0, vmax=100, cmap="gray", **kwargs)


def get_image_transform(input:Union[str, np.ndarray],
                        transform:Optional[rasterio.Affine]=None,
                        bands:List[int]=None,
                        window:Optional[rasterio.windows.Window]=None,
                        size_read:Optional[int]=None) -> Tuple[np.ndarray, rasterio.Affine]:

    if bands is not None:
        bands_rasterio = [b+1 for b in bands]
    else:
        bands_rasterio = None

    if isinstance(input, str):
        return _read_data(input, bands_rasterio, window=window, size_read=size_read)

    if hasattr(input, "cpu"):
        input = input.cpu()

    input = np.array(input)
    if window is None:
        window_slices = (slice(None), slice(None), slice(None))
    else:
        window_slices = (slice(None),) + window.toslices()

    output = input[bands, ...]
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
    (assume the image has values in [0, 10_000] -> https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2 )

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
    (assume the image has values in [0, 10_000] -> https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2 )

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
        prediction:
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
    ground truth `target` expected to be 2 channel image [{0: invalid: 1: land, 2: cloud}, {0:invalid, 1:land, 2: water}]

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


def download_tiff(local_folder: str, tiff_input: str, folder_ground_truth: str,
                  folder_permanent_water: Optional[str] = None) -> str:
    """
    Download a set of tiffs from the google bucket to a local folder

    Args:
        local_folder: local folder to download
        tiff_input: input tiff file
        folder_ground_truth: folder with ground truth images
        folder_permanent_water: folder with permanent water images

    Returns:
        location of tiff_input in the local file system

    """
    import fsspec
    fs = fsspec.filesystem("gs")

    folders = ["/S2/", folder_ground_truth]
    if folder_permanent_water is not None:
        folders.append(folder_permanent_water)

    for folder in folders:
        file_to_download = tiff_input.replace("/S2/", folder)
        if folder.startswith("/"):
            folder = folder[1:]
        folder_iter = os.path.join(local_folder, folder)  # remove /
        file_local = os.path.join(folder_iter, os.path.basename(file_to_download))
        if folder == "S2/":
            return_folder = file_local
        if os.path.exists(file_local):
            continue
        if not fs.exists(file_to_download):
            print(f"WARNING!! file {file_to_download} does not exists")
            continue

        os.makedirs(folder_iter, exist_ok=True)
        fs.get_file(file_to_download, file_local)
        print(f"Downloaded file {file_local}")

    return return_folder