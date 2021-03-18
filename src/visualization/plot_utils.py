import rasterio
from rasterio import plot as rasterioplt
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import numpy as np

# SEABORN SETTINGS
import seaborn as sns
sns.set_context(context='talk',font_scale=0.7)


COLORS_WORLDFLOODS_V1_1 = np.array([[0, 0, 0], # invalid
                               [139, 64, 0], # land
                               [0, 0, 139], # water
                               [220, 220, 220]], # cloud
                              dtype=np.float32) / 255


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


def plot_s2_rbg_image(file_path: str, **kwargs):
    
    
    # open image with rasterio
    with rasterio.open(file_path) as src:
        
        # read image
        image = src.read()
        # convert to RBG Image
        rgb = np.clip(image[(3,2,1),...]/3000.,0,1)
        
        # plot image
        rasterioplt.show(rgb, transform=src.transform, **kwargs)
    
    return None

def plot_s2_cloud_prob_image(file_path: str, **kwargs):
    
    
    # open image with rasterio
    with rasterio.open(file_path) as src:
        # assert the image has 15 channels
        assert src.meta["count"] == 15
        
        # plot image
        rasterioplt.show(src.read(15), transform=src.transform, **kwargs)
    
    return None