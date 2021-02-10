import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from termcolor import colored
from matplotlib import colors
from data.worldfloods_dataset import SENTINEL2_NORMALIZATION
import numpy as np
import os
from PIL import Image


def categorical_to_channels(categorical, num_categories=None):
    if categorical.dim() != 2:
        raise ValueError('Expecting a categorical with shape (HxW), received shape {}'.format(categorical.shape))
    if num_categories is None:
        num_categories = int(torch.max(categorical)) + 1

    return torch.nn.functional.one_hot(categorical.long(), num_categories).permute(2, 0, 1)


def mask_to_rgb(mask, values, colors_cmap):
    """
    Given a 2D mask it assign each value of the mask the corresponding color
    :param mask:
    :param values:
    :param colors_cmap:
    :return:
    """
    assert len(values) == len(colors_cmap), "Values and colors should have same length {} {}".format(len(values), len(colors_cmap))

    mask_return = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    colores = np.array(np.round(colors_cmap*255), dtype=np.uint8)
    for i, c in enumerate(colores):
        mask_return[mask == values[i], :] = c

    return mask_return


def plot_prediction(model_input, ground_truth_output_categorical, file_name, model_output=None, colors_cmap=None,
                    figsize=None, axis="off", plot_independent=True,
                    plot_probabilities_and_channels=True, normalized=True, num_ground_truth_categories=18):
    if model_input.dim() != 3:
        raise ValueError('Expecting model_input with shape (CxHxW)')
    if (model_output is not None) and (model_output.dim() != 3):
        raise ValueError('Expecting model_output with shape (CxHxW)')
    if ground_truth_output_categorical.dim() != 2:
        raise ValueError('Expecting ground_truth_output with shape (HxW)')
    num_model_input_channels = model_input.shape[0]

    if model_output is not None:
        assert not plot_probabilities_and_channels, "Not prepared for plot probs and model output being none"
        num_model_output_channels = model_output.shape[0]
        if num_model_output_channels != num_ground_truth_categories:
            print(colored(
                'Warning: model_output channels ({}) and num_ground_truth_categories channels ({}) are not the same'.format(
                    num_model_output_channels, num_ground_truth_categories), 'red', attrs=['bold']))
        cols = 3
    else:
        cols = 2

    if plot_probabilities_and_channels:
        rows = max(num_model_input_channels, num_model_output_channels, num_ground_truth_categories) + 1
    else:
        rows = 1

    if figsize is None:
        if plot_probabilities_and_channels:
            figsize = (5, 35)
        elif model_input.shape[1] == model_input.shape[2]:
            figsize = (15, 5)
        else:
            aspect_ratio = model_input.shape[1] / model_input.shape[2]
            if model_input.shape[1] < model_input.shape[2]:
                figsize = [7*cols, round(7 * aspect_ratio)]
            else:
                figsize = [round(7*cols / aspect_ratio), 7]
            figsize[0] = max(figsize[0],3)
            figsize[1] = max(figsize[1],3)

    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        ax = ax[np.newaxis]

    model_input_npy = model_input.cpu().numpy()
    if normalized:
        norm_facts = SENTINEL2_NORMALIZATION.copy()
        norm_facts = norm_facts[..., np.newaxis, np.newaxis]
        std = norm_facts[3:0:-1, 1]
        mean = norm_facts[3:0:-1, 0]
        model_input_rgb = model_input_npy[3:0:-1] * std + mean
    else:
        model_input_rgb = model_input_npy[3:0:-1]

    model_input_rgb_npy = np.clip(model_input_rgb / 2200, 0., 1.).transpose((1, 2, 0))

    invalid = (model_input_rgb[3:0:-1] <= 0).all(axis=0)
    if colors_cmap is not None:
        zero_color = np.array([[0, 0, 0]], dtype=np.float32)
        colors_cmap = np.concatenate([zero_color, colors_cmap], axis=0)
    vmin = -1

    if colors_cmap is not None:
        cmap_categorical = colors.ListedColormap(colors_cmap)
    else:
        cmap_categorical = "jet_r"

    norm_categorical = colors.Normalize(vmin=vmin,
                                        vmax=(num_ground_truth_categories - 1))

    ax[0, 0].imshow(model_input_rgb_npy)
    ax[0, 0].axis(axis)
    ax[0, 0].set_ylabel('Combined')
    ax[0, 0].set_title('Input')

    if plot_independent:
        filename, ext = os.path.splitext(file_name)
        filename += "_rgb"+ext
        model_input_rgb_npy = (model_input_rgb_npy * 255).round().astype(np.uint8)
        Image.fromarray(model_input_rgb_npy, mode="RGB").save(filename)

    _ix = 1
    if model_output is not None:
        model_output_category = torch.argmax(model_output, dim=0).detach().cpu().numpy()

        model_output_category[invalid] = -1

        ax[0, _ix].imshow(model_output_category,
                          cmap=cmap_categorical, norm=norm_categorical)
        ax[0, _ix].axis(axis)
        ax[0, _ix].set_title('Prediction')

        if plot_independent:
            rgbimg = mask_to_rgb(model_output_category, np.arange(-1, num_ground_truth_categories), colors_cmap)
            filename, ext = os.path.splitext(file_name)
            filename += "_pred" + ext
            Image.fromarray(rgbimg, mode="RGB").save(filename)

        _ix += 1

    ground_truth_output_category = ground_truth_output_categorical.detach().cpu().numpy()
    ground_truth_output_category[invalid] = -1
    ax[0, _ix].imshow(ground_truth_output_category,
                      cmap=cmap_categorical, norm=norm_categorical)
    ax[0, _ix].axis(axis)
    ax[0, _ix].set_title('Ground truth')

    if plot_independent:
        rgbimg = mask_to_rgb(ground_truth_output_category, np.arange(-1, num_ground_truth_categories), colors_cmap)
        filename, ext = os.path.splitext(file_name)
        filename += "_gt" + ext
        Image.fromarray(rgbimg, mode="RGB").save(filename)

    if plot_probabilities_and_channels:
        for c in range(1, rows):
            if c < (num_model_input_channels + 1):
                ax[c, 0].imshow(model_input_npy[c - 1], cmap='Greys')
                ax[c, 0].axis(axis)
                ax[c, 0].set_ylabel('Ch. {}'.format(c - 1))
            else:
                fig.delaxes(ax[c, 0])
        for c in range(1, rows):
            if c < (num_model_output_channels + 1):
                ax[c, 1].imshow(model_output[c - 1].detach().cpu().numpy(), cmap="Greys", vmin=0, vmax=1)
                ax[c, 1].axis(axis)
            else:
                fig.delaxes(ax[c, 1])
        for c in range(1, rows):
            ground_truth_output = categorical_to_channels(ground_truth_output_categorical,
                                                          num_ground_truth_categories)
            if c < (num_ground_truth_categories + 1):
                ax[c, 2].imshow(ground_truth_output[c - 1].detach().cpu().numpy(), cmap='Greys',
                                vmin=0, vmax=1)
                ax[c, 2].axis(axis)
            else:
                fig.delaxes(ax[c, 2])

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()


def colors_sen12ms(N, base_cmap="jet_r"):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))

    # Woodlands
    color_list[0] = [0.14509804, 0.34117647, 0.04705882, 1]
    color_list[1] = [0.14509804, 0.34117647, 0.04705882, 1]
    color_list[2] = [0.14509804, 0.34117647, 0.04705882, 1]
    color_list[3] = [0.14509804, 0.34117647, 0.04705882, 1]
    color_list[4] = [0.14509804, 0.34117647, 0.04705882, 1]

    color_list[6] = [1, 0.702, 0, 1]  # shrub
    color_list[7] = [1, 0.702, 0, 1]  # shrub

    color_list[9] = [1., 0.97254902, 0.45098039, 1]  # woody savannah
    color_list[8] = [0.10588235, 0.50980392, 0.14901961, 1]  # savannah

    color_list[10] = [0.83137255, 1., 0.76078431, 1]  # grassland
    color_list[11] = [0.55294118, 0.94117647, 0.8, 1]  # wetland

    color_list[12] = [0.54509804, 1, 0.52156863, 1]  # Cropland
    color_list[13] = [0.4, 0.4, 0.4, 1]  # Urban
    color_list[14] = [0.54509804, 1, 0.52156863, 1]  # Cropland

    color_list[15] = [0.6784, 0.9137, 1, 1]  # Ice
    color_list[16] = [0.3, 0.19, 0.0078, 1]  # Barren
    color_list[17] = [0, 0.4156, 0.7607, 1]  # Water

    # cmap_name = base.name + str(N)
    return color_list


from jinja2 import Environment
web_page ="""
<html>
<head><title>Test data.</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<meta name="Description" content="Test images"/>
</head>
<body>
<h1>Test images</h1>

{% for it in imgs_show %}
<div><p>{{it["idx"]}}: {{it["filename"]}}</p><img src='{{it["src"]}}'/></div>
{% endfor %}
</body>
</html>

"""


def generate_html(idx_showed_dataset, dataset,filenames_generated, name_html):
    imgs_show = []
    for i in range(len(filenames_generated)):
        src = os.path.basename(filenames_generated[i])
        idx = idx_showed_dataset[i]
        filename = dataset.x_filenames[idx]
        imgs_show.append({"src":src,
                          "idx": idx,
                          "filename":filename})

    with open(name_html, "w") as s:
        Environment().from_string(web_page).stream(imgs_show=imgs_show).dump(s)


