import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ml4floods.preprocess.worldfloods import normalize
from ml4floods.data.worldfloods.configs import COLORS_WORLDFLOODS, CHANNELS_CONFIGURATIONS, BANDS_S2


def compute_uncertainties(dataloader, p_pred_fun, d_pred_fun, num_class, config, num_samples=10):
    """
    Run dropout during inference on a dataloader and compute uncertainty metrics for that data, plot union and intersection of 'focus_label' predictions onto an image for visual uncertainty inspection
    
    Args:
    :param dataloader: pytorch Dataloader for test set
    :param pred_fun: function to perform inference using a model
    :param num_class: number of classes
    :param focus_label: name of class we're interested in finding uncertainty around
    :param num_samples: number of samples to take from network to calculate uncertainty from
    :return: None
    """
    for i, batch in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / dataloader.batch_size)):
        test_inputs, ground_truth_outputs = batch['image'], batch['mask'].squeeze(1)
        compute_uncertainties_for_image_pair(test_inputs, ground_truth_outputs, p_pred_fun, d_pred_fun, num_samples, num_class, config)
        
        


def compute_uncertainties_for_image_pair(inputs, targets, p_pred_fun, d_pred_fun, num_samples, num_class, config, denorm=True):
    """
    Runs probabilistic inference on image pair - takes multiple samples from network and computes uncertainty maps, generates a plot of those maps

    :param inputs: Tensor (BxCxHxW)
    :param targets: Tensor (1xCxHxW)
    :param p_pred_fun: probabilistic function - every sample from function varies
    :param d_pred_fun: deterministic function - every sample from function is the same
    :param num_samples: number of samples to take using the probabilistic function
    :param num_class: number of classes in target
    :param config: experiment config
    :return: None
    """
    with torch.no_grad():
        prediction_samples = np.zeros([num_samples] + list(targets.shape))
        prediction_logits = np.zeros([num_samples] + [1, num_class] + list(targets.squeeze().shape))

        for s in range(num_samples):
            sample_logits = p_pred_fun(inputs)

            prediction_logits[s] = sample_logits.cpu().numpy()

            sample_prediction = torch.argmax(sample_logits, dim=1).long()

            prediction_samples[s] = sample_prediction.cpu().numpy()

        samples = prediction_samples
        images = inputs.cpu().numpy()
        gts = targets.cpu().numpy()
        predictions = torch.argmax(d_pred_fun(inputs), dim=1).long().cpu().numpy()

        if denorm:
            optical = s1_to_unnormed_rgb(images, config)[0]
        else:
            optical = s1_to_rgb(images, config)[0]

        gt = mask_to_rgb(gts[0])
        prediction = mask_to_rgb(predictions[0] + 1)
        diff = (predictions[0] - gts[0])

        water_prob = water_probability(samples)
        water_bound = water_bounds(samples)
        water_ent = water_entropy(samples)
        class_var = variance_map(samples)

        fig, ax = plt.subplots(2, 4, figsize=(40, 20))

        # Optical Image
        ax[0, 0].imshow(optical)
        ax[0, 0].set_title('Input', fontweight="bold", size=40)
        # Ground Truth
        ax[0, 1].imshow(gt, interpolation='nearest')
        ax[0, 1].set_title('Ground Truth', fontweight="bold", size=40)
        # Prediction
        ax[0, 2].imshow(prediction, interpolation='nearest')
        ax[0, 2].set_title('Prediction', fontweight="bold", size=40)
        # Diff Map
        ax[0, 3].imshow(diff)
        ax[0, 3].set_title('Difference Map', fontweight="bold", size=40)
        # Uncertainty 1: Water probability
        ax[1, 0].imshow(water_prob, cmap='Blues')
        ax[1, 0].set_title('Water Probability', fontweight="bold", size=40)
        # Uncertainty 2: Water bounds
        ax[1, 1].imshow(water_bound, cmap='Blues')
        ax[1, 1].set_title('Water Bounds', fontweight="bold", size=40)
        # Uncertainty 3: Water Entropy
        ax[1, 2].imshow(water_ent)
        ax[1, 2].set_title('Water Entropy', fontweight="bold", size=40)
        # Uncertainty 4: All Class Variance (TODO: Replace with Mutual Information)
        ax[1, 3].imshow(class_var)
        ax[1, 3].set_title('Class Variance', fontweight="bold", size=40)

        plt.show()


def water_probability(samples):
    water_only = to_binary_segmentation(samples, 1)
    total_image = np.sum(water_only, axis=0)[0]
    return total_image


def water_bounds(samples):
    water_only = to_binary_segmentation(samples, 1)
    total_image = np.sum(water_only, axis=0)

    water_union = total_image.copy()
    water_union[water_union != 0] = 0.5

    water_intersection = total_image.copy()
    water_intersection[water_intersection != len(samples)] = 0
    water_intersection[water_intersection != 0] = 0.5

    water_bounds = water_union - water_intersection

    water_bounds[water_intersection == 0.5] = 1

    return water_bounds[0]


def water_entropy(samples):
    water_only = to_binary_segmentation(samples, 1)
    total_image = np.sum(water_only, axis=0)
    p = (total_image / len(samples))[0]

    entropy_image = np.zeros_like(p)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i, j] != 0:
                entropy_image[i, j] = -(p[i, j] * np.log(p[i, j]))

    return entropy_image


def variance_map(samples):
    var_samples = np.var(samples, axis=0)[0]
    return var_samples


def to_binary_segmentation(samples, class_value):
    # Extract single class maps from each sample
    binary_only = []
    for s in samples:
        binary_image = s.copy()
        binary_image[binary_image != class_value] = 0
        binary_only.append(binary_image)

    binary_only = np.array(binary_only)

    return binary_only


def s1_to_unnormed_rgb(image, config):
    """
    :param image: normalised input image
    :param config: experiment config
    :return: unnormalized rgb image
    """
    mean, std = normalize.get_normalisation("bgr")  # B, R, G!
    mean = mean[np.newaxis]
    std = std[np.newaxis]

    # Find the RGB indexes within the S2 bands
    bands_read_names = [BANDS_S2[i] for i in
                        CHANNELS_CONFIGURATIONS[config["model_params"]["hyperparameters"]['channel_configuration']]]
    bands_index_rgb = [bands_read_names.index(b) for b in ["B4", "B3", "B2"]]

    model_input_rgb_npy = image[:, bands_index_rgb].transpose(0, 2, 3, 1) * std[..., -1::-1] + mean[..., -1::-1]
    model_input_rgb_npy = np.clip(model_input_rgb_npy / 3000., 0., 1.)
    return model_input_rgb_npy


def s1_to_rgb(image, config):

    # Find the RGB indexes within the S2 bands
    bands_read_names = [BANDS_S2[i] for i in CHANNELS_CONFIGURATIONS[config["model_params"]["hyperparameters"]['channel_configuration']]]
    bands_index_rgb = [bands_read_names.index(b) for b in ["B4", "B3", "B2"]]

    model_input_rgb_npy = image[:, bands_index_rgb].transpose(0, 2,3,1) 
    model_input_rgb_npy = np.clip(model_input_rgb_npy / 3000., 0., 1.)
    return model_input_rgb_npy


def mask_to_rgb(mask, values=[0,1,2,3], colors_cmap=COLORS_WORLDFLOODS):
    """
    Given a 2D mask it assign each value of the mask the corresponding color
    :param mask:
    :param values:
    :param colors_cmap:
    :return:
    """
    assert len(values) == len(
        colors_cmap), f"Values and colors should have same length {len(values)} {len(colors_cmap)}"
    mask_return = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    colores = np.array(np.round(colors_cmap * 255), dtype=np.uint8)
    for i, c in enumerate(colores):
        mask_return[mask == values[i], :] = c
    return mask_return
