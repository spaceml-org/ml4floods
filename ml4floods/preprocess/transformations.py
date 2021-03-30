from typing import Dict, Tuple

import albumentations
import cv2
import numpy as np
import torch
from albumentations import (Compose, Flip, GaussNoise, MotionBlur, Normalize,
                            PadIfNeeded, RandomRotate90, ShiftScaleRotate)
from albumentations.augmentations import functional as F
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import (BasicTransform,
                                                      DualTransform)

from ml4floods.preprocess.worldfloods.normalize import get_normalisation

# TODO: split the ToTensor to ToImageTensor and ToMaskTensor for better clarity.
# TODO: separate functions for the operators (e.g. permute_channels)


def permute_channels(
    image: np.ndarray,
) -> np.ndarray:
    return image.transpose(2, 0, 1)


class ToTensor(BasicTransform):
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True, p=1.0)

    # def __call__(self, input_data: dict, force_apply=True) -> dict:
    def __call__(self, force_apply=True, **data) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert image to tensor
        image, mask = data["image"], data["mask"]

        # convert to tensor
        image = self._image_to_tensor(image)
        mask = self._mask_to_tensor(mask)

        # create output dictionary
        data["image"], data["mask"] = image, mask
        return data

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image)

    def _mask_to_tensor(self, mask):
        mask_tensor = torch.from_numpy(mask).long()
        return mask_tensor

    @property
    def targets(self):
        raise NotImplementedError


class PermuteChannels(BasicTransform):
    def __init__(self):
        super(PermuteChannels, self).__init__(always_apply=True, p=1.0)

    def __call__(self, force_apply=True, **data) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert image to tensor
        image, mask = data["image"], data["mask"]

        image = self._permute_channels(image)
        mask = self._permute_channels(mask)

        # create output dictionary
        data["image"], data["mask"] = image, mask
        return data

    def _permute_channels(self, input_image: np.ndarray) -> np.ndarray:
        if input_image.ndim == 3:
            input_image = input_image.transpose(2, 0, 1)
        return input_image

    @property
    def targets(self):
        raise NotImplementedError


class InversePermuteChannels(BasicTransform):
    def __init__(
        self,
    ):
        super(InversePermuteChannels, self).__init__(always_apply=True, p=1.0)

    def __call__(self, force_apply=True, **data) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert image to tensor
        image, mask = data["image"], data["mask"]

        image = self._inverse_permute_channels(image)
        mask = self._inverse_permute_channels(mask)

        # create output dictionary
        data["image"], data["mask"] = image, mask
        return data

    def _inverse_permute_channels(self, input_image: np.ndarray) -> np.ndarray:
        if input_image.ndim == 3:
            input_image = input_image.transpose(1, 2, 0)
        return input_image

    @property
    def targets(self):
        raise NotImplementedError


class OneHotEncoding(BasicTransform):
    """One hot encode the input ground truth labels.
    TODO: Make sure that we are getting the output shape as desired
        - current output shape = C x H x W x num_classes
    """

    def __init__(self, num_classes: int):
        super(OneHotEncoding, self).__init__(always_apply=True, p=1.0)
        self.num_classes = num_classes

    def __call__(self, force_apply=True, **data) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert image to tensor
        image, mask = data["image"], data["mask"]

        mask = self._one_hot_encode(mask)

        # create output dictionary
        data["image"], data["mask"] = image, mask
        return data

    def _one_hot_encode(self, input_mask: torch.Tensor) -> torch.Tensor:
        if self.num_classes > 1:
            input_mask = torch.nn.functional.one_hot(input_mask, self.num_classes)

        return input_mask

    @property
    def targets(self):
        raise NotImplementedError


# IGNORE FOR NOW
class PerChannel(BaseCompose):
    """Apply transformations per-channel

    Args:
        transforms (list): list of transformations to compose.
        channels (list): channels to apply the transform to. Pass None to apply to all.
                         Default: None (apply to all)
        p (float): probability of applying the transform. Default: 0.5.
    """

    def __init__(self, transforms, channels=None, p=0.5):
        super(PerChannel, self).__init__(transforms, p)
        self.transforms = transforms
        self.channels = channels

    def __call__(self, force_apply=True, **data) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert image to tensor
        image, mask = data["image"], data["mask"]
        print(image.shape)
        # Mono images
        if image.ndim == 2:
            image = np.expand_dims(image, 0)

        if self.channels is None:
            self.channels = range(image.shape[-1])

        for ichannel in self.channels:
            for itransform in self.transforms:
                # get result (dictionary output)

                res = itransform(image=image[:, :, ichannel])
                # augment the channel of the dictionary
                image[:, :, ichannel] = res["image"]

        # create output dictionary
        data["image"], data["mask"] = image, mask
        return data


class ResizeFactor(DualTransform):
    """Resize the input by the given factor (assume input (rows,cols,channels) or (rows,cols)

    Args:
        p (float): probability of applying the transform. Default: 1.
        downsampling_factor (int): desired height of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        downsampling_factor,
        interpolation=cv2.INTER_LINEAR,
        always_apply=True,
        p=1,
    ):
        super(ResizeFactor, self).__init__(always_apply, p)
        self.downsampling_factor = downsampling_factor
        self.interpolation = interpolation

    def __call__(self, force_apply=True, **data) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = data["image"], data["mask"]

        new_size = np.round(
            np.array(image.shape[:2]) / self.downsampling_factor
        ).astype(np.int64)
        image = F.resize(
            image,
            height=new_size[0],
            width=new_size[1],
            interpolation=self.interpolation,
        )
        # create output dictionary
        data["image"], data["mask"] = image, mask
        return data


def transforms_generator(config: Dict) -> albumentations.Compose:
    """Function to create a transformation composition using the parameters of a config file.
    This motive of this function is to provide an ability to modify transformations without changing the code.

    Args:
        config (Dict): Configuration file read a dictionary

    Returns:
        albumentations.Compose: Returns the composition of all the transformations mentioned in the config file.
    """
    # initialize list of transformations
    list_of_transforms = []
    list_of_transforms += [
        InversePermuteChannels(),
    ]
    # populate arguments
    if config["resizefactor"] is not None:
        transform_resize = ResizeFactor(**config["resizefactor"])
        list_of_transforms += [
            transform_resize,
        ]

    if config["use_channels"] is not None:
        channel_mean, channel_std = get_normalisation(config["use_channels"])
        transform_normalize = Normalize(
            mean=channel_mean, std=channel_std, max_pixel_value=1.0
        )
        list_of_transforms += [
            transform_normalize,
        ]

    if config["gaussnoise"] is not None:
        transform_gaussnoise = GaussNoise(
            var_limit=(
                config["gaussnoise"]["var_limit_lower"],
                config["gaussnoise"]["var_limit_upper"],
            ),
            p=config["gaussnoise"]["p"],
        )
        list_of_transforms += [transform_gaussnoise]

    if config["motionblur"] is not None:
        transform_motionblue = MotionBlur(**config["motionblur"])
        list_of_transforms += [transform_motionblue]

    list_of_transforms += [
        PermuteChannels(),
    ]

    if config["totensor"] is True:
        list_of_transforms += [ToTensor()]

#     if config["num_classes"] > 1:
#         list_of_transforms += [OneHotEncoding(num_classes=3)]

    return albumentations.Compose(list_of_transforms)
