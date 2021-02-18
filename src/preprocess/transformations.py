import cv2
import numpy as np
import torch
from albumentations.augmentations import functional as F
from albumentations.core.composition import BaseCompose
from albumentations.core.transforms_interface import BasicTransform, DualTransform
from albumentations import (
    Compose,
    Flip,
    GaussNoise,
    MotionBlur,
    Normalize,
    PadIfNeeded,
    RandomRotate90,
    ShiftScaleRotate,
)


# TODO: split the ToTensor to ToImageTensor and ToMaskTensor for better clarity.
# TODO: separate functions for the operators (e.g. permute_channels)


def permute_channels(
    image: np.ndarray,
) -> np.ndarray:
    return image.transpose(2, 0, 1)


class ToTensor_NOPE(BasicTransform):
    """Convert image and mask to `torch.Tensor`.

    Multi-channel images will be returned as HxWxC by default, unless `permute_channels` is disabled.

    Masks will be returned with their input shape as a long tensor, unless `convert_one_hot` is enabled.

    Args:
        num_classes (int): only for segmentation
        convert_one_hot (bool): convert input labels to one-hot, only for segmentation, default False
        permute_channels (bool): whether to attempt to convert the image to HxWxC

    """

    def __init__(
        self,
        num_classes: int = 1,
        convert_one_hot: bool = False,
        permute_channels: bool = True,
    ):
        super(ToTensor, self).__init__(always_apply=True, p=1.0)
        self.num_classes = num_classes
        self.convert_one_hot = convert_one_hot
        self.permute_channels = permute_channels

    def __call__(self, force_apply=True, **kwargs):
        # Convert image to tensor
        kwargs.update(
            {"image": self._image_to_tensor(kwargs["image"].astype(np.float32))}
        )

        # Convert mask to tensor:
        if "mask" in kwargs.keys():
            kwargs.update({"mask": self._mask_to_tensor(kwargs["mask"])})

        for k, v in kwargs.items():
            if self._additional_targets.get(k) == "image":
                kwargs.update({k: self._image_to_tensor(kwargs[k])})
            if self._additional_targets.get(k) == "mask":
                kwargs.update({k: self._mask_to_tensor(kwargs[k])})
        return kwargs

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert input image to a tensor and permute colour channel if appropriate.

        Args:
            image (np.ndarray): image to be converted to a tensor
            If the size is
        """

        # Only permute if multi-channel
        if image.ndim > 2 and self.permute_channels:
            image = image.transpose(2, 0, 1)

        return torch.from_numpy(image)

    def _mask_to_tensor(self, mask):
        """
        Convert input mask into a long tensor.
        """
        mask_tensor = torch.from_numpy(mask).long()

        if self.num_classes > 1 and self.convert_one_hot:
            assert mask_tensor.ndim == 2
            mask_tensor = torch.nn.functional.one_hot(
                mask_tensor, self.num_classes
            ).permute(2, 0, 1)

        return mask_tensor

    @property
    def targets(self):
        raise NotImplementedError

    def get_transform_init_args_names(self):
        return ("num_classes", "convert_one_hot")


class ToTensor(BasicTransform):
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True, p=1.0)
        
    def __call__(self, input_data: dict, force_apply=True) -> dict:
        # Convert image to tensor
        input_data['image'] = self._image_to_tensor(input_data['image'])
        input_data['mask'] = self._mask_to_tensor(input_data['mask'])

        return input_data

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
        
    def __call__(self, input_data: dict, force_apply=True) -> dict:
        input_data["image"] = self._permute_channels(input_data["image"])
        input_data["mask"] = self._permute_channels(input_data["mask"])

        return input_data

    def _permute_channels(self, input_image: np.ndarray) -> np.ndarray:
        if input_image.ndim == 3:
            input_image = input_image.transpose(2, 0, 1)
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
        
    def __call__(self, input_data: dict, force_apply=True) -> dict:
        input_data["mask"] = self._one_hot_encode(input_data["mask"])

        return input_data

    def _one_hot_encode(self, input_mask: torch.Tensor) -> torch.Tensor:
        if self.num_classes > 1:
            input_mask = torch.nn.functional.one_hot(
                input_mask, self.num_classes
            )
        
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

    def __call__(self, input_data: dict, force_apply=False) -> dict:
        image = data["image"]

        # Mono images
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)

        if self.channels is None:
            self.channels = range(image.shape[2])

        for c in self.channels:
            for t in self.transforms:
                image[:, :, c] = t(image=image[:, :, c])["image"]

        data["image"] = image

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

    def __call__(self, input_data: dict, interpolation=cv2.INTER_LINEAR, **params) -> dict:
        image = input_data["image"]
        new_size = np.round(np.array(image.shape[:2]) / self.downsampling_factor).astype(
            np.int64
        )
        image = F.resize(
            image, height=new_size[0], width=new_size[1], interpolation=interpolation
        )

        input_data["image"] = image

        return input_data


def get_augmentation(
    channel_mean,
    channel_std,
    input_size=256,
    downsampling_factor=1,
    augment=True,
    normalize=True,
):
    # Pad to a square

    transform_list = []
    if augment and (input_size > 0):
        transform_list.append(PadIfNeeded(input_size, input_size))

    # Downsample if required
    if downsampling_factor > 1.01:
        transform_list.append(
            ResizeFactor(downsampling_factor=downsampling_factor, always_apply=True)
        )

    if normalize:
        transform_list.append(
            Normalize(mean=channel_mean, std=channel_std, max_pixel_value=1.0)
        )

    # Other augmentation
    if augment:
        channel_jitter = ShiftScaleRotate(
            shift_limit=0.001, scale_limit=0.01, rotate_limit=0.01
        )

        transform_list += [
            PerChannel([channel_jitter]),
            GaussNoise(var_limit=(1e-6, 1e-3), p=0.2),
            MotionBlur(3),
            Flip(),
            RandomRotate90(),
        ]

    # if drop_channels:
    #    transform_list.append(ChannelDropout(channel_drop_range=(1, 3)))

    transform_list.append(ToTensor())

    return Compose(transform_list)


def get_augmentation_train(
    channel_mean,
    channel_std,
    input_size=256,
    downsampling_factor=1,
    augment=True,
    normalize=True,
):
    """
    Get transformation for train time, by default augmentation is applied.

    Images are initially converted to a square with sides of length `input_size`. If
    `downsampling_factor > 1, then the image is resized.

    The final result is returned as a normalised tensor.
    """
    return get_augmentation(
        channel_mean, channel_std, input_size, downsampling_factor, augment, normalize
    )


def get_augmentation_test(
    channel_mean, channel_std, input_size=256, downsampling_factor=1, normalize=True
):
    """
    Get transformation for test time. Augmentation is not applied.

    Images are initially converted to a square with sides of length `input_size`. If
    `downsampling_factor > 1, then the image is resized.

    The final result is returned as a normalised tensor.
    """
    return get_augmentation(
        channel_mean,
        channel_std,
        input_size,
        downsampling_factor,
        augment=False,
        normalize=normalize,
    )
