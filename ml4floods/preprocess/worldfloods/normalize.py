from typing import Tuple

import numpy as np

from ml4floods.data.worldfloods.configs import (CHANNELS_CONFIGURATIONS,
                                                SENTINEL2_NORMALIZATION)


def get_normalisation(use_channels: str, channels_first:bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Normalization for the S2 datasets.

    Args:
        use_channels (str): Channels that are to be used.
        channels_first (bool): whether the axis should be first or last

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns the mean and standard deviation.

    Example:
        >>> from ml4floods.preprocess.worldfloods import get_normalization
        >>> use_channels = 'all'
        >>> mu, std = get_normalization(use_channels)    
    """    

    s2_channels = CHANNELS_CONFIGURATIONS[use_channels]

    s2_norm = SENTINEL2_NORMALIZATION[s2_channels]

    #  channel stats for now
    sentinel_means = s2_norm.copy()[:, 0]
    sentinel_std = s2_norm.copy()[:, 1]

    if channels_first:
        sentinel_means = sentinel_means[..., np.newaxis, np.newaxis]  # (nchannels, 1, 1)
        sentinel_std = sentinel_std[..., np.newaxis, np.newaxis]  # (nchannels, 1, 1)
    else:
        sentinel_means = sentinel_means[np.newaxis, np.newaxis]  # (1, 1, nchannels)
        sentinel_std = sentinel_std[np.newaxis, np.newaxis]  # (1, 1, nchannels)

    return sentinel_means, sentinel_std
