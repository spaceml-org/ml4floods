from typing import Tuple

import numpy as np

from src.data.worldfloods.configs import (CHANNELS_CONFIGURATIONS,
                                          SENTINEL2_NORMALIZATION)


def get_normalisation(use_channels: str) -> Tuple[np.ndarray, np.ndarray]:
    """Normalization for the S2 datasets.

    Args:
        use_channels (str): Channels that are to be used.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns the mean and standard deviation.

    Example:
        >>> from src.preprocess.worldfloods import get_normalization
        >>> use_channels = 'all'
        >>> mu, std = get_normalization(use_channels)    
    """    

    s2_channels = CHANNELS_CONFIGURATIONS[use_channels]

    s2_norm = SENTINEL2_NORMALIZATION[s2_channels]

    #  channel stats for now
    sentinel_means = s2_norm.copy()[:, 0]
    sentinel_means = sentinel_means[np.newaxis, np.newaxis]
    sentinel_std = s2_norm.copy()[:, 1]
    sentinel_std = sentinel_std[np.newaxis, np.newaxis]

    return sentinel_means, sentinel_std
