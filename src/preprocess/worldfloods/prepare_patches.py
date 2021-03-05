from src.data.worldfloods.dataset import WorldFloodsDatasetTiled
from src.preprocess.tiling import WindowSlices
from typing import List, Callable
from tqdm import trange
import numpy as np

def _filter_windows(fun_clouds_invalids:Callable,
                    dataset:WorldFloodsDatasetTiled, threshold_clouds:float=.5) ->List[WindowSlices]:
    """ filter windows from the dataset with more tham threshold_clouds * 100 of clouds or invalids """

    valid_slices = []
    for idx in trange(len(dataset), desc="Filtering invalid and cloudy windows"):
        label = dataset.get_label(idx)
        n_invalids = np.sum(fun_clouds_invalids(label))
        total_n_pixels = np.prod(label.shape)
        if n_invalids/total_n_pixels < threshold_clouds:
            valid_slices.append(dataset.list_of_windows[idx])

    return valid_slices


def filter_windows_v1(dataset: WorldFloodsDatasetTiled, threshold_clouds: float = .5) -> List[WindowSlices]:
    """ filter windows from the dataset with more that threshold_clouds * 100 of clouds or invalids """

    return _filter_windows(lambda label: (label == 0) | (label == 3),
                           dataset, threshold_clouds=threshold_clouds)


def filter_windows_v2(dataset: WorldFloodsDatasetTiled, threshold_clouds: float = .5) -> List[WindowSlices]:
    """ filter windows from the dataset with more that threshold_clouds * 100 of clouds or invalids """

    # Assumes first channel is water second channel is cloud
    return _filter_windows(lambda label: (label[0] == 0) | (label[1] == 2),
                           dataset, threshold_clouds=threshold_clouds)



