from ml4floods.data.worldfloods.dataset import WorldFloodsDatasetTiled
from ml4floods.preprocess.tiling import WindowSlices
from typing import List, Callable
from tqdm import trange
import numpy as np

def _filter_windows(fun_frac_clouds_invalids:Callable,
                    dataset:WorldFloodsDatasetTiled, threshold_clouds:float=.5) ->List[WindowSlices]:
    """ filter windows from the dataset with more tham threshold_clouds * 100 of clouds or invalids """

    valid_slices = []
    for idx in trange(len(dataset), desc="Filtering invalid and cloudy windows"):
        label = dataset.get_label(idx)
        frac_invalids = fun_frac_clouds_invalids(label)
        if frac_invalids < threshold_clouds:
            valid_slices.append(dataset.list_of_windows[idx])

    return valid_slices


def filter_windows_v1(dataset: WorldFloodsDatasetTiled, threshold_clouds: float = .5) -> List[WindowSlices]:
    """ filter windows from the dataset with more that threshold_clouds * 100 of clouds or invalids """

    return _filter_windows(lambda label: ((label == 0) | (label == 3)).sum()/np.prod(label.shape),
                           dataset, threshold_clouds=threshold_clouds)


def filter_windows_v2(dataset: WorldFloodsDatasetTiled, threshold_clouds: float = .5) -> List[WindowSlices]:
    """ filter windows from the dataset with more that threshold_clouds * 100 of clouds or invalids """

    # Assumes first channel is cloud second channel is water
    return _filter_windows(lambda label: (label[1] == 0).sum() / np.prod(label.shape[1:]),
                           dataset, threshold_clouds=threshold_clouds)



