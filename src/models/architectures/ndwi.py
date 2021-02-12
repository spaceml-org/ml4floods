import torch
from torch import nn

# s2bands:
#     green = 2
#     nir = 7
#     swir_cirrus = 10
#     swir11 = 11
#     swir12 = 12


def extract_ndwi(x, green=2, nir=7, epsilon=1e-8):
    """ Mcfeeters 1996 NDWI """
    band_sum = x[:, green, :, :] + x[:, nir, :, :]
    band_diff = x[:, green, :, :] - x[:, nir, :, :]
    ndwi = band_diff / (band_sum + epsilon)
    # Set all invalids to -1 (perfect masking)
    invalids = torch.all(x <= 0, dim=1)
    ndwi[invalids] = -1

    return ndwi


# def extract_ndwi_binary(x, green=2, nir=7, epsilon=1e-8, threshold=0):
#     return (extract_ndwi(x, green, nir, epsilon) >= threshold).float()


class ManualWaterBodyNDWI(nn.Module):
    def __init__(self, threshold_water=0., to_probability=False):
        super().__init__()
        self.threshold_water = threshold_water
        self.to_probability = to_probability

    def forward(self, x, gt):
        # ground_truth_outputs expected {0: invalid, 1: land, 2: water, 3: cloud}
        assert gt.dim() == 3, "Expected gt 3 dims"
        assert x.dim() == 4, "Expected x 4 dims"
        # Covert to ToA (would not be needed since 10000 factors out in normalized diff)
        invalids = (gt == 0)

        ndwi = extract_ndwi(x / 10000.)
        assert ndwi.dim() == 3, "Expected computed ndwi 3 dims"

        # Convert ndwi to probability with the selected threshold
        if self.to_probability:
            ndwi_prob = torch.sigmoid(ndwi - self.threshold_water)
            ndwi_prob_land = (1 - ndwi_prob)
        else:
            ndwi_prob = ndwi
            ndwi_prob_land = 0

        valids = (~invalids).float()
        is_cloud = (gt == 3).float().to(ndwi.device)

        # Generate probability outputs
        test_outputs = torch.zeros((x.shape[0], 3,) + x.shape[2:], dtype=torch.float32, device=ndwi.device)
        test_outputs[:, 2] = valids * is_cloud
        not_cloud = (1 - is_cloud)
        test_outputs[:, 1] = valids * not_cloud * ndwi_prob
        test_outputs[:, 0] = valids * not_cloud * ndwi_prob_land + invalids.float()

        return test_outputs
