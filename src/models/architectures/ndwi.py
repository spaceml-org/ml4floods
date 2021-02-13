import torch


# s2bands (0-based index):
#     green = 2
#     nir = 7
#     swir_cirrus = 10
#     swir11 = 11
#     swir12 = 12


def extract_ndwi(x: torch.Tensor, green=2, nir=7, epsilon=1e-8) -> torch.Tensor:
    """
    Normalize difference water index (Mcfeeters 1996)

    :param x: BCHW image
    :param green: index of green band
    :param nir: index of nir band
    :param epsilon: eps value to avoid dividing by zero

    :return: B1HW image with the ndwi
    """
    band_sum = x[:, green, :, :] + x[:, nir, :, :]
    band_diff = x[:, green, :, :] - x[:, nir, :, :]
    ndwi = band_diff / (band_sum + epsilon)

    return ndwi

