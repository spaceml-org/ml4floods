import torch

# s2bands (0-based index):
#     green = 2
#     nir = 7
#     swir_cirrus = 10
#     swir11 = 11
#     swir12 = 12


def extract_ndwi(x: torch.Tensor, green=2, nir=7, epsilon=1e-8) -> torch.Tensor:
    """
    Normalize difference water index (Mcfeeters 1996) Band index defaults to Sentinel-2 image bands.

    Args:
        x: BCHW image
        green: index of green band
        nir: index of nir band
        epsilon: eps value to avoid dividing by zero

    Returns:
        B1HW image with the ndwi
    """
    band_sum = x[:, green, :, :] + x[:, nir, :, :]
    band_diff = x[:, green, :, :] - x[:, nir, :, :]
    ndwi = band_diff / (band_sum + epsilon)

    return ndwi

