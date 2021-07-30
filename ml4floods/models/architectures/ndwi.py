import torch

# s2bands (0-based index):
#     green = 2
#     nir = 7
#     swir_cirrus = 10
#     swir11 = 11
#     swir12 = 12


def extract_ndwi(x: torch.Tensor, green:int=2, nir:int=7, epsilon:float=1e-8) -> torch.Tensor:
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

    return ndwi[:, None]

def extract_mndwi(x: torch.Tensor, green:int=2, swir:int=11, epsilon:float=1e-8) -> torch.Tensor:
    """
    Modified normalize difference water index (Xu 2006) Band index defaults to Sentinel-2 image bands.

    Args:
        x: BCHW image
        green: index of green band
        swir: index of swir band (if Sentinel-2 could be indexes 11 or 12)
        epsilon: eps value to avoid dividing by zero

    Returns:
        B1HW image with the nndwi
    """
    band_sum = x[:, green, :, :] + x[:, swir, :, :]
    band_diff = x[:, green, :, :] - x[:, swir, :, :]
    ndwi = band_diff / (band_sum + epsilon)

    return ndwi[:, None]

