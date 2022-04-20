import numpy as np


BANDS_S2 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

# 0 based channels based on BANDS_S2
CHANNELS_CONFIGURATIONS = {
    "all": list(range(0,len(BANDS_S2))),
    "rgb": [3, 2, 1],
    "swirnirred": [11, 7, 3],
    "bgr": [1, 2, 3],
    "bgri": [1, 2, 3, 7],
    "riswir" : [3, 7, 11],
    "bgriswir" : [1, 2, 3, 7, 11],
    "bgriswirs" : [1, 2, 3, 7, 11, 12],
    "l89s2": [0, 1, 2, 3, 7, 10, 11, 12], # Same bands as Landsat-7 and Landsat-8
    "sub_20": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
    "hyperscout2": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}

BANDS_L8 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]

# 0 based channels based on BANDS_L8
CHANNELS_CONFIGURATIONS_LANDSAT = {
    "rgb": [3, 2, 1],
    "swirnirred": [5, 4, 3],
    "bgr": [1, 2, 3],
    "bgri": [1, 2, 3, 4],
    "riswir" : [3, 4, 5],
    "bgriswir" : [1, 2, 3, 4, 5],
    "bgriswirs" : [1, 2, 3, 4, 5, 6],
    "l89s2": [0, 1, 2, 3, 4, 7, 5, 6],
}


SENTINEL2_NORMALIZATION = np.array(
    [
        [3787.0604973, 2634.44474043],
        [3758.07467509, 2794.09579088],
        [3238.08247208, 2549.4940614],
        [3418.90147615, 2811.78109878],
        [3450.23315812, 2776.93269704],
        [4030.94700446, 2632.13814197],
        [4164.17468251, 2657.43035126],
        [3981.96268494, 2500.47885249],
        [4226.74862547, 2589.29159887],
        [1868.29658114, 1820.90184704],
        [399.3878948, 761.3640411],
        [2391.66101119, 1500.02533014],
        [1790.32497137, 1241.9817628],
    ],
    dtype=np.float32,
)

COLORS_WORLDFLOODS = np.array([[0, 0, 0],
                               [139, 64, 0],
                               [0, 0, 139],
                               [220, 220, 220]], dtype=np.float32) / 255


COLORS_WORLDFLOODS_INVLANDWATER = COLORS_WORLDFLOODS[[0, 1, 2],...]
COLORS_WORLDFLOODS_INVCLEARCLOUD = COLORS_WORLDFLOODS[[0, 1, 3],...]


CLASS_FREQUENCY_WORLDFLOODSV1 = [0.516942, 0.027322, 0.455787]
CLASS_FREQUENCY_WORLDFLOODSV1_FILTERED = [0.8680312076050476, 0.052033908148693186, 0.07993488424625929]
