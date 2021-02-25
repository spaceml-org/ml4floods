BANDS_S2 = ["B1", "B2", "B3", "B4", "B5",
            "B6", "B7", "B8", "B8A", "B9",
            "B10", "B11", "B12", "QA60"]
"""
Map from names in the floodmap to rasterised value of watermask (see function compute_water)

The meaning of the codes are: 
 {0: 'land', 1: 'flood', 2: 'hydro', 3: 'permanent_water_jrc'}

"""

CODES_FLOODMAP = {
    # CopernicusEMS (flood)
    'Flooded area': 1,
    'Not Applicable': 1,
    'Flood trace': 1,
    'Dike breach': 1,
    'Standing water': 1,
    'Erosion': 1,
    'River': 2,
    'Riverine flood': 1,
    # CopernicusEMS (hydro)
    'BH140-River': 2,
    'BH090-Land Subject to Inundation': 2,
    'BH080-Lake': 2,
    'BA040-Open Water': 2,
    # 'BA030-Island': 2, islands are excluded! see filter_land func
    'BH141-River Bank': 2,
    'BH130-Reservoir': 2,
    'BH141-Stream': 2,
    # UNOSAT
    "preflood water": 2,
    # "Flooded area": 1,  # 'flood water' DUPLICATED
    "flood-affected land / possible flood water": 1,
    # "Flood trace": 1,  # 'probable flash flood-affected land' DUPLICATED
    "satellite detected water": 1,
    # "Not Applicable": 1,  # unknown see document DUPLICATED
    "possible saturated, wet soil/ possible flood water": 1,
    "aquaculture (wet rice)": 1,
    "tsunami-affected land": 1,
    "ran of kutch water": 1,
    "maximum flood water extent (cumulative)": 1
}

# Unosat names definition https://docs.google.com/document/d/1i-Fz0o8isGTpRr39HqvUOQBs0yh8_Pz_WcyF5JK0bM0/edit#heading=h.3neqeg3hyp0t
UNOSAT_CLASS_TO_TXT = {
    0: "preflood water",
    1: "Flooded area",  # 'flood water'
    2: "flood-affected land / possible flood water",
    3: "Flood trace",  # 'probable flash flood-affected land'
    4: "satellite detected water",
    5: "Not Applicable",  # unknown see document
    6: "possible saturated, wet soil/ possible flood water",
    9: "aquaculture (wet rice)",
    14: "tsunami-affected land",
    77: "ran of kutch water",
    99: "maximum flood water extent (cumulative)"
}