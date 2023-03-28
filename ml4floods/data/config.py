BANDS_S2_WITH_QA60 = ["B1", "B2", "B3", "B4", "B5",
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
    'Previous flooded area': 1,
    'Not Applicable': 1,
    'Not Application': 1,
    'Flood trace': 1,
    'Dike breach': 1,
    'Standing water': 1,
    'Erosion': 1,
    'River': 2,
    'Riverine flood': 1,
    # CopernicusEMS (hydro)
    'BH140-River': 2,
    'BH090-Land Subject to Inundation': 0,
    'BH080-Lake': 2,
    'BA040-Open Water': 2,
    'BA030-Island': 0, # islands are excluded! see filter_land func
    'BH141-River Bank': 2,
    'BH170-Natural Spring': 2,
    'BH130-Reservoir': 2,
    'BH141-Stream': 2,
    'BA010-Coastline': 0,
    'BH180-Waterfall': 2,
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
    "maximum flood water extent (cumulative)": 1,
    # Preds
    "flood-trace": 1,
    "water": 1,
    "flood_trace": 1
}

CLASS_LAND_COPERNICUSEMSHYDRO = ["BH090-Land Subject to Inundation", "BA030-Island", 'BA010-Coastline']

ACCEPTED_FIELDS = list(CODES_FLOODMAP.keys()) + CLASS_LAND_COPERNICUSEMSHYDRO


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

RENAME_SATELLITE = {"landsat 5": "Landsat-5",
                    "landsat 8": "Landsat-8",
                    "landsat-8": "Landsat-8",
                    "landsat 7": "Landsat-7",
                    "sentinel-1": "Sentinel-1",
                    "pleadies": "Pleiades-1A-1B",
                    "sentinel-2": "Sentinel-2",
                    "radarsat-1": "RADARSAT-1",
                    "radarsat-2": "RADARSAT-2",
                    "terrasar-x": "TERRASAR-X",
                    'spot 6' : "SPOT-6-7",
                    "worldview-2": "WorldView-2"}

SATELLITE_TYPE = {'COSMO-SkyMed': "SAR",
                  'GeoEye-1': "SAR",
                  'Landsat-5': "Optical",
                  'Landsat-7': "Optical",
                  'Landsat-8': "Optical",
                  'PlanetScope': "Optical",
                  'Pleiades-1A-1B': "Optical",
                  'RADARSAT-1': "SAR",
                  'RADARSAT-2': "SAR",
                  'SPOT-6-7': "Optical",
                  'Sentinel-1': "SAR",
                  'Sentinel-2': "Optical",
                  'TERRASAR-X': "SAR",
                  'WorldView-1': "Optical",
                  'WorldView-2': "Optical",
                  'WorldView-3': "Optical",
                  'alos palsar': "SAR",
                  'asar imp': "SAR",
                  'dmc': "Optical",
                  'earth observing 1': "Optical",
                  'modis-aqua': "Optical",
                  'modis-terra': "Optical",
                  'spot4': "Optical"}