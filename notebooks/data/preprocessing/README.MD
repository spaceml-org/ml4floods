# Notebooks for Data Retrieval

## 1. Webscraping the Copernicus Emergency Management System's Rapid (EMSR) Activations
In this notebook we go over how to get started webscraping EMSR codes for flood events in order to
retrieve zip files that contain the following Copernicus Rapid Mapping Products:
- Source Data
- Area of Interest
- Observed Events
- Hydrography
We then download and unzip the files.

## 2. Generating Flood Maps with Copernicus Rapid Mapping Products
In this notebook we go over how to retrieve the associated Rapid Mapping Products
introduced in (1) and store them as a Copernicus EMS metadata dictionary for individual
EMS alert codes. Furthermore, we take the shapefiles/polygons retrieved from the
Rapid Mapping Products and concatenate them over an area of interest to form multipolygons to
be used as a single floodmap layer as opposed to having separate layers for each activation
mapping.

## 3. Sentinel-2 Images from Google Earth Engine
Using the Area of Interest derived from the Rapid Mapping Products for a flood event we then
use the Google Earth Engine Python API to interact with a map and to download intersecting
Sentinel-2 imagery and to superimpose the floodmap derived in (2) for comparison.

## 4. Generate Ground Truth Images
This notebook generates flood labeled images for use in a supervised machine learning pipeline.
