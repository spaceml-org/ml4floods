# Scripts for Generating the GroundTruth for WorldFloods Dataset



---
## WorldFloods 1.0



### `generate_ground_truth_wf1.0.py`


> Create Ground Truth from Images in the Bucket

In this case

This will download the following:

* Ground truth for a given Sentinel-2 Image, `.tif`
* Floodmap for the, `.json`



```python
python -u scripts/worldfloods/generate_ground_truth_wf1.0.py 
```


---
## WorldFloods 1.1


---
## WorldFloods 2.0


---
## WorldFloods 2.1


---
## WorldFloods 3.0



---
## Known Issues


### Permanent Water Maps

Loading all of the permanent water maps gives this error. It doesn't appear to change the results very much.

```bash
ERROR:fiona._env:Unable to open EPSG support file gcs.csv.  Try setting the GDAL_DATA environment variable to point to the directory containing EPSG csv files.
```

### Something Crashes...

This happens during the segment when rasterio is saving the GT and uploading to the bucket.

```bash
Segmentation fault (core dumped)
```