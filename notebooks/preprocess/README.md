# Notebooks

---
## 1 - PyTorch Dataset

This is for tiling your images first, saving and then loading as train/test/val split.


**Source**: [`1.0_demo_pytorch_tiling.ipynb`](1.0_demo_pytorch_tiling.ipynb)

#### Supplementary

For more details about how to do the tiling from scratch using rasterio, please see notebooks within the `tiling` folder.


---

## 2 - PyTorch Dataset + Transforms

How to use the Albumentations library when defining the transformations within the PyTorch Dataset.


**Source**: [`2.0_demo_pytorch_transforms.ipynb`](2.0_demo_pytorch_transforms.ipynb)

#### Supplementary

For more details about the albumentations library or potentially other libraries, please see notebooks within the `transformations` folder.


---

## 3 - PyTorch Lightning Data Module

In this series of demo notebooks, we demonstrate how to use the PyTorch-Lightning DataModule which abstracts many processes to streamline the data generation process. We showcase how this will allow the user flexibility but also allow us to provide config files with set parameters to enable reproducibility.

**Notebook I**: [`3.0_demo_pl_datamodule.ipynb`](3.0_demo_pl_datamodule.ipynb)

> This notebook is a simple version that takes in local files. A demo download function is provided for demonstration purposes.

**Notebook II**: [`3.1_demo_pl_datamodule_gcp.ipynb`](3.1_demo_pl_datamodule_gcp.ipynb)

> This notebook reads files directly from the GCP bucket.


---

## 4 - Data Pipeline

In this notebook, we generalize the above notebooks to provide an all inclusive pipeline showcase how we go from data to dataloader for the WorldFloods dataset. This features a configuration file which will drive all subsequent steps including the downloading (optional), the tiling (optional), the dataset and ML-ready dataloader.

**Notebook**: [`4.0_demo_preprocess_pipeline.ipynb`](4.0_demo_preprocess_pipeline.ipynb)



---

## 5 - GroundTruth Generation

In this notebook, we showcase how one can generate groundtruth given a Sentinel-2 Image and some floodmap meta data that we have preprocessed within the bucket. We demonstrate the two types of groundtruth that we have available: 

* 3-Class: water, land, cloud
* Binary:
  * Water/Land
  * Cloud/No Cloud

**Notebook**: [`demo_S2_floodmap_gt.ipynb`](./groundtruth/demo_S2_floodmap_gt.ipynb)


> This notebook reads files directly from the GCP bucket.
