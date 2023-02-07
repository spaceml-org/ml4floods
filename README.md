
<p align="center">
    <img src="https://raw.githubusercontent.com/spaceml-org/ml4floods/main/jupyterbook/ml4floods_banner.png" alt="awesome ml4floods" width="50%">
</p>

ML4Floods is an end-to-end ML pipeline for flood extent estimation: from data preprocessing, model training, model deployment to visualization.

<p align="center">
    <img src="https://raw.githubusercontent.com/spaceml-org/ml4floods/main/jupyterbook/content/ml4ops/ts_albania.gif" alt="awesome flood extent estimation" width="100%">
</p>

## Install

Install from [pip](https://pypi.org/project/ml4floods/):
```bash
pip install ml4floods
```

Install the latest version from GitHub:

```bash
pip install git+https://github.com/spaceml-org/ml4floods#egg=ml4floods
```

## Docs
[spaceml-org.github.io/ml4floods](https://spaceml-org.github.io/ml4floods)

These tutorials may help you explore the datasets and models:
* [Project rationale](https://spaceml-org.github.io/ml4floods/content/intro/introduction.html).
* [Run the model on time series of Sentinel-2 images](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_inference_on_image_time_series.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_inference_on_image_time_series.ipynb)
* [ML-models step by step](https://spaceml-org.github.io/ml4floods/content/ml_overview.html)
    * [Training](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_Train_models.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Train_models.ipynb)
    * [Inference on new data](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_Run_Inference_on_new_data.html) (a Sentinel-2 image) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_on_new_data.ipynb)
    * [Perf metrics](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_performance_metrics_workflow.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_performance_metrics_workflow.ipynb)
* [Ingest data from Copernicus EMS](https://spaceml-org.github.io/ml4floods/content/prep/full_data_ingest.html)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/prep/full_data_ingest.ipynb)

## The *WorldFloods* database

The [*WorldFloods* database](https://www.nature.com/articles/s41598-021-86650-z) contains 444 pairs of Sentinel-2 images and flood segmentation masks. 
It requires approximately 300GB of hard-disk storage. 
The *WorldFloods* database is released under a [Creative Commons non-commercial licence](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt) <img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" alt="licence" width="60"/>

To download the *WorldFloods* database or the pretrained flood segmentation models for Sentinel-2 see [the instructions to download the database](https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html).

## Cite

If you find this work useful please cite:

```
@article{mateo-garcia_towards_2021,
	title = {Towards global flood mapping onboard low cost satellites with machine learning},
	volume = {11},
	issn = {2045-2322},
	doi = {10.1038/s41598-021-86650-z},
	number = {1},
	urldate = {2021-04-01},
	journal = {Scientific Reports},
	author = {Mateo-Garcia, Gonzalo and Veitch-Michaelis, Joshua and Smith, Lewis and Oprea, Silviu Vlad and Schumann, Guy and Gal, Yarin and Baydin, Atılım Güneş and Backes, Dietmar},
	month = mar,
	year = {2021},
	pages = {7249},
}
```
