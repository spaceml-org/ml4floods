
[![Article DOI:10.1038/s41598-023-47595-7](https://img.shields.io/badge/Article%20DOI-10.1038%2Fs41598.023.47595.7-blue)](https://doi.org/10.1038/s41598-023-47595-7)  [![PyPI](https://img.shields.io/pypi/v/ml4floods)](https://pypi.org/project/ml4floods/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ml4floods)](https://pypi.org/project/ml4floods/) [![PyPI - License](https://img.shields.io/pypi/l/ml4floods)](https://github.com/spaceml-org/ml4floods/blob/main/LICENSE) [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow)](https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2) [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/isp-uv-es/ml4floods) [![docs](https://badgen.net/badge/docs/spaceml-org.github.io%2Fml4floods/blue)](https://spaceml-org.github.io/ml4floods/)

<p align="center">
    <img src="https://raw.githubusercontent.com/spaceml-org/ml4floods/main/jupyterbook/ml4floods_banner.png" alt="awesome ml4floods" width="50%">
</p>

ML4Floods is an end-to-end ML pipeline for flood extent estimation: from data preprocessing, model training, model deployment to visualization. Here you can find the [WorldFloodsV2🌊 dataset](https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html) and [trained models 🤗](https://huggingface.co/isp-uv-es/ml4floods) for flood extent estimation in Sentinel-2 and Landsat.

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

[![docs](https://badgen.net/badge/docs/spaceml-org.github.io%2Fml4floods/blue)](https://spaceml-org.github.io/ml4floods/)

These tutorials may help you explore the datasets and models:
* [Kherson Dam Break *end-to-end* flood extent map](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_postprocess_inference.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_postprocess_inference.ipynb)
* [Run the model on time series of Sentinel-2 images](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_inference_on_image_time_series.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_inference_on_image_time_series.ipynb)
* [Ingest data from Copernicus EMS](https://spaceml-org.github.io/ml4floods/content/prep/full_data_ingest.html)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/prep/full_data_ingest.ipynb)
* [ML-models step by step](https://spaceml-org.github.io/ml4floods/content/ml_overview.html)
    * [Training](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_Train_models.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Train_models.ipynb)
    * [Inference on new data](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_Run_Inference_on_new_data.html) (a Sentinel-2 image) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_on_new_data.ipynb)
    * [Perf metrics](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_performance_metrics_workflow.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_performance_metrics_workflow.ipynb)

## The *WorldFloods* database
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8153514.svg)](https://doi.org/10.5281/zenodo.8153514)
 
The [*WorldFloods* database](https://www.nature.com/articles/s41598-023-47595-7) contains 509 pairs of Sentinel-2 images and flood segmentation masks. 
It requires approximately 76GB of hard-disk storage. 

The *WorldFloods* database and all pre-trained models are released under a [Creative Commons non-commercial licence](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt) 
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" alt="licence" width="60"/>

To download the *WorldFloods* database or the pretrained flood segmentation models see [the instructions to download the database](https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html).

## Cite

If you find this work useful please cite:

```
@article{portales-julia_global_2023,
	title = {Global flood extent segmentation in optical satellite images},
	volume = {13},
	issn = {2045-2322},
	doi = {10.1038/s41598-023-47595-7},
	number = {1},
	urldate = {2023-11-30},
	journal = {Scientific Reports},
	author = {Portalés-Julià, Enrique and Mateo-García, Gonzalo and Purcell, Cormac and Gómez-Chova, Luis},
	month = nov,
	year = {2023},
	pages = {20316},
}
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

## About

ML4Floods has been funded by the United Kingdom Space Agency (UKSA) and led by [Trillium Technologies](http://trillium.tech/). In addition, this research has been partially supported by the DEEPCLOUD project (PID2019-109026RB-I00) funded by the Spanish Ministry of Science and Innovation (MCIN/AEI/10.13039/501100011033) and the European Union (NextGenerationEU).
