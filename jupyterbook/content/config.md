# Introduction

---
ML4Floods is python package to do *end-to-end* flood extent estimation from optical images using deep learning models.

```{figure} ./ml4ops/ts_albania.gif
---
name: ts_albania
width: 100%
align: center
---
```

## Democratising AI-Enhanced Flooding Tools

Machine learning (ML) algorithms have the potential to offer significantly faster and more accurate flood mapping than traditional methods. Their adaptability means they can easily grow to accommodate more data over time, and expand to ingest a wide range of data types. Such ‘data fusion’ capabilities are potentially game-changing for a broad spectrum of users, from emergency workers to urban planners.

Training robust and reliable ML models is almost an art-form, requiring specialist knowledge of statistics, computing and data platforms. ML workflows have become much more accessible because of dedicated open-source libraries like PyTorch and TensorFlow. However, there are a myriad of subtle pitfalls associated with training and deploying ML models - these can produce deeply skewed results that still appear reasonable to the untrained eye. Democratising end-to-end integrated AI workflows avoids these pitfalls by creating a series of linked tools that non-ML expert users can trust to deploy machine learning. These tools incorporate data acquisition, preparation, calibration, enhancement and deployment steps, wrapped in an accessible interface. The full workflow is embedded in a visualisation tool that enables users to query, understand and use the insights and information generated. This unified suite of tools can get users 80% of the way and then allows straightforward customisation to a specific problem sub-domain.

## Install

To install the package run:

```
pip install ml4floods
```

## Tutorials

ML4Floods is a self-contained tool for training and deploying flood extent segmentation models for Sentinel-2 and Landsat. These tools include: image downloading, flood map acquisition, neural network training, testing and the visualization of the results in an interactive map. 
 See the [project rationale](./intro/introduction.md) for a more detailed explanation of the goals of the tool.

These tutorials may help you explore the datasets and models:
* [Run the clouds-aware flood segmentation model in Sentinel-2 and Landsat and vectorise the flood maps](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_postprocess_inference.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_postprocess_inference.ipynb)
* [Run the model on time series of Sentinel-2 images](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_inference_on_image_time_series.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_inference_on_image_time_series.ipynb)
* [Ingest data from Copernicus EMS](https://spaceml-org.github.io/ml4floods/content/prep/full_data_ingest.html)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/prep/full_data_ingest.ipynb)
* [ML-models step by step](https://spaceml-org.github.io/ml4floods/content/ml_overview.html)
    * [Training](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_Train_models.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Train_models.ipynb)
    * [Inference on new data](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_Run_Inference_on_new_data.html) (a Sentinel-2 image) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_on_new_data.ipynb)
    * [Perf metrics](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_performance_metrics_workflow.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_performance_metrics_workflow.ipynb)

<iframe width="600" height="400"
src="https://www.youtube.com/embed/wMLuHf9s9zk?autoplay=0">
</iframe>
 
## About 

This work is an extension of the [FDL Europe 2019](https://fdleurope.org/) *"Disaster Prevention, Progress and Response"* team which results are published in:
 
 > G. Mateo-Garcia, J. Veitch-Michaelis, L. Smith, S. Oprea, G. Schumann, Y. Gal, Baydin G.A., Backes D.  [Towards global flood mapping onboard low cost satellites with machine learning](https://www.nature.com/articles/s41598-021-86650-z). _Scientific Reports 11, 7249_ (2021). DOI: 10.1038/s41598-021-86650-z

This work has been further extended in:

> E. Portalés-Julià, G. Mateo-García, C. Purcell, and L. Gómez-Chova [Global flood extent segmentation in optical satellite images](https://www.nature.com/articles/s41598-023-47595-7). _Scientific Reports 13, 20316_ (2023). DOI: 10.1038/s41598-023-47595-7.


ML4Floods has been funded by the United Kingdom Space Agency (UKSA) and led by [Trillium Technologies](http://trillium.tech/). It has also been partially supported by the Spanish Ministry of Science and Innovation project PID2019-109026RB-I00 (MINECO-ERDF MCIN/AEI/10.13039/501100011033).

 ## Citation
 
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

## Licence
The package is available in [GitHub](https://github.com/spaceml-org/ml4floods). ML4Floods is published under a [GNU Lesser GPL v3 licence](https://www.gnu.org/licenses/lgpl-3.0.en.html) <img src="https://www.gnu.org/graphics/lgplv3-88x31.png" alt="licence" width="80">.

The *WorldFloods* database and all pre-trained models are released under a [Creative Commons non-commercial licence](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt) 
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" alt="licence" width="60"/>

 ## Contributors
 
 *Gonzalo Mateo-García*, *Enrique Portalés-Julià*, *Tarun Narayanan*, *J. Emmanuel Jonhson*, *Nadia Ahmed, Sam Budd, Satyarth Praveen, Lucas Kruitwagen, Margaret Maynard-Reid, Nicholas Roth, Cormac Purcell, Richard Strange, Leo Silverberg, Guy Schumann, Edoardo Nemni, Luis Gómez-Chova, Freddie Kalaitzis, Sara Jennings, Jodie Hughes* and *James Parr*.