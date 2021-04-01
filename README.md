# ML4Floods: _an ML pipeline to tackle flooding_

This repository contains an end-to-end ML pipeline for flood extent estimation: from data preprocessing, model training, model deployment to visualization.

<p align="center">
    <img src="images/ml4floods_logo_black.png" alt="awesome ml4floods" width="300">
</p>

Install the package:

```bash

pip install git+https://github.com/spaceml-org/ml4floods#egg=ml4floods

# alternatively use
python setup.py install

```

These notebooks may help you explore the datasets and models:
* [Data Preprocessing](https://github.com/spaceml-org/ml4floods/tree/main/notebooks/data/preprocessing)
* Models
    * [Training](https://github.com/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Train_models.ipynb)
    * Inference on [new data](https://github.com/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_on_new_data.ipynb) (a Sentinel-2 image)
    * [Perf metrics](https://github.com/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_performance_metrics_workflow.ipynb)
    * [Uncertainty visualisation](https://github.com/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Calculate_uncertainty_maps.ipynb)  


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
