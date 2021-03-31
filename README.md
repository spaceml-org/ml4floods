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

