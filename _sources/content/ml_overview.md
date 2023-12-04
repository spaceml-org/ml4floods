# MLOps overview

* **Authors**: Sam Budd, Gonzalo Mateo-García
---

The MLOps section have tutorials for training, testing and running inference of flood extent segmentation models for Sentinel-2. 
Models are trained in the *WorldFloods* dataset which is [freely accessible](./worldfloods_dataset.md).
Each of the tutorials is self-contained and can be run on Google Colab. 

```{image} ./ml4ops/diagram_mlops.png
:alt: MLOps diagram
:width: 90%
:align: center
```

---
Tutorials on models of [Portalés-Julià et al 2023](https://www.nature.com/articles/s41598-023-47595-7).

* [Inference with clouds aware flood segmentation model](./ml4ops/HOWTO_Run_Inference_multioutput_binary.ipynb): Run inference of the multioutput binary classification model. This model is able to predict land/water under the clouds. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_multioutput_binary.ipynb)
* [Inference on time series of Sentinel-2 images](./ml4ops/HOWTO_inference_on_image_time_series.ipynb): Download a time series of Sentinel-2 images over an area of interest and run inference on them. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_inference_on_image_time_series.ipynb)
* [Run the clouds-aware flood segmentation model in Sentinel-2 and Landsat and vectorise the flood maps](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_postprocess_inference.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_postprocess_inference.ipynb)

Tutorials on models of [Mateo-García et al 2021](https://www.nature.com/articles/s41598-023-47595-7).

* [Train models](./ml4ops/HOWTO_Train_models.ipynb): trains the WorldFloods model on the WorldFloods dataset from scratch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Train_models.ipynb)
* [Model metrics](./ml4ops/HOWTO_performance_metrics_workflow.ipynb): loads a worldfloods trained model and run inference on all images of the *WorldFloods* test dataset. It computes displays some standard metrics and the PR and ROC curves for water detection. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_performance_metrics_workflow.ipynb)
* [Inference on Sentinel-2 images](./ml4ops/HOWTO_Run_Inference_on_new_data.ipynb): loads a worldfloods pretrained model and runs inference on a Sentinel-2 image from the WorldFloods dataset. It shows the predictions vs the ground truth on that image. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_on_new_data.ipynb)


Exploratory work: 

* [Probabilistic Neural Networks](./ml4ops/HOWTO_Calculate_uncertainty_maps.ipynb): Run inference of the U-Nets trained with dropout. We apply Bayesian dropout at inference time to obtain an ensemble of predictions. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Calculate_uncertainty_maps.ipynb)

-----
[[Mateo-Garcia et al 2021]] Mateo-Garcia, G. et al. [Towards global flood mapping onboard low cost satellites with machine learning](https://www.nature.com/articles/s41598-021-86650-z). _Scientific Reports 11, 7249_ (2021). DOI: 10.1038/s41598-021-86650-z.

[[Portalés-Julià et al 2023]] E. Portalés-Julià, G. Mateo-García, C. Purcell, and L. Gómez-Chova [Global flood extent segmentation in optical satellite images](https://www.nature.com/articles/s41598-023-47595-7). _Scientific Reports 13, 20316_ (2023). DOI: 10.1038/s41598-023-47595-7.
