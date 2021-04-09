# MLOps overview

* **Authors**: Sam Budd, Gonzalo Mateo-García
---

The MLOps section have tutorials for training, testing and running inference of flood extent segmentation models for Sentinel-2. 
Models are trained in the *WorldFloods*[1] dataset which is freely accessible at `gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/`. 
Each of the tutorials is self-contained and can be run on Google Colab. 

```{image} ./ml4ops/diagram_mlops.png
:alt: MLOps diagram
:width: 90%
:align: center
```

---
Content:

* [Train models](./ml4ops/HOWTO_Train_models.ipynb): trains the WorldFloods model on the WorldFloods dataset from scratch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Train_models.ipynb)
* [Model metrics](./ml4ops/HOWTO_performance_metrics_workflow.ipynb): loads a worldfloods trained model and run inference on all images of the *WorldFloods* test dataset. It computes displays some standard metrics and the PR and ROC curves for water detection. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_performance_metrics_workflow.ipynb)
* [Inference on Sentinel-2 images](./ml4ops/HOWTO_Run_Inference_on_new_data.ipynb): loads a worldfloods pretrained model and runs inference on a Sentinel-2 image from the WorldFloods dataset. It shows the predictions vs the ground truth on that image. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_on_new_data.ipynb)
* [Probabilistic Neural Networks](./ml4ops/HOWTO_Calculate_uncertainty_maps.ipynb): Run inference of the U-Nets trained with dropout. We apply Bayesian dropout at inference time to obtain an ensemble of predictions. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Calculate_uncertainty_maps.ipynb)
* [Cloud removal inference](./ml4ops/HOWTO_Run_Inference_multioutput_binary.ipynb): Run inference of the multioutput binary classification model. This model is able to predict land/water under the clouds. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/ml4floods/blob/main/jupyterbook/content/ml4ops/HOWTO_Run_Inference_multioutput_binary.ipynb)

[1] Mateo-Garcia, G. et al. [Towards global flood mapping onboard low cost satellites with machine learning](https://www.nature.com/articles/s41598-021-86650-z). _Scientific Reports 11, 7249_ (2021). 