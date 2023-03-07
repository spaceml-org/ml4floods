# The *WorldFloods* database

The [*WorldFloods* database](https://www.nature.com/articles/s41598-021-86650-z) contains 424 pairs of Sentinel-2 images and flood segmentation masks. 
It requires approximately 300GB of hard-disk storage. The *WorldFloods* database is released under a [Creative Commons non-commercial licence](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt) 
<img src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" alt="licence" width="60"/>

```{image} ./prep/images/worldfloods_v1.png
:alt: Floods devestate the world every year.
:align: center
```


## Download the data from the Google Drive

A subset of the data and the pretrained models are available in this [public Google Drive folder](https://drive.google.com/folderview?id=1dqFYWetX614r49kuVE3CbZwVO6qHvRVH). 

If you want to use this data from the Google Colab you can *'add a shortcut to your Google Drive'* from the [public Google Drive folder](https://drive.google.com/folderview?id=1dqFYWetX614r49kuVE3CbZwVO6qHvRVH) and mount that directory:


```{image} ./prep/images/add_shortcut_drive.png
:alt: Floods devestate the world every year.
:align: center
```
	
```python
from google.colab import drive
drive.mount('/content/drive')
!ls '/content/drive/My Drive/Public WorldFloods Dataset'
```

Alternatively you can download it manually from that folder or even automatically with the [gdown](https://github.com/wkentaro/gdown) package. 

```bash
gdown --id 11O6aKZk4R6DERIx32o4mMTJ5dtzRRKgV
```

## Download the data from the Google Bucket

The database is available in this Google bucket: `gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/`. This Google bucket is in *requester pays* mode, hence you'd need a GCP project to download the data. To download the entire dataset run:

```bash
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/worldfloods_v1_0.zip .
```

If you want only an specific subset (train, train_sample, val or test) run:

```bash
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/train_v1_0.zip .
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/train_sample_v1_0.zip .
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/val_v1_0.zip .
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_v1_0/test_v1_0.zip .
```

If you want to download the pre-trained models of [this work](https://www.nature.com/articles/s41598-021-86650-z) run:

```bash
mkdir WFV1_scnn20
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/2_MLModelMart/WFV1_scnn20/config.json WFV1_scnn20/
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/2_MLModelMart/WFV1_scnn20/model.pt WFV1_scnn20/

mkdir WFV1_unet
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/2_MLModelMart/WFV1_unet/config.json WFV1_unet/
gsutil -u your-project cp gs://ml4cc_data_lake/2_PROD/2_Mart/2_MLModelMart/WFV1_unet/model.pt WFV1_unet/
```


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
