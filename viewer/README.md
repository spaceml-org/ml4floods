## *WorldFloods* viewer

Flask web application to view and label the *WorldFloods* extended dataset. It requires 
that some part of the dataset is downloaded in the hard disk (it could be copied from `gs://ml4cc_data_lake/2_PROD/2_Mart/worldfloods_extra_v2_0`)

Test it on the new version of the data:
```bash
python serve.py --root_location /path/to/worldfloods_extra_v2
```

Testing on the original *WorldFloods* dataset. Follow [these instructions to download the dataset](https://spaceml-org.github.io/ml4floods/content/worldfloods_dataset.html).

```bash
python serve.py --root_location /path/to/worldfloods_v1_0/ --gt_version v1 --no_save_floodmap_bucket
```

<img src="web/screenshot.png" alt="example" width="100%">