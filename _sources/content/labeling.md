# Viewer and label editor

* Author: Gonzalo Mateo-García
---
## Brief description
The ml4floods viewer and label editor is a Flask web application to view and manually edit the floodmaps of the *WorldFloods* dataset.
We use this application to keep growing *WorldFloods* and to improve the quality of its labels. In order to test it, you need
to [download the *Worldfloods* dataset (at least a subset of it)](./worldfloods_dataset.md). 

To launch the viewer clone the package and run:

```bash
git clone git@github.com:spaceml-org/ml4floods.git
cd ml4floods/viewer

python serve.py --root_location /path/to/worldfloods_v1_0/ --gt_version v1 --no_save_floodmap_bucket
```

The following video shows the viewer in action:

<iframe width="600" height="400"
src="https://www.youtube.com/embed/Rh7-ght-mY8?autoplay=0">
</iframe>
