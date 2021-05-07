# Environments

> We provide environments for reproducing certain aspects of our code. There will be dedicated environments to facilitate certain bits of the pipeline and then more general environments.

---
## Environments

* `environment.yml` - the more general environment to install **all dependencies** for every aspect of the pipeline. (Use this! :])
* `environment_dataprep.yml` there are some extra environments for getting and loading data. Will be shown in specific notebooks.
* `environment_vis.yml`
* `jupyterlab.yml` - for setting up your own jupyterlab environment. Can be for dev or just to explore.

---
## Installation Instructions

The `environment.yml` file will have the most updated distributions for the **packages**. If you'd like to have **jupyterlab**, see below.

1. Clone the repository.

```bash
git clone https://github.com/spaceml-org/ml4floods/
```

2. Install using conda.

```bash
conda env create -f environments/environment.yml
```

3. If you already have the environment installed, you can update it.

```bash
conda activate ml4fl_py38
conda env update --file environments/environment.yml
```


---
## JupyterLab

Some instructions for how to install JupyterLab on your own system. There are detailed instructions [here](https://jejjohnson.github.io/research_journal/tutorials/remote_computing/vscode_jlab/) but below are the basics.

### Why?

* The paths system is a mess. It also not only affects your productivity and mental sanity.
* Because there are two separate paths `/home/jupyter/` and `/home/user` inevitably, people tend to call things from `/home/jupyter` that just don't exist or vice versa. There are always artifacts in peoples code of this which causes things to break for others when they pull changes.
* Transferable skills. You won't always have a `notebook instance` at your disposal, so you'll be able to do this with any remote system that allows for ssh! :)

### 1 Install JupyterLab

```bash
conda env install -f jupyterlab.yml
```

### 2 Install some extensions

It's nice to have and it's worth having for later.

```bash
# activate environment
conda activate jupyterlab
# Install jupyter lab extension manager
jupyter labextension install @jupyter-widgets/jupyterlab-manager
# Enable
jupyter serverextension enable --py jupyterlab-manager
```

### 3 Run JupyterLab

You can run the jupyterlab instance through vscode

```bash
# activate environment (if you haven't already)
conda activate jupyterlab
# or source activate jupyterlab
jupyter-lab --no-browser --port XXXX
```

**or** you can run an instance through `gcloud + ssh` via the terminal

```bash
# sign in and have a port open
gcloud compute ssh --project XXX --zone XXXX USERNAME@VM-INSTANCE -- -L XXXX:localhost:XXXX
# start jupyterlab
conda activate jupyterlab
jupyter-lab --no-browser --port XXXX
```

Then you go to your local browser and type in: `localhost:XXXX`

### 4 Create Your Conda Environment

On the front [README.md](../README.md), there is a common `environment.yml`.

**Everyone should be using this and if you add a new package, you need to update this package environment so that everyone is using the same environment**.

---

#### Note - Other Environments

This is super important, you do not need to install jupyterlab in every conda environment you make. Just make sure that you install `ipykernel` and the general `jupyterlab` conda environment will be able to see other environments. So you'll be able to select the python kernel for your appropriate notebook when you run it.

---

#### Note - Directory to Run

Whichever directory you run JupyterLab, it's the same directory you're going to see in your file explorer on the left panel of JupyterLab. I'd recommend that you run JupyterLab from the top level directory of either your home directory `/home/user/` or the top top directory `/`. That way you'll have access to the file system on your file explorer to the left. So you simply need to navigate to that directory `cd /`, run jupyterlab and you'll be good to go.

#### Getting the Root Directory


Most of the time we want to call stuff from our source module, e.g. `/home/user/repo/`. So there is a package installed in the environment.yml that's called pyprojroot which fixes this issue. Basically, there is a `.here` file in the top directory that marks the top level directory to our modules. And this package will spyder it's way up through the parents until it finds it. Then we append that to our paths and now we can call functions.


##### Demo Notebook Start


```python

import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(here()))

```

In general, if you can, try to avoid hacky things like `sys.path.append("../../")` because weird things tend to happen when other people use your code. (I do this all the time btw...)
### JupyterLab + VSCode

A helpful tutorial if you wish to use VSCode+JupyterLab [together](https://jejjohnson.github.io/research_notebook/content/tutorials/remote/vscode_jlab.html).


---
## Google Earth Engine Authentication


