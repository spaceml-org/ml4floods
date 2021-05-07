## Models

This part of the repo have the code to train and run inference with the WorldFloods trained models. 
Examples of intended use of these models can be found in [notebooks in the models section](https://github.com/spaceml-org/ml4floods/tree/main/jupyterbook/content/ml4ops)

Additionally, in this folder there is a script called `main.py` to run training or inference from the command line.  

The following command will train with the default configuration (linear model). The configuration file should be specified 
in the file: `configurations/worldfloods_template.json`
 
```bash
PYTHONPATH=. python main.py --config configurations/worldfloods_template.json --train --wandb_entity USER_PROJECT_WANDB \
           --wandb_project worldfloods
```
