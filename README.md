Hello!

Here are the instructions to reproduce the CAFA6 2nd solution using given code

# CONTENTS

* `embeddings`                 : scripts for training Neural Network base models
* `protlib`                    : utils and code to train Py-Boost and LogReg models, data preprocessing and efficient metric computation
* `protnn`                     : utils and code to train GCN stacker model
* `articles`                   : tools for download and extract articles features
* `configs`                    : all base model config files. Should not be modified
* `data_scripts`               : data pipeline steps
* `model_scripts`              : models pipeline steps
* `config.yaml`                : config used to execute training and inference. Should be modified according the env
* `setup-env.sh`               : install all the requirements 
* `get_data.sh`                : execute data steps
* `get_models.sh`              : execute models steps
* `run.sh`                     : run all steps
* `CAFA6docs.pdf`              : detailed solution description


# HARDWARE 

We used the following setup to train:

* 48 CPUs
* 1T RAM
* 8 x Tesla A100 48 GB

It is possible to run with lower hardware, however it takes a lot of time. At least 1 NVIDIA GPU is required

# DATA AND ENV SETUP

To install default python dependencies, please execute `./setup-env.sh`

# NEXT STEPS

To reproduce the solution please run `./run.sh`.

You can also run data and model steps separately by executing `./get_data.sh` and `./get_models.sh`. 