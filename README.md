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

Before running the code you need to change `config.yaml` according to you environment. What should be changed:
* Number of available GPUs
* Path to store the artifact 
* Put any valid email in order to be able to download articles data via API 

Also, the code uses kaggle API to download competition data and previous CAFA 5 competition 
which is also used to train the models. After `kaggle` library is installed you need to set it up to be able to use 
API via existing account. For the details pls refer the page [https://www.kaggle.com/discussions/getting-started/524433](https://www.kaggle.com/discussions/getting-started/524433) 

# NEXT STEPS

To reproduce the solution please run `./run.sh`.

You can also run data and model steps separately by executing `./get_data.sh` and `./get_models.sh`. 

# Make docker image

`docker build -t kaggle-cafa6-2nd-place-solution .`

# Run docker image
```
docker run --rm --gpus all \
    -v "$PWD/output:/workspace/output" \
    kaggle-cafa6-2nd-place-solution
```
