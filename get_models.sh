#!/bin/bash

# Inference DL embeddings
python ./model_scripts/run_embeds_inference.py

# run base models
python ./model_scripts/run_pb_and_lin.py

# aggregate base models
python ./model_scripts/model_aggregator.py

# aggregate base models
python ./model_scripts/model_aggregator.py

# stacker fit_predict
python ./model_scripts/run_stacker.py

# postprocess and make submission
python ./model_scripts/postproc_and_sub.py
