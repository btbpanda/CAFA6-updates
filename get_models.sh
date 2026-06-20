#!/bin/bash

# Inference DL embeddings
python ./model_scipts/run_embeddings_inference.py

# run base models
python ./model_scipts/run_pb_and_lin.py

# aggregate base models
python ./model_scipts/model_aggregator.py

# aggregate base models
python ./model_scipts/model_aggregator.py

# stacker fit_predict
python ./run_stacker/run_stacker.py

# postprocess and make submission
python ./run_stacker/postproc_and_sub.py