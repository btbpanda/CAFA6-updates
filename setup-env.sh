#!/bin/bash

conda create -n rapids-25.12 -c rapidsai -c conda-forge \
    cudf=25.12 python=3.12 'cuda-version>=12.2,<=12.9' \
    'pytorch=*=*cuda*' --solver=libmamba

pip install -r requirements.txt