#!/bin/bash

#conda create -n rapids-25.12 -c rapidsai -c conda-forge \
#    cudf=25.12 python=3.12 'cuda-version>=12.2,<=12.9' \
#    'pytorch=*=*cuda*' --solver=libmamba

conda create -n rapids-26.06 -c rapidsai -c conda-forge \
    rapids=26.06 python=3.12 'cuda-version>=12.2,<=12.9' \
    'pytorch=*=*cuda*' --solver=libmamba

conda activate rapids-26.06

pip install -r requirements.txt
