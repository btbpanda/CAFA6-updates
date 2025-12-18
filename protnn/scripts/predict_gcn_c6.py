import argparse
import sys
import os
import glob
import pandas as pd
import polars as pl
import numpy as np
import joblib
import yaml
import s3fs
import subprocess
from tqdm import tqdm
from pyarrow.parquet import read_schema

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))
print(sys.executable)
parser = argparse.ArgumentParser()

parser.add_argument('-g', '--graph-path', type=str)
parser.add_argument('-el', '--elabels-path', type=str)
parser.add_argument('-f', '--fasta', type=str)
parser.add_argument('-m', '--model-path', type=str)
parser.add_argument('-out', '--output', type=str)

parser.add_argument('-o', '--ontology', type=str, nargs='+',default=['bp', 'mf', 'cc'])
parser.add_argument('-d', '--devices', type=int, nargs='+')


ont_dict = {'bp': 0, 'mf': 1, 'cc': 2}

def get_params_from_cfg(path):

    with open(os.path.join(path, 'dumps/config.yaml'), 'r') as f:
         cfg = yaml.safe_load(f)

    params = {'cond': cfg['conditional']}
    params = {**params, **(cafa5_priors if cfg['train_data'] == 'cafa5' else cafa6_priors)}

    return params

if __name__ == '__main__':

    args = parser.parse_args()
    # Optional: set the device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.devices))

    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader

    try:
        from protlib.metric import obo_parser, Graph, ia_parser, get_topk_targets, get_depths
        from protnn.utils import get_labels, CAFAEvaluator, estimate_prior, Prediction, make_raw_prediction, \
            make_submission, CAFA6Evaluator

        from protnn.dataset import *
        from protnn.stacker import *

    except ImportError:
        print('Alarm')
        pass

    graph_path = args.graph_path

    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    for n, ontology in enumerate(args.ontology):
        mode = 'w' if n == 0 else 'a'
        nout = ont_dict[ontology]
        G = ontologies[nout]






