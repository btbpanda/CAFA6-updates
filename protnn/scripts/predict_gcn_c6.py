import argparse
import sys
import os
import glob
import pandas as pd
import numpy as np
import yaml


sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))
print(sys.executable)
parser = argparse.ArgumentParser()

parser.add_argument('-g', '--graph-path', type=str)
parser.add_argument('-el', '--elabels-path', type=str)
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
        # mode = 'w' if n == 0 else 'a'
        nout = ont_dict[ontology]
        G = ontologies[nout]

        model_path = os.path.join(args.model_path, ontology)
        with open(os.path.join(model_path, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)
            
        model = GCNStacker(
            len(cfg['models']), 1, G,
            hidden_size=cfg['train_params']['hidden_size'],
            n_layers=cfg['train_params']['n_layers'],
            embed_size=cfg['train_params']['embed_size']
        ).cuda()
        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pth')))


        cafa5_priors = np.load(
            os.path.join(model_path, 'cafa5_priors.npz')
        )

        cafa6_priors = np.load(
            os.path.join(model_path, 'cafa6_priors.npz')
        )

        # iterate over tta cfgs
        for k, tta_cfg in enumerate(cfg['tta']):
            output_path = os.path.join(args.output, ontology, f'pred_tta_{k}.tsv')
            model_names = cfg['tta'][tta_cfg]

            # get partitions from first prediction
            for j, part in enumerate(
                    map(os.path.basename, glob.glob(os.path.join(model_names[0], '*.parquet')))
            ):
                mode ='a' if n + j else 'w'
                test_id = pd.read_parquet(
                    os.path.join(model_names[0], part), columns=['EntryID']
                )['EntryID'].tolist()

                test_preds = [
                    Prediction(
                        # TODO: check predictions path
                        path=os.path.join(x, 'test', part),
                        graph=G, prot_ids=test_id, **get_params_from_cfg(model_path)
                    ) for x in model_names
                ]

                test_goa_data = [
                    get_labels(
                        path=os.path.join(args.elabels_path, 'test_auto.tsv'),
                        G=G, idx=test_id
                    )
                ]

                test_ds = StackDataset(
                    test_preds,
                    G,
                    goa_list=test_goa_data,
                    p_goa=1,
                    targets=None
                )
                test_dl = DataLoader(
                    test_ds, batch_size=cfg['train_params']['test_batch_size'], shuffle=False,
                    num_workers=min(os.cpu_count(), cfg['train_params']['num_workers'])
                )

                make_submission(
                    model,
                    test_dl,
                    G,
                    test_id,
                    output_path,
                    mode=mode,
                    topk=500,
                    tau=0.01
                )





