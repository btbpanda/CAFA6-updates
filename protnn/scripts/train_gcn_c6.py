import argparse
import sys
import os
import glob
import pandas as pd
import numpy as np
import joblib
import yaml
from tqdm import tqdm
from numba import set_num_threads

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))
print(sys.executable)
parser = argparse.ArgumentParser()

parser.add_argument('-g', '--graph-path', type=str)
parser.add_argument('-ia', '--ia-path', type=str)
parser.add_argument('-t', '--target-path', type=str)
parser.add_argument('-to', '--target-old-path', type=str)
parser.add_argument('-el', '--elabels-path', type=str)
parser.add_argument('-tl', '--test-path', type=str)
parser.add_argument('-tt', '--train-terms', type=str)

parser.add_argument('-f', '--fasta', type=str)
parser.add_argument('-out', '--output', type=str)

parser.add_argument('-o', '--ontology', type=str)
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-d', '--devices', type=int, nargs='+')


ont_dict = {'bp': 0, 'mf': 1, 'cc': 2}

def train_gcn(model, swa, train_dl, val_dl, evaluator, n_ep=20, lr=1e-3, clip_grad=1, weight_decay=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    scores = []
    for n in range(n_ep):

        model.train()
        for batch in tqdm(train_dl):
            batch = {x: batch[x].cuda() for x in batch}
            opt.zero_grad()

            output = model(batch)
            loss = loss_fn(output, batch['y'])
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad)
            opt.step()

        score = evaluator(model, val_dl)
        swa.add_checkpoint(model, score=score)
        print(f'Epoch {n}: CAFA5 score {score}')
        scores.append(score)

    return model, swa, scores

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
            make_submission, CAFA6x3Evaluator

        from protnn.dataset import *
        from protnn.stacker import *
        from protnn.swa import SWA

    except ImportError:
        print('Alarm')
        pass

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_num_threads(config['train_params']['num_workers'])

    graph_path = args.graph_path
    ia_path = args.ia_path
    NOUT = ont_dict[args.ontology]  # ontology to train

    work_dir = os.path.join(args.output, args.ontology)
    temp_dir = os.path.join(work_dir, 'temp')
    swa_dir = os.path.join(work_dir, 'swa')
    os.makedirs(temp_dir, exist_ok=True)
    ############################################
    # PREPARE
    ############################################

    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, ia_parser(ia_path), True))

    G = ontologies[NOUT]
    root_id = [x['id'] for x in G.terms_list if len(x['adj']) == 0][0]

    train_part = sorted(
        glob.glob(os.path.join(args.target_path, G.namespace, '*.parquet'))
    )

    train_old_part = sorted(
        glob.glob(os.path.join(args.target_old_path, G.namespace, '*.parquet'))
    )

    ############################################
    # ESTIMATE PRIORS
    ############################################
    cafa5_priors, cafa6_priors = {}, {}

    cafa5_priors['prior_raw'], cafa5_priors['prior_cond'] = estimate_prior(
        train_part + train_old_part, G, batch_size=100
    )

    cafa6_priors['prior_raw'], cafa6_priors['prior_cond'] = estimate_prior(
        train_part, G, batch_size=100
    )

    ############################################
    # TRAIN DATA
    ############################################
    target_path = train_part
    pred_files = ['oof_pred.parquet', ]
    train = pd.read_feather(
        os.path.join(args.fasta, 'train_seq.feather')
    )
    # if cafa5, add more data
    if config['train_data'] == 'cafa5':
        target_path = target_path + train_old_part
        pred_files = pred_files + ['oof_old_pred.parquet', ]
        old_train = pd.read_feather(
            os.path.join(args.fasta, 'old_train_seq.feather')
        )
        train = pd.concat([train, old_train, ], ignore_index=True)

    target = pd.read_parquet(
        target_path,
        columns=[x['id'] for x in G.terms_list]
    ).fillna(0)

    train_sl = np.nonzero(target[root_id].values == 1)[0]
    target = target.values[train_sl]
    train = train.iloc[train_sl].reset_index(drop=True)

    # get electronic annotation
    goa_data = [
        get_labels(
            path=os.path.join(args.elabels_path, 'train_auto.tsv'),
            G=G, idx=train['EntryID']
        ) + get_labels(
            path=os.path.join(args.elabels_path, 'old_train_auto.tsv'),
            G=G, idx=train['EntryID']
        ),
    ]  # in old format there is a list of goa labels Series

    # load model predictions
    train_preds = [
        Prediction(
            path=[os.path.join(model_path, 'predictions', x) for x in pred_files],
            graph=G, prot_ids=train['EntryID'], **get_params_from_cfg(model_path)
        ) for model_path in config['models']
    ]

    ############################################
    # VALID DATA
    ############################################

    no_know = pd.read_csv(
        os.path.join(args.test_path, 'no-know.tsv'), sep='\t'
    )

    lim_know = pd.read_csv(
        os.path.join(args.test_path, 'lim-know.tsv'), sep='\t'
    )

    part_know = pd.read_csv(
        os.path.join(args.test_path, 'part-know.tsv'), sep='\t'
    )

    ids_to_take = pd.concat([no_know, lim_know, part_know], ignore_index=True)['EntryID'] \
        .drop_duplicates().values

    # test_labels = pd.read_csv(
    #     os.path.join(args.elabels_path, 'test_leak_terms.tsv'), sep='\t'
    # ).drop_duplicates().reset_index(drop=True)
    #
    # test_labels.to_csv(os.path.join(temp_dir, 'labels.tsv'), index=False, sep='\t')
    # ids_to_take = test_labels['EntryID'].drop_duplicates().values

    test_goa_data = [
        get_labels(
            path=os.path.join(args.elabels_path, 'test_auto226.tsv'),
            G=G, idx=ids_to_take
        )
    ]

    test_preds = [
        Prediction(
            path=os.path.join(model_path, 'predictions/test', ),
            graph=G, prot_ids=ids_to_take, **get_params_from_cfg(model_path)
        ) for model_path in config['models']
    ]

    ############################################
    # DEFINE MODEL
    ############################################
    train_ds = StackDataset(
        train_preds,
        G,
        goa_list=goa_data,
        p_goa=config['train_params']['p_goa'],
        targets=target
    )
    train_dl = DataLoader(
        train_ds, batch_size=config['train_params']['batch_size'], shuffle=True,
        num_workers=min(os.cpu_count(), config['train_params']['num_workers'])
    )

    val_ds = StackDataset(
        test_preds,
        G,
        goa_list=test_goa_data,
        p_goa=1,
        targets=None
    )
    val_dl = DataLoader(
        val_ds, batch_size=config['train_params']['test_batch_size'], shuffle=False,
        num_workers=min(os.cpu_count(), config['train_params']['num_workers'])
    )

    model = GCNStacker(
        len(config['models']), 1, G,
        hidden_size=config['train_params']['hidden_size'],
        n_layers=config['train_params']['n_layers'],
        embed_size=config['train_params']['embed_size']
    ).cuda()

    if len(args.devices) > 1:
        model = nn.DataParallel(model)

    swa = SWA(
        config['train_params']['store_swa'], path=swa_dir, rewrite=True
    )

    evaluator = CAFA6x3Evaluator(
        args.graph_path,
        args.ia_path,
        temp_dir=temp_dir,
        G=G,
        idx=ids_to_take,
        train_terms=args.train_terms,
        test_terms=args.test_path
    )

    # test evaluation
    score = evaluator(model, val_dl)
    print('Test score value: ', score)

    ############################################
    # FIT AND SAVE
    ############################################
    model, swa, scores = train_gcn(
        model,
        swa,
        train_dl,
        val_dl,
        evaluator,
        n_ep=config['train_params']['n_ep'],
        lr=config['train_params']['lr'],
        clip_grad=config['train_params']['clip_grad'],
        weight_decay=config['train_params']['weight_decay']
    )
    joblib.dump(swa, os.path.join(work_dir, f'swa.pkl'))

    # validate SWA
    model = swa.set_weights(model, config['train_params']['use_swa'], weighted=False)

    if type(model) is nn.DataParallel:
        model = model.module

    torch.save(model.state_dict(), os.path.join(work_dir, f'checkpoint.pth'))

    with open(os.path.join(work_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    np.savez(
        os.path.join(work_dir, 'cafa5_priors.npz'), **cafa5_priors
    )

    np.savez(
        os.path.join(work_dir, 'cafa6_priors.npz'), **cafa6_priors
    )

    score = evaluator(model, val_dl)
    print('Final CAFA5 score', score)
