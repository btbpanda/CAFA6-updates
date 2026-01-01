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

def train_gcn(model, train_dl, val_dl, evaluator, n_ep=20, lr=1e-3, clip_grad=1, weight_decay=1e-2, 
              swa_start=5, swa_lr=0.05, eval_frequency=None):
    """
    Train GCN with SWALR and cosine annealing
    Evaluate CAFA5 metrics only at specified frequency (or only at the end if None)
    
    Args:
        eval_frequency: Evaluate every N epochs. If None, evaluate only at the end.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Initialize SWALR wrapper
    from protnn.swalr import SWALRWrapper
    swa_wrapper = SWALRWrapper(
        model, opt, 
        swa_start=swa_start, 
        swa_lr=swa_lr, 
        anneal_epochs=swa_start,
        anneal_strategy='cos'
    )
    swa_wrapper.set_train_loader(train_dl)
    
    scores = []
    
    for epoch in range(n_ep):
        # Training phase
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}/{n_ep}')
        for batch in pbar:
            batch = {x: batch[x].cuda() for x in batch}
            opt.zero_grad()

            output = model(batch)
            loss = loss_fn(output, batch['y'])
            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_grad)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{epoch_loss/n_batches:.4f}'})
        
        # Update SWA model and scheduler after each epoch
        swa_wrapper.step(model)
        
        # Get current learning rate
        current_lr = opt.param_groups[0]['lr']
        avg_loss = epoch_loss / n_batches
        print(f'Epoch {epoch+1}/{n_ep}: Loss = {avg_loss:.4f}, LR = {current_lr:.6f}')
        
        # Evaluate based on frequency
        should_evaluate = False
        if eval_frequency is not None and (epoch + 1) % eval_frequency == 0:
            should_evaluate = True
        elif eval_frequency is None and epoch == n_ep - 1:  # Only last epoch
            should_evaluate = True
            
        if should_evaluate:
            print(f'Evaluating at epoch {epoch+1}...')
            if epoch >= swa_start:
                # Use SWA model for evaluation
                eval_model = swa_wrapper.swa_model
                print('Using SWA model for evaluation')
            else:
                eval_model = model
                print('Using regular model for evaluation')
                
            score = evaluator(eval_model, val_dl)
            scores.append((epoch + 1, score))
            
            print(f'Epoch {epoch+1}: CAFA5 score = {score:.4f}')
    
    # Get final model
    final_model = swa_wrapper.get_final_model(model)
    
    print(f'\nTraining completed!')
    if scores:
        best_epoch, best_score = max(scores, key=lambda x: x[1])
        print(f'Best score: {best_score:.4f} at epoch {best_epoch}')
        print(f'Final score: {scores[-1][1]:.4f}')
    
    return final_model, scores


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

    from protlib.metric import obo_parser, Graph, ia_parser, get_topk_targets, get_depths
    from protnn.utils import get_labels, CAFAEvaluator, estimate_prior, Prediction, make_raw_prediction, \
        make_submission, CAFA6x3Evaluator

    from protnn.dataset import *
    from protnn.stacker import *

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_num_threads(min(os.cpu_count(), config['train_params']['num_workers']))

    graph_path = args.graph_path
    ia_path = args.ia_path
    NOUT = ont_dict[args.ontology]  # ontology to train

    work_dir = os.path.join(args.output, args.ontology)
    temp_dir = os.path.join(work_dir, 'temp')
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
    ]

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
    print('Initial test score value: ', score)

    ############################################
    # FIT AND SAVE
    ############################################
    
    # Set eval_frequency to None to evaluate only at the end
    # Or set to a number (e.g., 5) to evaluate every N epochs
    eval_frequency = config['train_params'].get('eval_frequency', None)
    
    model, scores = train_gcn(
        model,
        train_dl,
        val_dl,
        evaluator,
        n_ep=config['train_params']['n_ep'],
        lr=config['train_params']['lr'],
        clip_grad=config['train_params']['clip_grad'],
        weight_decay=config['train_params']['weight_decay'],
        swa_start=config['train_params'].get('swa_start', 5),
        swa_lr=config['train_params'].get('swa_lr', 0.05),
        eval_frequency=eval_frequency  # None = only at end, N = every N epochs
    )

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
    
    # Save scores with epoch numbers
    if scores:
        scores_dict = {
            'epochs': [s[0] for s in scores],
            'scores': [s[1] for s in scores]
        }
        np.savez(os.path.join(work_dir, 'training_scores.npz'), **scores_dict)
    
    print('\nTraining completed successfully!')
    print(f'Final model saved to {work_dir}')

