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
parser.add_argument('-tm', '--test-mode', type=str, default='226')

parser.add_argument('-out', '--output', type=str)

parser.add_argument('-o', '--ontology', type=str, nargs='+', default=['bp', 'mf', 'cc'])
parser.add_argument('-tk', '--topk', type=int, default=500)
parser.add_argument('-t', '--tau', type=float, default=0.01)

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
        from protnn.utils import get_labels, estimate_prior, Prediction, make_submission

        from protnn.dataset import *
        from protnn.stacker import *

    except ImportError:
        print('Alarm: Failed to import required modules')
        raise

    graph_path = args.graph_path

    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    postfix = args.test_mode
    # if not a number, '_' is used to define the mode...
    try:
        int(postfix)
    except ValueError:
        postfix = '_' + postfix

    elabels_path = os.path.join(args.elabels_path, f'test_auto{postfix}.tsv')

    for n, ontology in enumerate(args.ontology):
        print(f'Predicting {ontology}...')

        nout = ont_dict[ontology]
        G = ontologies[nout]

        model_path = os.path.join(args.model_path, ontology)
        
        # Load config
        with open(os.path.join(model_path, 'config.yaml'), 'r') as f:
            cfg = yaml.safe_load(f)

        # Initialize model
        model = GCNStacker(
            len(cfg['models']), 1, G,
            hidden_size=cfg['train_params']['hidden_size'],
            n_layers=cfg['train_params']['n_layers'],
            embed_size=cfg['train_params']['embed_size']
        ).cuda()
        
        # Load trained weights
        checkpoint_path = os.path.join(model_path, 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        model.eval()  # Set to evaluation mode
        
        print(f"Loaded model from {checkpoint_path}")

        # Multi-GPU support
        if len(args.devices) > 1:
            model = nn.DataParallel(model)

        # Load priors
        cafa5_priors = np.load(
            os.path.join(model_path, 'cafa5_priors.npz')
        )

        cafa6_priors = np.load(
            os.path.join(model_path, 'cafa6_priors.npz')
        )

        # Iterate over TTA configs
        for k, tta_cfg in enumerate(cfg['tta']):
            print(f'Running TTA config {k}: {tta_cfg}...')

            output_path = os.path.join(args.output, f'pred_tta_{k}.tsv')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            model_names = cfg['tta'][tta_cfg]
            print(f"Using models: {model_names}")

            # Get partitions from first prediction
            partitions = sorted(
                glob.glob(os.path.join(model_names[0], 'predictions/test', '*.parquet'))
            )
            
            if not partitions:
                print(f"Warning: No partitions found in {model_names[0]}/predictions/test/")
                continue
            
            print(f"Found {len(partitions)} partitions to process")

            for j, part in enumerate(map(os.path.basename, partitions)):
                mode = 'a' if n + j else 'w'
                
                print(f"Processing partition {j+1}/{len(partitions)}: {part}")
                
                # Load test IDs
                test_id = pd.read_parquet(
                    os.path.join(model_names[0], 'predictions/test', part), 
                    columns=['EntryID']
                )['EntryID'].tolist()

                # Load predictions from all models
                test_preds = [
                    Prediction(
                        path=os.path.join(x, 'predictions/test', part),
                        graph=G, 
                        prot_ids=test_id, 
                        **get_params_from_cfg(x)
                    ) for x in model_names
                ]

                # Load GO annotations
                test_goa_data = [
                    get_labels(
                        path=elabels_path, 
                        G=G, 
                        idx=test_id
                    )
                ]

                # Create dataset and dataloader
                test_ds = StackDataset(
                    test_preds,
                    G,
                    goa_list=test_goa_data,
                    p_goa=1,
                    targets=None
                )
                test_dl = DataLoader(
                    test_ds, 
                    batch_size=cfg['train_params']['test_batch_size'], 
                    shuffle=False,
                    num_workers=min(os.cpu_count(), cfg['train_params']['num_workers'])
                )

                # Generate predictions
                with torch.no_grad():  # Ensure no gradients are computed
                    make_submission(
                        model,
                        test_dl,
                        G,
                        test_id,
                        output_path,
                        mode=mode,
                        topk=args.topk,
                        tau=args.tau
                    )
                
                print(f"Completed partition {j+1}/{len(partitions)}")
            
            print(f"Predictions saved to {output_path}")
        
        print(f"Completed predictions for {ontology}")
    
    print("All predictions completed successfully!")
