"""This is GCN Stacker train and inference part. Script only manages training jobs, not implement train logic.
For algorithms pls refer protnn module
All runs are sequential, GPUs are utilized via DataParallel
"""
import sys
import subprocess
import yaml
from itertools import product
from pathlib import Path

if __name__ == '__main__':

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )

    data_path = Path(config['data_path']).resolve()

    uniprot_raw_path = data_path / 'uniprot/raw'
    stacker_feats_path = data_path / 'features'
    cafa6_path = data_path / 'cafa-6-protein-function-prediction'
    cafa6_target = cafa6_path / 'Train/train_terms.tsv'
    graph_path = cafa6_path / 'Train/go-basic.obo'
    ia_path = cafa6_path / 'IA.tsv'
    sparse_target = data_path / '/helpers/cafa6-sparse-labels/'
    old_sparse_target = data_path / '/helpers/uniprot-old-sparse-labels/'
    fasta_path = data_path / 'helpers/fasta'
    uniprot_raw_test_path = data_path / 'uniprot/raw/test'

    models_path = Path(config['models_path']).resolve()

    PYTHON = sys.executable
    DEVICES = ' '.join(map(str, config['devices']))

    train_template = f"""{PYTHON} ./protnn/scripts/train_gcn_c6_swalr_f.py \
        --models-path {models_path} \
        --graph-path {graph_path} \
        --ia-path {ia_path}\
        --target-path {sparse_target} \
        --target-old-path {old_sparse_target} \
        --elabels-path {stacker_feats_path} \
        --test-path {uniprot_raw_test_path} \
        --train-terms {cafa6_target} \
        --fasta {fasta_path} \
        --ontology {{ontology}} \
        --config ./configs/gcn_conf_with_nn_agg_tfidf_{{ontology}}.yaml \
        --devices {DEVICES} \
        --output stacking_nn_cross_pbtfidf_f_{{seed}} \
        --seed {{seed}}
    """

    prediction_template = f"""{PYTHON} ./protnn/scripts/predict_gcn_c6_swalr_f.py \
        --models-path {models_path} \
        --graph-path {graph_path} \
        --elabels-path {stacker_feats_path} \
        --model-path stacking_nn_cross_pbtfidf_f_{{seed}} \
        --test-mode 228 \
        --output prediction_nn_cross_pbtfidf_f_{{seed}} \
        --ontology bp mf cc \
        --devices {DEVICES}
    """

    TRAIN_TASKS = [
        train_template.format(ontology=ont, seed=s) for (ont, s) in product(['bp', 'mf', 'cc'], [42, 53])
    ]

    PREDICTION_TASKS = [
        prediction_template.format(seed=s) for s in [42, 53]
    ]

    # -----------------------------------
    # THIS IS TAKES LONG TIME ...
    # -----------------------------------

    for task in TRAIN_TASKS:
        print('Executing: ', task)
        subprocess.run(task, shell=True)


    # -----------------------------------
    # PREDICTION
    # -----------------------------------
    for task in PREDICTION_TASKS:
        print('Executing: ', task)
        subprocess.run(task, shell=True)
