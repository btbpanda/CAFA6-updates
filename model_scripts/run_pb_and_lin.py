"""This is py-boost and lin reg model runner. Script only manages training jobs, not implement train logic.
For algorithms pls refer prolib module
Finally, this script performs aggregation across folds models, collect single OOF and do single test prediction as avg
This is the longest running part
If some jobs are failed, you can re-run the script from start and it skips completed jobs
"""
import argparse
import subprocess
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import Queue
from itertools import product
from joblib import Parallel, delayed


parser = argparse.ArgumentParser()

parser.add_argument(
    '-d', '--DEVICES',nargs='*', type=str, default=['0', '1', '2', '3', '4', '5', '6', '7']
)

LIN_CONFIGS = [
    # '/kaggle/working/CAFA6-updates/configs/lin_debug.yaml',  # single config to debug
    # tasks based on T5/ESM embeddings
    "lin_cafa5_t5_cafa6-sparse-labels13500_cond.yaml",
    "lin_cafa5_esm2S1280_cafa6-sparse-labels13500_cond.yaml",
    "lin_cafa5_t5esm2S1280_cafa6-sparse-labels13500_cond.yaml",
    "lin_cafa6_t5_cafa6-sparse-labels13500_cond.yaml",
    "lin_cafa6_esm2S1280_cafa6-sparse-labels13500_cond.yaml",
    "lin_cafa6_t5esm2S1280_cafa6-sparse-labels13500_cond.yaml",
    "lin_cafa5_t5_cafa6-sparse-labels13500_raw.yaml",
    "lin_cafa5_esm2S1280_cafa6-sparse-labels13500_raw.yaml",
    "lin_cafa5_t5esm2S1280_cafa6-sparse-labels13500_raw.yaml",
    "lin_cafa6_t5_cafa6-sparse-labels13500_raw.yaml",
    "lin_cafa6_esm2S1280_cafa6-sparse-labels13500_raw.yaml",
    "lin_cafa6_t5esm2S1280_cafa6-sparse-labels13500_raw.yaml",

    # tasks based on SVD cross embeddings
    'lin_cafa6_svd512bpsvd512mf_cafa6-sparse-labels_raw.yaml',
    'lin_cafa6_svd512bpsvd512cc_cafa6-sparse-labels_raw.yaml',
    'lin_cafa6_svd512mfsvd512cc_cafa6-sparse-labels_raw.yaml',
    'lin_cafa6_svd512mfsvd512cc_cafa6-sparse-labels_cond.yaml',
    'lin_cafa6_svd512bpsvd512mf_cafa6-sparse-labels_cond.yaml',
    'lin_cafa6_svd512bpsvd512cc_cafa6-sparse-labels_cond.yaml',

]

PB_CONFIGS = [
    # '/kaggle/working/CAFA6-updates/configs/pb_debug.yaml', # single config to debug
    # tasks based on T5/ESM embeddings
    "pb_cafa5_t5_cafa6-sparse-labels4500_cond.yaml",
    "pb_cafa5_esm2S1280_cafa6-sparse-labels4500_cond.yaml",
    "pb_cafa5_t5esm2S1280_cafa6-sparse-labels4500_cond.yaml",
    "pb_cafa6_t5_cafa6-sparse-labels4500_cond.yaml",
    "pb_cafa6_esm2S1280_cafa6-sparse-labels4500_cond.yaml",
    "pb_cafa6_t5esm2S1280_cafa6-sparse-labels4500_cond.yaml",
    "pb_cafa5_t5_cafa6-sparse-labels4500_raw.yaml",
    "pb_cafa5_esm2S1280_cafa6-sparse-labels4500_raw.yaml",
    "pb_cafa5_t5esm2S1280_cafa6-sparse-labels4500_raw.yaml",
    "pb_cafa6_t5_cafa6-sparse-labels4500_raw.yaml",
    "pb_cafa6_esm2S1280_cafa6-sparse-labels4500_raw.yaml",
    "pb_cafa6_t5esm2S1280_cafa6-sparse-labels4500_raw.yaml",

    # tasks based on SVD cross embeddings
    'pb_cafa6_svd512mfsvd512cc_cafa6-sparse-labels_raw.yaml',
    'pb_cafa6_svd512bpsvd512cc_cafa6-sparse-labels_raw.yaml',
    'pb_cafa6_svd512bpsvd512mf_cafa6-sparse-labels_raw.yaml',
    'pb_cafa6_svd512bpsvd512mf_cafa6-sparse-labels_cond.yaml',
    'pb_cafa6_svd512mfsvd512cc_cafa6-sparse-labels_cond.yaml',
    'pb_cafa6_svd512bpsvd512cc_cafa6-sparse-labels_cond.yaml',

    # tasks based on T5+tfidf
    'pb_cafa6_t5tfidf_uniprot-sparse-labels4500_raw.yaml',
]

NN_CONFIGS = [
    'nn_cafa6_t5esm2S1280_cafa6-sparse-labels13500_raw.yaml',
    'nn_cafa6_t5esm2S1280_cafa6-sparse-labels13500_cond.yaml'
]


def run_task(task, ):
    # check if already done:
    params = dict(map(lambda x: x.split(' ', 1), task.split('--')))
    with open(params['config'], 'r') as f:
        cfg = yaml.safe_load(f)

    dump_path = Path(params['output'].strip()) / cfg['name'].strip() / 'dumps'
    path = dump_path / f'model_{params["fold-id"].strip()}.pkl'
    print(path)

    if path.exists():
        return

    gpu = QUEUE.get()
    task = task + f' --device {gpu}'

    print('EXECUTE COMMAND: ', task, end='\n\n')

    dump_path.mkdir(parents=True, exist_ok=True)

    stdout = dump_path / f'stdout_{params["fold-id"].strip()}.log'
    stderr = dump_path / f'stderr_{params["fold-id"].strip()}.log'

    with open(stdout, 'wb') as f0, open(stderr, 'wb') as f1:

        try:
            log = subprocess.run(task, shell=True, stdout=f0, stderr=f1)
        except subprocess.CalledProcessError:
            pass

    QUEUE.put(gpu)
    return


if __name__ == '__main__':
    args = parser.parse_args()

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()
    helpers_path = data_path / 'helpers'
    cafa6_path = data_path / 'cafa-6-protein-function-prediction'

    embed_path = Path(config['embed_path']).resolve()
    models_path = Path(config['models_path']).resolve()
    RAPIDS_ENV = config['rapids-env']

    # -----------------------------------
    # COLLECT TASKS LIST
    # -----------------------------------
    QUEUE = Queue(maxsize=len(args.devices))
    for i in args.devices:
        QUEUE.put(i)

    ENV_PARAMS = {
        # config
        # fold_id
        # device
        'fasta': helpers_path / 'fasta',
        'embed_path': embed_path,
        'output': models_path,
        'graph_path': cafa6_path / 'Train/go-basic.obo',
        'freq_labels': data_path / 'train_terms.tsv',
        'sparse_labels': helpers_path,
    }

    params_row = ' '.join(['--' + x.replace('_', '-') + f' {ENV_PARAMS[x]}'  for x in ENV_PARAMS])
    configs_path = Path(__file__).parent.parent / 'configs'

    TASKS = []

    # linear models
    task_template = f'{RAPIDS_ENV} ./protlib/scripts/train_lin_c6.py ' + params_row + ' --fold-id {f} --config {c}'
    TASKS.extend(
        task_template.format(f=f, c=configs_path / c) for f, c in product(range(5), LIN_CONFIGS)
    )

    # py-boost models
    task_template = f'{RAPIDS_ENV} ./protlib/scripts/train_pb_c6.py ' + params_row + ' --fold-id {f} --config {c}'
    TASKS.extend(
        task_template.format(f=f, c=configs_path / c) for f, c in product(range(5), PB_CONFIGS)
    )

    # nn models
    nn_template = f'{RAPIDS_ENV} ./protlib/scripts/train_nn.py ' + params_row + ' --fold-id {f} --config {c}'
    TASKS.extend(
        task_template.format(f=f, c=configs_path / c) for f, c in product(range(5), NN_CONFIGS)
    )
    # -----------------------------------
    # Get a rest for a few days ..
    # -----------------------------------
    with Parallel(n_jobs=len(args.devices), backend="threading") as p:
        p(delayed(run_task)(x) for x in tqdm(TASKS))

    # -----------------------------------
    # TRAIN: Aggregate models across folds
    # -----------------------------------

    index = pd.read_feather(
        data_path / 'helpers/fasta/train_seq.feather',
        # os.path.join(PATH_TO_DATASET, 'helpers/fasta/train_seq.feather'),
        columns=['EntryID']
    )['EntryID'].values

    for path in models_path.glob('*/oof_pred'):
    # for path in glob.glob('./models/*/oof_pred'):
        print(path)

        df = pd.read_parquet(path)
        pred_agg = df.groupby('EntryID').mean().loc[index]

        pred_agg = (pred_agg * 255).clip(0, 255).astype(np.uint8)
        pred_agg.reset_index(inplace=True)

        # path is parquet path. We need folder, so, go to parent
        pred_path = path.parent / 'predictions'
        pred_path.mkdir(parents=True, exist_ok=True)

        # os.makedirs(os.path.join(os.path.dirname(path), 'predictions'), exist_ok=True)

        pred_agg.to_parquet(
            pred_path / 'oof_pred.parquet',
            # os.path.join(os.path.dirname(path), 'predictions', 'oof_pred.parquet'),
            index=False
        )

    # -----------------------------------
    # OLD TRAIN: Aggregate models across folds
    # -----------------------------------

    index = pd.read_feather(
        data_path / 'helpers/fasta/old_train_seq.feather',
        # os.path.join(PATH_TO_DATASET, 'helpers/fasta/old_train_seq.feather'),
        columns=['EntryID']
    )['EntryID'].values

    for path in models_path.glob('*/oof_old_pred'):
    # for path in glob.glob('./models/*/oof_old_pred'):
        print(path)

        df = pd.read_parquet(path)
        pred_agg = df.groupby('EntryID').mean().loc[index]

        pred_agg = (pred_agg * 255).clip(0, 255).astype(np.uint8)
        pred_agg.reset_index(inplace=True)

        pred_path = path.parent / 'predictions'
        pred_path.mkdir(parents=True, exist_ok=True)

        # os.makedirs(os.path.join(os.path.dirname(path), 'predictions'), exist_ok=True)

        pred_agg.to_parquet(
            pred_path / 'oof_old_pred.parquet',
            # os.path.join(os.path.dirname(path), 'predictions', 'oof_old_pred.parquet'),
            index=False
        )

    # -----------------------------------
    # TEST: Aggregate models across folds
    # -----------------------------------

    BATCH_SIZE = 10000

    index = pd.read_feather(
        data_path / 'helpers/fasta/test_seq.feather',
        # os.path.join(PATH_TO_DATASET, 'helpers/fasta/test_seq.feather'),
        columns=['EntryID']
    )['EntryID'].values

    for path in models_path.glob('*/test_pred'):
    # for path in glob.glob('./models/*/test_pred'):
        print(path)

        pred_agg = None

        # partitions = glob.glob(os.path.join(path, '*.parquet'))
        # for part in partitions:

        n_folds = 0
        for part in path.glob('*.parquet'):
            n_folds += 1
            df = pd.read_parquet(part)

            assert (df['EntryID'].values == index).all()
            terms = df.columns.drop('EntryID').tolist()

            if pred_agg is None:
                pred_agg = df[terms].values
            else:
                pred_agg += df[terms].values

        pred_agg /= n_folds # len(partitions)

        pred_agg = pd.DataFrame(pred_agg, columns=terms, index=pd.Index(index, name='EntryID'))
        pred_agg = (pred_agg * 255).clip(0, 255).astype(np.uint8)
        pred_agg.reset_index(inplace=True)

        pred_path = path.parent / 'predictions/test'
        pred_path.mkdir(parents=True, exist_ok=True)

        # os.makedirs(os.path.join(os.path.dirname(path), 'predictions/test'), exist_ok=True)

        for n, i in enumerate(range(0, len(pred_agg), BATCH_SIZE)):
            pred_agg[i: i + BATCH_SIZE].to_parquet(
                pred_path / f'part_{str(n).zfill(2)}.parquet',
                # os.path.join(os.path.dirname(path), 'predictions/test', f'part_{str(n).zfill(2)}.parquet'),

                index=False
            )


