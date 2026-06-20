"""This is DL embeds inference. Script only manages training jobs, not implement train logic.
For algorithms pls refer embeddings module
"""
import sys
import subprocess
import tqdm
import yaml

from pathlib import Path
from multiprocessing import Queue
from joblib import Parallel, delayed


def run_task(task, ):

    gpu = QUEUE.get()
    task = task + f' --device {gpu}'

    print('EXECUTE COMMAND: ', task, end='\n\n')
    subprocess.run(task, shell=True, )

    QUEUE.put(gpu)
    return


if __name__ == '__main__':

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )

    data_path = Path(config['data_path']).resolve()
    fasta_path = data_path / 'helpers/fasta'
    embed_path = Path(config['embed_path']).resolve()
    PYTHON = sys.executable
    DEVICES = config['devices']

    # -----------------------------------
    # COLLECT TASKS LIST
    # -----------------------------------
    QUEUE = Queue(maxsize=len(DEVICES))
    for i in DEVICES:
        QUEUE.put(i)

    ENV_PARAMS = {
        # file
        # device
        'fasta': fasta_path,
        'embed_path': embed_path,
    }

    params_row = ' '.join(['--' + x.replace('_', '-') + f' {ENV_PARAMS[x]}' for x in ENV_PARAMS])

    TASKS = []

    # T5
    task_template = f'{PYTHON} ./embeddings/t5.py ' + params_row + ' --file {f}'
    TASKS.extend(
        task_template.format(f=f, ) for f in ['train', 'old_train', 'test']
    )

    # ESM
    task_template = f'{PYTHON} ./embeddings/esm2sm.py ' + params_row + ' --file {f}'
    TASKS.extend(
        task_template.format(f=f, ) for f in ['train', 'old_train', 'test']
    )

    # -----------------------------------
    # Get a rest for a few days ..
    # -----------------------------------
    with Parallel(n_jobs=len(DEVICES), backend="threading") as p:
        p(delayed(run_task)(x) for x in tqdm.tqdm(TASKS))