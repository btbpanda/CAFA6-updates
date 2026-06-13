"""Propagate all term information we have according to CAFA propagation rules
Building sparkse
"""


import subprocess
import yaml
from pathlib import Path

if __name__ == '__main__':

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()
    uniprot_raw_path = data_path / 'uniprot/raw'
    cafa6_path = data_path / 'cafa-6-protein-function-prediction'
    graph_path = cafa6_path / 'Train/go-basic.obo'
    RAPIDS_ENV = config['rapids-env']

    # -----------------------------------
    # Build CAFA 6 target as propagated
    # -----------------------------------

    task = f"""{RAPIDS_ENV} ./protlib/scripts/prop_tsv_c6.py \
        --path {cafa6_path / 'Train/train_terms.tsv'} \
        --graph {graph_path} \
        --output {data_path / 'train_terms.tsv'} \
        --device 0 \
        --n-props 2 \
        --direction backward \
        --func max \
        --batch-size 30000 \
        --batch-inner 5000
    """
    subprocess.run(task, shell=True)

    # -----------------------------------
    # Propagate all GOA data
    # -----------------------------------

    for inp in uniprot_raw_path.glob('*/*.tsv'):
        outp = list(inp.parts)
        outp[-3] = 'prop' # replace raw
        outp = Path(*outp)
        direction = 'forward' if inp.name.startswith('not') else 'backward'

        print(outp, direction)

        task = f"""{RAPIDS_ENV} ./protlib/scripts/prop_tsv_c6.py \
            --path {inp} \
            --graph {graph_path} \
            --output {outp} \
            --device 0 \
            --n-props 2 \
            --direction {direction} \
            --func max \
            --batch-size 30000 \
            --batch-inner 5000
        """
        subprocess.run(task, shell=True)

    # -----------------------------------
    # Build sparse multilabel target
    # -----------------------------------

    task = f"""{RAPIDS_ENV} ./protlib/scripts/create_helpers.py \
        --output {data_path / 'helpers/cafa6-sparse-labels'} \
        --terms {data_path / 'train_terms.tsv'} \
        --graph {graph_path} \
        --ia {cafa6_path / 'IA.tsv'} \
        --seq {data_path / 'helpers/fasta/train_seq.feather'} \
        --batch-size 10000 
    """
    subprocess.run(task, shell=True)

    task = f"""{RAPIDS_ENV} ./protlib/scripts/create_helpers.py \
        --output {data_path / 'helpers/uniprot-old-sparse-labels'} \
        --terms {data_path / 'uniprot/prop/labels/old_train_terms.tsv'} \
        --graph {graph_path} \
        --ia {cafa6_path / 'IA.tsv'}  \
        --seq {data_path / 'helpers/fasta/old_train_seq.feather'} \
        --batch-size 10000 
    """
    subprocess.run(task, shell=True)










