"""Propagate extra evidence/type features
"""


import subprocess
import yaml
from pathlib import Path

if __name__ == '__main__':

    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )
    data_path = Path(config['data_path']).resolve()

    features_path = data_path / 'uniprot/raw/features'
    uni_codes_path = data_path / 'cafa6-uniprot-codes'
    raw_codes_path = uni_codes_path / 'raw'

    cafa6_path = data_path / 'cafa-6-protein-function-prediction'
    graph_path = cafa6_path / 'Train/go-basic.obo'
    RAPIDS_ENV = config['rapids-env']

    # -----------------------------------
    # Build features from auto labels (counts)
    # -----------------------------------

    task = f"""{RAPIDS_ENV} ./protlib/scripts/prop_tsv_sum.py \
        --path {features_path / 'train_auto.tsv'} \
        --graph {graph_path} \
        --output {uni_codes_path / 'train_auto.tsv'} \
        --device 0 \
        --n-props 2 \
        --batch-size 30000 \
        --batch-inner 5000
    """
    subprocess.run(task, shell=True)

    task = f"""{RAPIDS_ENV} ./protlib/scripts/prop_tsv_sum.py \
        --path {features_path / 'old_train_auto.tsv'} \
        --graph {graph_path} \
        --output {uni_codes_path / 'old_train_auto.tsv'}  \
        --device 0 \
        --n-props 2 \
        --batch-size 30000 \
        --batch-inner 5000
    """
    subprocess.run(task, shell=True)

    task = f"""!{RAPIDS_ENV} ./protlib/scripts/prop_tsv_sum.py \
        --path {features_path / 'test_auto228.tsv'} \
        --graph {graph_path} \
        --output {uni_codes_path / 'test_auto228.tsv'} \
        --device 0 \
        --n-props 2 \
        --batch-size 30000 \
        --batch-inner 5000
    """
    subprocess.run(task, shell=True)

    # -----------------------------------
    # Build features from evidences and types
    # -----------------------------------

    for inp in raw_codes_path.glob('*/*.tsv'):
        outp = list(inp.parts)
        outp[-3] = 'prop' # replace raw with prop
        outp = Path(*outp)

        print(outp, )

        task = f"""{RAPIDS_ENV} ./CAFA6-updates/protlib/scripts/prop_tsv_sum.py \
            --path {inp} \
            --graph {graph_path} \
            --output {outp} \
            --device 0 \
            --n-props 2 \
            --batch-size 30000 \
            --batch-inner 5000
        """
        subprocess.run(task, shell=True)