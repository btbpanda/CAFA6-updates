"""This script is the starting point, it downloads all data and stores into the folder specified in global config

"""
import subprocess
import yaml
from pathlib import Path


if __name__ == '__main__':
    config = yaml.safe_load(
        Path('./config.yaml').read_text()
    )

    data_path = Path(config['data_path']).resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    tasks = [
        # read actual competition data
        f"""rm -rf {data_path / 'cafa-6-protein-function-prediction'}
        mkdir {data_path / 'cafa-6-protein-function-prediction'}
        kaggle competitions download -c cafa-6-protein-function-prediction -p {data_path}
        unzip {data_path / 'cafa-6-protein-function-prediction.zip'} -d {data_path / 'cafa-6-protein-function-prediction'}
        """,

        # read actual competition data
        f"""rm -rf {data_path / 'cafa-5-protein-function-prediction'}
        mkdir {data_path / 'cafa-5-protein-function-prediction'}
        kaggle competitions download -c cafa-5-protein-function-prediction -p {data_path}
        unzip {data_path / 'cafa-5-protein-function-prediction.zip'} -d {data_path / 'cafa-5-protein-function-prediction'}
        """,

        # get GOA annotations version 226 (available at competition start)
        f"""wget http://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/goa_uniprot_all.gaf.226.gz \
            -O {data_path / 'goa_uniprot_all.gaf.226.gz'}
        """,

        # get GOA annotations version 228 (the latest release during the competition)
        f"""wget http://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/goa_uniprot_all.gaf.228.gz \
            -O {data_path / 'goa_uniprot_all.gaf.228.gz'}
        """
    ]

    procs = []

    for task in tasks:
        proc = subprocess.Popen(task, shell=True)
        procs.append(proc)

    for proc in procs:
        proc.wait()