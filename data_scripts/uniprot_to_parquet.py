"""This script is simpy parallel launcher of raw_uniprot to handle both 226/228 in parallel

"""
import sys
import subprocess

if __name__ == '__main__':

    python = sys.executable
    tasks = [
        f'{python} ./data_scripts/raw_uniprot.py --file goa_uniprot_all.gaf.226.gz --output raw_uniprot226.parquet',
        f'{python} ./data_scripts/raw_uniprot.py --file goa_uniprot_all.gaf.228.gz --output raw_uniprot228.parquet'
    ]

    procs = []

    for task in tasks:
        proc = subprocess.Popen(task, shell=True)
        procs.append(proc)

    for proc in procs:
        proc.wait()

        