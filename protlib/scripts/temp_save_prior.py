import argparse
import os
import sys
import pandas as pd
import joblib
import numpy as np
import yaml
from sklearn.utils import murmurhash3_32

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()

parser.add_argument('-o', '--output', type=str)
parser.add_argument('-t', '--fasta', type=str)
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-e', '--embed-path', type=str)
parser.add_argument('-g', '--graph-path', type=str)
parser.add_argument('-fl', '--freq-labels', type=str)
parser.add_argument('-sl', '--sparse-labels', type=str)
parser.add_argument('-f', '--fold-id', type=int)
parser.add_argument('-d', '--device', type=str)


def get_sample_prior(source, cond):
    prior = np.load(
        os.path.join(source, 'prior.npy')
    )
    nulls = np.load(
        os.path.join(source, 'nulls.npy')
    )

    if cond:
        prior, denom = prior, (1 - nulls)
    else:
        prior, denom = prior * (1 - nulls), 1

    return prior, denom


if __name__ == '__main__':

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train = pd.read_feather(
        os.path.join(args.fasta, 'train_seq.feather')
    )
    train['is_cafa6'] = True
    cafa6_size = train.shape[0]

    old_train = pd.read_feather(
        os.path.join(args.fasta, 'old_train_seq.feather')
    )
    old_train['is_cafa6'] = False
    cafa_old_size = old_train.shape[0]

    # save priors

    for ns in [
        'biological_process',
        'molecular_function',
        'cellular_component'
    ]:
        prior, denom = get_sample_prior(
            os.path.join(args.sparse_labels, config['train_labels'], ns, ), config['conditional']
        )

        # print(prior)

        if config['train_data'] != 'cafa6':
            prior_old, denom_old = get_sample_prior(
                os.path.join(args.sparse_labels, config['old_train_labels'], ns, ), config['conditional']
            )
            # print(prior_old)

            wc6, wc5 = denom * cafa6_size, denom_old * cafa_old_size
            prior = (prior * wc6 + prior_old * wc5) / np.clip(wc6 + wc5, 0.1, None)
            # print(prior)

        np.save(os.path.join(args.output, config['name'], 'dumps', f'{ns}.npy'), prior, )
