import argparse
import os
import sys
import pandas as pd
import joblib
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--fasta', type=str)
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-e', '--embed_path', type=str)
parser.add_argument('-g', '--graph-path', type=str)
parser.add_argument('-fl', '--freq-labels', type=str)
parser.add_argument('-sl', '--sparse-labels', type=str)
parser.add_argument('-f', '--fold-id', type=int)
parser.add_argument('-d', '--device', type=int)


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

    # Optional: set the device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    try:
        from protlib.metric import obo_parser, Graph, get_topk_targets
        from protlib.models.prepocess import get_features, get_folds, get_targets_from_parquet
        from protlib.models.gbdt import BCEWithNaNLoss, BCEwithNaNMetric

    except ImportError:
        print('Alarm')
        pass

    from py_boost import GradientBoosting
    from py_boost.multioutput.sketching import RandomProjectionSketch

    ################################
    # PREPARE
    ################################

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    ontologies = []
    for ns, terms_dict in obo_parser(args.graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    # select required targets

    split = [config['bp'], config['mf'], config['cc']]
    cols = []

    for n, i in enumerate(split):
        cols.extend(get_topk_targets(
            ontologies[n],
            i,
            train_path=args.freq_labels
        ))

    ################################
    # READ TARGETS
    ################################

    Y_train = get_targets_from_parquet(
        os.path.join(args.sparse_labels, config['train_labels'], ),
        ontologies,
        split=split,
        ids=cols,
        fillna=not config['conditional']
    )

    Y_old = get_targets_from_parquet(
        os.path.join(args.sparse_labels, config['old_train_labels'], ),
        ontologies,
        split=split,
        ids=cols,
        fillna=not config['conditional']
    )

    Y_train = pd.concat([Y_train, Y_old, ], axis=0, ignore_index=True)

    Y_train, prot_names = Y_train.values, Y_train.columns.tolist()

    ################################
    # GET FEATURES
    ################################
    # main train file
    train = pd.read_feather(
        os.path.join(args.fasta, 'train_seq.feather')
    )
    train['is_cafa6'] = True
    cafa6_size = train.shape[0]

    X_train = get_features(
        train,
        embed_path=args.embed_path,
        prefix='train',
        embed_list=config['embeds'],
        tax_list=config['tax_list']
    )

    old_train = pd.read_feather(
        os.path.join(args.fasta, 'old_train_seq.feather')
    )
    old_train['is_cafa6'] = False
    cafa_old_size = old_train.shape[0]

    X_old = get_features(
        old_train,
        embed_path=args.embed_path,
        prefix='old_train',
        embed_list=config['embeds'],
        tax_list=config['tax_list']
    )

    # joint dataset
    train = pd.concat([train, old_train, ], ignore_index=True)
    train['fold'] = get_folds(train['seq'].apply(murmurhash3_32, seed=42))  # hash to speed up

    X_train = np.concatenate([X_train, X_old, ], axis=0)

    # test file
    test = pd.read_feather(
        os.path.join(args.fasta, 'test_seq.feather')
    )
    X_test = get_features(
        test,
        embed_path=args.embed_path,
        prefix='test',
        embed_list=config['embeds'],
        tax_list=config['tax_list']
    )

    ################################
    # DUMP METADATA
    ################################

    os.makedirs(
        os.path.join(args.output, config['name'], 'dumps'),
        exist_ok=True
    )

    # oof pred
    os.makedirs(
        os.path.join(args.output, config['name'], 'oof_pred'),
        exist_ok=True
    )

    # old oof pred
    os.makedirs(
        os.path.join(args.output, config['name'], 'oof_old_pred'),
        exist_ok=True
    )

    # test pred
    os.makedirs(
        os.path.join(args.output, config['name'], 'test_pred'),
        exist_ok=True
    )

    joblib.dump(
        prot_names,
        os.path.join(args.output, config['name'], 'dumps', 'prot_names.pkl')
    )

    joblib.dump(
        cols,
        os.path.join(args.output, config['name'], 'dumps', 'prot_ids.pkl')
    )

    with open(os.path.join(args.output, config['name'], 'dumps', 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

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
            prior = (prior * wc6 + prior_old * wc5) / (wc6 + wc5).clip(0.1, None)
            # print(prior)

            np.save(os.path.join(args.output, config['name'], 'dumps', f'{ns}.npy'), prior, )

    ################################
    # FIT PREDICT
    ################################
    train_sl = train['fold'] != args.fold_id
    valid_sl = train['fold'] == args.fold_id
    pred_sl = valid_sl

    if config['train_data'] == 'cafa6':
        train_sl = train_sl & train['is_cafa6']

    if config['valid_data'] == 'cafa6':
        valid_sl = valid_sl & train['is_cafa6']

    train_sl, valid_sl, pred_sl = np.nonzero(train_sl)[0], np.nonzero(valid_sl)[0], np.nonzero(pred_sl)[0]

    params = config['train_params'].copy()
    params['multioutput_sketch'] = RandomProjectionSketch(params.pop('sketch_size'))

    model = GradientBoosting(
        BCEWithNaNLoss(), BCEwithNaNMetric(),
        **params
    )

    model.fit(
        X_train[train_sl], Y_train[train_sl],
        eval_sets=[{'X': X_train[valid_sl], 'y': Y_train[valid_sl]}]
    )

    joblib.dump(
        model,
        os.path.join(args.output, config['name'], 'dumps', f'model_{args.fold_id}.pkl'),
    )

    oof_pred = model.predict(X_train[pred_sl], batch_size=5000)
    test_pred = model.predict(X_test, batch_size=5000)

    ################################
    # DUMP PREDS
    ################################

    test_pred = pd.DataFrame(test_pred, columns=prot_names)
    test_pred['EntryID'] = test['EntryID'].values

    test_pred.to_parquet(
        os.path.join(args.output, config['name'], 'test_pred', f'fold_{args.fold_id}.parquet'),
        index=False,
    )

    oof_pred = pd.DataFrame(oof_pred, columns=prot_names)
    oof_pred['EntryID'] = train['EntryID'].values[pred_sl]
    oof_pred['is_cafa6'] = train['is_cafa6'].values[pred_sl]

    oof_pred.query('is_cafa6').drop('is_cafa6', axis=1).to_parquet(
        os.path.join(args.output, config['name'], 'oof_pred', f'fold_{args.fold_id}.parquet'),
        index=False,
    )

    oof_pred.query('~is_cafa6').drop('is_cafa6', axis=1).to_parquet(
        os.path.join(args.output, config['name'], 'oof_old_pred', f'fold_{args.fold_id}.parquet'),
        index=False,
    )
