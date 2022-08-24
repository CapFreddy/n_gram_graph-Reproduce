from __future__ import print_function

import os
import argparse
import numpy as np
import json

from MoleculeBench import train_val_test_split, dataset_info

from n_gram_graph.util import *


task_dataset = {
    task: dataset 
    for dataset in ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
    for task in dataset_info(dataset).task_columns
}


def run_n_gram_xgb(task, data_seed, run_seed, n_gram_num, embedding_dimension):
    from n_gram_graph.model.xgboost_classification import XGBoostClassification

    dataset = task_dataset[task]
    info = dataset_info(dataset)
    config_json_file = f'./hyper/n_gram_xgb/{dataset}/{task}.json'
    file_path = f'./reproduce/saved_models/embedding/{dataset}/data_seed={data_seed}/run_seed={run_seed}/grammed_cbow_{embedding_dimension}_graph.npz' \
                if info.splitting == 'random' else \
                f'./reproduce/saved_models/embedding/{dataset}/run_seed={run_seed}/grammed_cbow_{embedding_dimension}_graph.npz'
    weight_file = f'./reproduce/saved_models/n_gram_xgb/{dataset}/data_seed={data_seed}/run_seed={run_seed}/saved_model/{task}/model.joblib' \
                  if info.splitting == 'random' else \
                  f'./reproduce/saved_models/n_gram_xgb/{dataset}/run_seed={run_seed}/saved_model/{task}/model.joblib'
    result_path = f'./reproduce/saved_models/n_gram_xgb/{dataset}/data_seed={data_seed}/run_seed={run_seed}/saved_model/{task}/test_roc.txt' \
                  if info.splitting == 'random' else \
                  f'./reproduce/saved_models/n_gram_xgb/{dataset}/run_seed={run_seed}/saved_model/{task}/test_roc.txt'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(config_json_file, 'r') as f:
        conf = json.load(f)

    X, y = extract_feature_and_label_npy(file_path,
                                         feature_name='embedded_graph_matrix_list',
                                         label_index=info.task_columns.index(task),
                                         n_gram_num=n_gram_num)
    train_indices, _, test_indices = train_val_test_split(dataset, return_indices=True, random_state=data_seed)
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    X_train, y_train = X_train[~np.isnan(y_train)], y_train[~np.isnan(y_train)]
    X_test, y_test = X_test[~np.isnan(y_test)], y_test[~np.isnan(y_test)]

    print('done data preparation')
    print(f'Train {X_train.shape} {y_train.shape}')
    print(f'Test {X_test.shape} {y_test.shape}')

    task = XGBoostClassification(conf=conf)
    print(X_train.shape, '\t', y_train.shape, '\t', X_test.shape, '\t', y_test.shape)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
    test_roc = task.eval_with_existing(X_train, y_train, X_test, y_test, weight_file)

    with open(result_path, 'w') as fout:
        fout.write(str(test_roc))

    return


def run_n_gram_rf(task, data_seed, run_seed, n_gram_num, embedding_dimension):
    from n_gram_graph.model.random_forest_classification import RandomForestClassification

    dataset = task_dataset[task]
    info = dataset_info(dataset)
    config_json_file = f'./hyper/n_gram_rf/{dataset}/{task}.json'
    file_path = f'./reproduce/saved_models/embedding/{dataset}/data_seed={data_seed}/run_seed={run_seed}/grammed_cbow_{embedding_dimension}_graph.npz' \
                if info.splitting == 'random' else \
                f'./reproduce/saved_models/embedding/{dataset}/run_seed={run_seed}/grammed_cbow_{embedding_dimension}_graph.npz'
    weight_file = f'./reproduce/saved_models/n_gram_rf/{dataset}/data_seed={data_seed}/run_seed={run_seed}/saved_model/{task}/model.joblib' \
                  if info.splitting == 'random' else \
                  f'./reproduce/saved_models/n_gram_rf/{dataset}/run_seed={run_seed}/saved_model/{task}/model.joblib'
    test_path = f'./reproduce/saved_models/n_gram_rf/{dataset}/data_seed={data_seed}/run_seed={run_seed}/saved_model/{task}/test_roc.txt' \
                if info.splitting == 'random' else \
                f'./reproduce/saved_models/n_gram_rf/{dataset}/run_seed={run_seed}/saved_model/{task}/test_roc.txt'
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    with open(config_json_file, 'r') as f:
        conf = json.load(f)

    X, y = extract_feature_and_label_npy(file_path,
                                         feature_name='embedded_graph_matrix_list',
                                         label_index=info.task_columns.index(task),
                                         n_gram_num=n_gram_num)
    train_indices, _, test_indices = train_val_test_split(dataset, return_indices=True, random_state=data_seed)
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    X_train, y_train = X_train[~np.isnan(y_train)], y_train[~np.isnan(y_train)]
    X_test, y_test = X_test[~np.isnan(y_test)], y_test[~np.isnan(y_test)]

    print('done data preparation')
    print(f'Train {X_train.shape} {y_train.shape}')
    print(f'Test {X_test.shape} {y_test.shape}')

    task = RandomForestClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_test, y_test, weight_file)
    test_roc = task.eval_with_existing(X_train, y_train, X_test, y_test, weight_file)

    with open(test_path, 'w') as fout:
        fout.write(str(test_roc))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--run_seed', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['n_gram_xgb', 'n_gram_rf'], required=True)
    parser.add_argument('--n_gram_num', type=int, default=6)
    parser.add_argument('--embedding_dimension', type=int, default=100)
    given_args = parser.parse_args()

    task = given_args.task
    data_seed = given_args.data_seed
    run_seed = given_args.run_seed
    model = given_args.model
    n_gram_num = given_args.n_gram_num
    embedding_dimension = given_args.embedding_dimension

    if model == 'n_gram_xgb':
        run_n_gram_xgb(task, data_seed, run_seed, n_gram_num, embedding_dimension)
    elif model == 'n_gram_rf':
        run_n_gram_rf(task, data_seed, run_seed, n_gram_num, embedding_dimension)
    else:
        raise Exception('No such model! Should be among [{}, {}].'.format(
            'n_gram_xgb',
            'n_gram_rf'
        ))
