import os
from argparse import ArgumentParser

from tqdm import tqdm

from MoleculeBench import dataset_info

from main_classification import run_n_gram_xgb, run_n_gram_rf


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace'], required=True)
    parser.add_argument('--run_seed', type=int, default=0)
    parser.add_argument('--model', type=str, choices=['n_gram_xgb', 'n_gram_rf'], required=True)
    args = parser.parse_args()

    dataset = args.dataset
    run_seed = args.run_seed
    model = args.model

    info = dataset_info(dataset)
    for task in tqdm(info.task_columns):
        result_path = f'./reproduce/saved_models/{model}/{dataset}/data_seed=42/run_seed={run_seed}/saved_model/{task}/test_roc.txt' \
                      if info.splitting == 'random' else \
                      f'./reproduce/saved_models/{model}/{dataset}/run_seed={run_seed}/saved_model/{task}/test_roc.txt'

        if os.path.exists(result_path):
            print(f'Skip task {task} since it has been run.')
            continue

        if args.model == 'n_gram_xgb':
            run_n_gram_xgb(task, 42, args.run_seed, 6, 100)
        else:
            run_n_gram_rf(task, 42, args.run_seed, 6, 100)
