import os

import numpy as np
from prettytable import PrettyTable

from MoleculeBench import dataset_info


columns = ['Model', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace']
rows = []


for model in ['n_gram_rf', 'n_gram_xgb']:
    row = [model]
    for dataset in columns[1:]:
        info = dataset_info(dataset)
        result_seeds = []
        for run_seed in [0, 1, 2]:
            result_tasks = []
            missing_tasks = []
            tasks = info.task_columns
            for task in tasks:
                num_nans = 0
                result_path = f'./reproduce/saved_models/{model}/{dataset}/data_seed=42/run_seed={run_seed}/saved_model/{task}/test_roc.txt' \
                              if info.splitting == 'random' else \
                              f'./reproduce/saved_models/{model}/{dataset}/run_seed={run_seed}/saved_model/{task}/test_roc.txt'

                if not os.path.exists(result_path):
                    missing_tasks.append(task)
                    continue

                with open(result_path, 'r') as fin:
                    content = fin.readline()
                    if 'nan' not in content:
                        result_tasks.append(eval(content) * 100)

            if len(missing_tasks) != 0:
                print(f'{len(missing_tasks)} of {len(tasks)} tasks are missing for {model}-{dataset}-{run_seed}!')
                continue

            result_tasks = np.array(result_tasks).mean()
            result_seeds.append(result_tasks)

        result_seeds = np.array(result_seeds)
        result_seeds = f'%.1f \u00B1 %.1f ({len(result_seeds)})' % (result_seeds.mean(), result_seeds.std())
        row.append(result_seeds)

    rows.append(row)


table = PrettyTable()
table.field_names = columns
table.add_rows(rows)
print(table)
