import os
from argparse import ArgumentParser

from MoleculeBench import dataset_info

from datasets.data_preprocess import extract_graph


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace'], required=True)
    args = parser.parse_args()

    out_file_path = f'./reproduce/dataset/{args.dataset}_graph.npz'
    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

    info = dataset_info(args.dataset)
    extract_graph(data_path=info.filtered_path,
                  out_file_path=out_file_path,
                  max_atom_num=-1,
                  smiles_column=info.smiles_column,
                  label_name=info.task_columns)
