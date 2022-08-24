from __future__ import print_function

import argparse
import time
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np

from MoleculeBench import dataset_info

from node_embedding import CBoW


mode2task_list = {
    'qm8': [
        'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
        'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'
    ],
    'qm9': [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv', 'u0', 'u298', 'h298', 'g298',
    ]
}


def get_data(data_path):
    data = np.load(data_path)
    print(data.keys())
    max_atom_num = data['max_atom_num']
    max_degree = data['max_degree']
    adjacent_matrix_list = data['adjacent_matrix_list']
    distance_matrix_list = data['distance_matrix_list']
    bond_attribute_matrix_list = data['bond_attribute_matrix_list']
    node_attribute_matrix_list = data['node_attribute_matrix_list']

    kwargs = {}
    if mode in ['qm8', 'qm9']:
        for task in mode2task_list[mode]:
            kwargs[task] = data[task]
    else:
        kwargs['label_name'] = data['label_name']

    return max_atom_num, \
           max_degree, \
           adjacent_matrix_list, \
           distance_matrix_list, \
           bond_attribute_matrix_list, \
           node_attribute_matrix_list, \
           kwargs


class GraphDataset(Dataset):
    def __init__(self, node_attribute_matrix_list, adjacent_matrix_list, distance_matrix_list):
        self.node_attribute_matrix_list = node_attribute_matrix_list
        self.adjacent_matrix_list = adjacent_matrix_list
        self.distance_matrix_list = distance_matrix_list

    def __len__(self):
        return len(self.node_attribute_matrix_list)

    def __getitem__(self, idx):
        node_attribute_matrix = torch.from_numpy(self.node_attribute_matrix_list[idx])
        adjacent_matrix = torch.from_numpy(self.adjacent_matrix_list[idx])
        distance_matrix = torch.from_numpy(self.distance_matrix_list[idx])
        return node_attribute_matrix, adjacent_matrix, distance_matrix


def get_walk_representation(dataloader, device):
    X_embed = []
    embedded_graph_matrix_list = []
    for batch_id, (node_attribute_matrix, adjacent_matrix, distance_matrix) in enumerate(tqdm(dataloader)):
        node_attribute_matrix = Variable(node_attribute_matrix).float().to(device)
        adjacent_matrix = Variable(adjacent_matrix).float().to(device)
        distance_matrix = Variable(distance_matrix).float().to(device)

        tilde_node_attribute_matrix = model.embeddings(node_attribute_matrix)

        walk = tilde_node_attribute_matrix
        v1 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v2 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v3 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v4 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v5 = torch.sum(walk, dim=1)

        walk = torch.bmm(adjacent_matrix, walk) * tilde_node_attribute_matrix
        v6 = torch.sum(walk, dim=1)

        embedded_graph_matrix = torch.stack([v1, v2, v3, v4, v5, v6], dim=1)

        if torch.cuda.is_available():
            tilde_node_attribute_matrix = tilde_node_attribute_matrix.cpu()
            embedded_graph_matrix = embedded_graph_matrix.cpu()
        X_embed.extend(tilde_node_attribute_matrix.data.numpy())
        embedded_graph_matrix_list.extend(embedded_graph_matrix.data.numpy())

    embedded_node_matrix_list = np.array(X_embed)
    embedded_graph_matrix_list = np.array(embedded_graph_matrix_list)
    print('embedded_node_matrix_list: ', embedded_node_matrix_list.shape)
    print('embedded_graph_matrix_list shape: {}'.format(embedded_graph_matrix_list.shape))

    return embedded_node_matrix_list, embedded_graph_matrix_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace'], required=True)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--run_seed', type=int, default=0)
    parser.add_argument('--device', type=str, required=True)
    args = parser.parse_args()
    mode = args.mode
    data_seed = args.data_seed
    run_seed = args.run_seed
    device = args.device

    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(run_seed)

    embedding_dimension_list = [50, 100]
    if mode in ['hiv'] or 'pcba' in mode or 'clintox' in mode:
        feature_num = 42
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                             range(36, 38), range(38, 40), range(40, 42)]
    elif mode in ['qm8', 'qm9']:
        feature_num = 32
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 32)]
    else:
        feature_num = 42
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                             range(36, 38), range(38, 40), range(40, 42)]

    segmentation_list = np.array(segmentation_list)
    segmentation_num = len(segmentation_list)

    max_atom_num, \
    max_degree, \
    adjacent_matrix_list, \
    distance_matrix_list, \
    bond_attribute_matrix_list, \
    node_attribute_matrix_list, \
    kwargs = get_data(f'./reproduce/dataset/{mode}_graph.npz')

    dataset = GraphDataset(node_attribute_matrix_list, adjacent_matrix_list, distance_matrix_list)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    info = dataset_info(mode)
    for embedding_dimension in embedding_dimension_list:
        model = CBoW(feature_num=feature_num, embedding_dim=embedding_dimension,
                     task_num=segmentation_num, task_size_list=segmentation_list)

        weight_file = f'./reproduce/saved_models/embedding/{mode}/data_seed={data_seed}/run_seed={run_seed}/{embedding_dimension}_CBoW_non_segment.pt' \
                      if info.splitting == 'random' else \
                      f'./reproduce/saved_models/embedding/{mode}/run_seed={run_seed}/{embedding_dimension}_CBoW_non_segment.pt'
        print('weight file is {}'.format(weight_file))
        model.load_state_dict(torch.load(weight_file))
        model.to(device)
        model.eval()

        embedded_node_matrix_list, embedded_graph_matrix_list = get_walk_representation(dataloader, device)
        print('embedded_graph_matrix_list\t', embedded_graph_matrix_list.shape)

        out_file_path = f'./reproduce/saved_models/embedding/{mode}/data_seed={data_seed}/run_seed={run_seed}/grammed_cbow_{embedding_dimension}_graph' \
                        if info.splitting == 'random' else \
                        f'./reproduce/saved_models/embedding/{mode}/run_seed={run_seed}/grammed_cbow_{embedding_dimension}_graph'
        kwargs['max_atom_num'] = max_atom_num
        kwargs['max_degree'] = max_degree
        kwargs['adjacent_matrix_list'] = adjacent_matrix_list
        kwargs['distance_matrix_list'] = distance_matrix_list
        kwargs['embedded_node_matrix_list'] = embedded_node_matrix_list
        kwargs['embedded_graph_matrix_list'] = embedded_graph_matrix_list
        np.savez_compressed(out_file_path, **kwargs)
        print(kwargs.keys())
        print()
