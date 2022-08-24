from __future__ import print_function

import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from MoleculeBench import train_val_test_split, dataset_info


class CBoW(nn.Module):
    def __init__(self, feature_num, embedding_dim, task_num, task_size_list):
        super(CBoW, self).__init__()
        self.task_num = task_num
        self.embeddings = nn.Linear(feature_num, embedding_dim, bias=False)
        self.layers = nn.ModuleList()
        for task_size in task_size_list:
            self.layers.append(nn.Sequential(
                nn.Linear(embedding_dim, 20),
                nn.ReLU(),
                nn.Linear(20, len(task_size)),
            ))

    def forward(self, x):
        embeds = self.embeddings(x)
        embeds = embeds.sum(1)

        outputs = []
        for layer in self.layers:
            output = layer(embeds)
            outputs.append(output)
        return outputs


def get_data(max_atom_num,
             padding_size,
             feature_num,
             adjacent_matrix_list,
             node_attribute_matrix_list,
             segmentation_list):
    molecule_num = adjacent_matrix_list.shape[0]
    print('molecule num\t', molecule_num)

    X_data = []
    Y_label_list = []

    print('adjacent_matrix_list shape: {}\tnode_attribute_matrix_list shape: {}'.format(adjacent_matrix_list.shape, node_attribute_matrix_list.shape))

    for adjacent_matrix, node_attribute_matrix in zip(adjacent_matrix_list, node_attribute_matrix_list):
        assert len(adjacent_matrix) == max_atom_num
        assert len(node_attribute_matrix) == max_atom_num
        for i in range(max_atom_num):
            if sum(adjacent_matrix[i]) == 0:
                break
            x_temp = np.zeros((padding_size, feature_num))
            cnt = 0
            for j in range(max_atom_num):
                if adjacent_matrix[i][j] == 1:
                    x_temp[cnt] = node_attribute_matrix[j]
                    cnt += 1
            x_temp = np.array(x_temp)

            y_temp = []
            atom_feat = node_attribute_matrix[i]
            for s in segmentation_list:
                y_temp.append(atom_feat[s].argmax())

            X_data.append(x_temp)
            Y_label_list.append(y_temp)

    X_data = np.array(X_data)
    Y_label_list = np.array(Y_label_list)
    return X_data, Y_label_list


class GraphDataset(Dataset):

    def __init__(self,
                 max_atom_num,
                 padding_size,
                 feature_num,
                 adjacent_matrix_list,
                 node_attribute_matrix_list,
                 segmentation_list):
        self.X_data, self.Y_label_list = get_data(max_atom_num,
                                                  padding_size,
                                                  feature_num,
                                                  adjacent_matrix_list,
                                                  node_attribute_matrix_list,
                                                  segmentation_list)
        print('data size: ', self.X_data.shape, '\tlabel size: ', self.Y_label_list.shape)
        self.segmentation_list = segmentation_list

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        x_data = self.X_data[idx]
        y_label_list = self.Y_label_list[idx]

        x_data = torch.from_numpy(x_data)
        y_label_list = torch.from_numpy(y_label_list)
        return x_data, y_label_list


def train(train_dataloader, epochs, segmentation_num, random_dimension, device, weight_file):
    criterion = nn.CrossEntropyLoss()
    model.train()

    optimal_loss = 1e7
    for epoch in range(epochs):
        train_loss = []

        for batch_id, (x_data, y_actual) in enumerate(tqdm(train_dataloader)):
            x_data = Variable(x_data).float().to(device)
            optimizer.zero_grad()
            y_predict = model(x_data)

            loss = 0
            for i in range(segmentation_num):
                y_true = Variable(y_actual[..., i]).long().to(device)
                y_pred = y_predict[i]
                temp_loss = criterion(y_pred, y_true)
                loss += temp_loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)
        print('epoch: {}\tloss is: {}'.format(epoch, train_loss))
        if train_loss < optimal_loss:
            optimal_loss = train_loss
            print('Saving model at epoch {}\toptimal loss is {}.'.format(epoch, optimal_loss))
            torch.save(model.state_dict(), weight_file)
    print('For random dimension as {}.'.format(random_dimension))
    return


def test(dataloader, segmentation_num, random_dimension, device):
    model.eval()
    accuracy, total = 0, 0
    for batch_id, (x_data, y_actual) in enumerate(dataloader):
        x_data = Variable(x_data).float().to(device)
        y_actual = Variable(y_actual).long().to(device)
        y_predict = model(x_data)

        for i in range(segmentation_num):
            y_true, y_pred = y_actual[..., i].cpu().data.numpy(), y_predict[i].cpu().data.numpy()
            y_pred = y_pred.argmax(1)
            accuracy += np.sum(y_true == y_pred)
            total += y_pred.shape[0]
    accuracy = 1. * accuracy / total
    print('Accuracy: {}'.format(accuracy))

    print('For random dimension as {}.'.format(random_dimension))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'muv', 'hiv', 'bace'], required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--run_seed', type=int, default=0)
    parser.add_argument('--device', type=str, required=True)
    args = parser.parse_args()
    mode = args.mode
    epochs = args.epochs
    data_seed = args.data_seed
    run_seed = args.run_seed
    device = args.device

    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(run_seed)
        cudnn.benchmark = True

    random_dimension_list = [50, 100]

    if mode in ['qm8', 'qm9']:
        feature_num = 32
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 32)]
    else:
        feature_num = 42
        segmentation_list = [range(0, 10), range(10, 17), range(17, 24), range(24, 30), range(30, 36),
                             range(36, 38), range(38, 40), range(40, 42)]

    segmentation_list = np.array(segmentation_list)
    segmentation_num = len(segmentation_list)

    data = np.load(f'./reproduce/dataset/{mode}_graph.npz')
    max_atom_num, max_degree, adjacent_matrix_list, node_attribute_matrix_list = \
        data['max_atom_num'], data['max_degree'], data['adjacent_matrix_list'], data['node_attribute_matrix_list']

    train_indices, _, test_indices = train_val_test_split(mode, random_state=data_seed, return_indices=True)
    train_dataset = GraphDataset(max_atom_num,
                                 max_degree,
                                 feature_num,
                                 adjacent_matrix_list[train_indices],
                                 node_attribute_matrix_list[train_indices],
                                 segmentation_list)
    test_dataset = GraphDataset(max_atom_num,
                                max_degree,
                                feature_num,
                                adjacent_matrix_list[test_indices],
                                node_attribute_matrix_list[test_indices],
                                segmentation_list)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    info = dataset_info(mode)
    for random_dimension in random_dimension_list:
        weight_file = f'./reproduce/saved_models/embedding/{mode}/data_seed={data_seed}/run_seed={run_seed}/{random_dimension}_CBoW_non_segment.pt' \
                      if info.splitting == 'random' else \
                      f'./reproduce/saved_models/embedding/{mode}/run_seed={run_seed}/{random_dimension}_CBoW_non_segment.pt'
        os.makedirs(os.path.dirname(weight_file), exist_ok=True)

        model = CBoW(feature_num=feature_num, embedding_dim=random_dimension,
                     task_num=segmentation_num, task_size_list=segmentation_list)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

        train(train_dataloader, epochs, segmentation_num, random_dimension, device, weight_file)

        test(train_dataloader, segmentation_num, random_dimension, device)
        test(test_dataloader, segmentation_num, random_dimension, device)
        print()
        print()
        print()
