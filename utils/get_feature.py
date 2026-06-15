import torch
from torchvision.models import resnet18

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import GNN, GNN_graphpred
import pandas as pd
import pickle
import lmdb

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=6,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'v2', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'Mole-BERT', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    # parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--split', type = str, default="random", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "v2":
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    pre_path = "/remote-home/lzy_new/catpred/kcat_lzy/Mole-BERT/data"
    dataset = MoleculeDataset(pre_path+"/dataset/" + args.dataset, dataset=args.dataset)
    # print(dataset)

    gene_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    
    if not args.input_model_file == "None":
        print('Not from scratch')
        model.from_pretrained('/remote-home/lzy_new/catpred/kcat_lzy/Mole-BERT/data/model_gin/Mole-BERT.pth')
    
    model.to(device)
    model.eval()

    env = lmdb.open('/remote-home/lzy_new/catpred/kcat_lzy/data/stand_all_lmdb_v2_Molebert', map_size=1073741824)

    for step, batch in enumerate(tqdm(gene_loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            node_representation = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
        with env.begin(write=True) as txn:
            for i in range(len(batch.ptr) - 1):
                start_idx = batch.ptr[i]
                end_idx = batch.ptr[i + 1]
                node_features = node_representation[start_idx:end_idx]
                id = batch.y[i].unsqueeze(0)
                mean_feature = node_features.mean(dim=0)  # 沿着节点维度求均值，结果是一个 300 维的向量

                result = torch.cat((mean_feature,id),dim=0)
                serialized_data = pickle.dumps(result)
                txn.put(str(id).encode('ascii'), serialized_data)


if __name__ == "__main__":
    main()
