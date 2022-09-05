import os
import argparse
import pickle
import torch as th
import torchmetrics.functional as thm
from tqdm import tqdm
from project.utils.datasets import node_classification_datasets as load_dataset
from project.utils.models import DGCN
import torch_geometric.transforms as tgt


root_path = os.path.dirname('.')


def main(args):
    splitter = tgt.RandomNodeSplit(split="random", num_train_per_class=args.num_train_nodes, num_val=500, num_test=1000)

    data = load_dataset(args.data_path, args.dataset)
    
    data = splitter(data)
    data.to(args.device)

    feats = data.x
    labels = data.y
    edge_index = data.edge_index

    lr = args.lr
    weight_decay = args.weight_decay
    nhids = args.nhids
    proj_dim = args.proj_dim

    if args.tuned:
        with open('tuning/Cora.pkl', 'rb') as f:
            params = pickle.load(f)

            lr = params['lr']
            weight_decay = params['weight_decay']


    # feats = row_normalize(feats)

    train_idx = th.nonzero(data.train_mask, as_tuple=False).squeeze()
    valid_idx = th.nonzero(data.val_mask, as_tuple=False).squeeze() 
    test_idx = th.nonzero(data.test_mask, as_tuple=False).squeeze()
    
    args.nfeats = feats.shape[1]
    args.nout = (labels.max() + 1).item()

    model = DGCN(args.nfeats, nhids, args.nout, proj_dim, skip_connection='concat').to(args.device)
    
    softmax = th.nn.LogSoftmax(dim=1)
    criterion = th.nn.NLLLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    
    for epoch in tqdm(range(args.epochs), desc='Training', leave=False):
        model.train()
        optimizer.zero_grad()
        out = model(edge_index, feats)

        loss = criterion(softmax(out[train_idx]), labels[train_idx])
            

        loss.backward()
        optimizer.step()

    model.eval()
    with th.no_grad():
        out = model(edge_index, feats)
        loss = criterion(softmax(out[test_idx]), labels[test_idx])

        test_accuracy = thm.accuracy(th.exp(softmax(out[test_idx])), labels[test_idx]).item()

    print((args.num_train_nodes, test_accuracy)) 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--data-path', type=str, default='data', help='path to data directory')    
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    
    parser.add_argument('--model', type=str, required=True, help='model name', choices=['GCN', 'GAT', 'SGC', 'DGCN'])
    parser.add_argument('--nhids', type=int, default=16, help='number of hidden units')
    parser.add_argument('--proj-dim', type=int, default=8, help='dimension of projection (DGCN)')
    parser.add_argument('--num-train-nodes', type=int, default=20, help='number of training nodes per class')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparameters')

    args = parser.parse_args()
    main(args)