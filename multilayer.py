import os
import argparse
import pickle
import logging
import torch as th
import torchmetrics.functional as thm
from tqdm import tqdm
from project.utils.datasets import node_classification_datasets as load_dataset
from project.models.gnn import MultiLayerDGCN


root_path = os.path.dirname('.')


def main(args):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    
    
    data = load_dataset(args.data_path, args.dataset)
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

    train_idx = th.nonzero(data.train_mask, as_tuple=False).squeeze()
    valid_idx = th.nonzero(data.val_mask, as_tuple=False).squeeze() 
    test_idx = th.nonzero(data.test_mask, as_tuple=False).squeeze()
    
    args.nfeats = feats.shape[1]
    args.nout = (labels.max() + 1).item()

    model = MultiLayerDGCN(args.nfeats, nhids, args.nout, proj_dim, args.n_layers).to(args.device)
    
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
        test_acc = thm.accuracy(softmax(out[test_idx]), labels[test_idx]).item()
    
    print((args.n_layers, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--data-path', type=str, default='data', help='path to data directory')    
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    
    parser.add_argument('--nhids', type=int, default=16, help='number of hidden units')
    parser.add_argument('--proj-dim', type=int, default=8, help='dimension of projection (DGCN)')
    parser.add_argument('--n-layers', type=int, default=2, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparameters')

    args = parser.parse_args()
    main(args)