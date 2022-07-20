import os
import argparse
import torch as th
import torchmetrics.functional as thm

from project.utils.datasets import node_classification_datasets as load_dataset
from project.utils.models import load_model
from project.utils.plotting import plot_metrics


root_path = os.path.dirname('.')


def main(args):
    data = load_dataset(args.data_path, args.dataset)
    data.to(args.device)

    feats = data.x
    labels = data.y
    edge_index = data.edge_index

    train_idx = th.nonzero(data.train_mask, as_tuple=False).squeeze()
    valid_idx = th.nonzero(data.val_mask, as_tuple=False).squeeze() 
    test_idx = th.nonzero(data.test_mask, as_tuple=False).squeeze()
    
    args.nfeats = feats.shape[1]
    args.nout = (labels.max() + 1).item()

    model = load_model(args.model, args).to(args.device)
    
    softmax = th.nn.LogSoftmax(dim=1)
    criterion = th.nn.NLLLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_losses = []
    valid_losses = []
    
    train_accs = []
    valid_accs = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(edge_index, feats)
        loss = criterion(softmax(out[train_idx]), labels[train_idx])

        loss.backward()
        optimizer.step()

        train_acc = thm.accuracy(out[train_idx], labels[train_idx]).item()
        train_accs.append(train_acc)
        train_losses.append(loss.item())

        model.eval()
        with th.no_grad():
            out = model(edge_index, feats)
            loss = criterion(softmax(out[valid_idx]), labels[valid_idx])
            valid_acc = thm.accuracy(out[valid_idx], labels[valid_idx]).item()

            valid_accs.append(valid_acc)
            valid_losses.append(loss.item())
    
    model.eval()
    with th.no_grad():
        out = model(edge_index, feats)
        loss = criterion(softmax(out[test_idx]), labels[test_idx])
        print('Test Loss: {:.5f}'.format(loss.item()))

        print('Test Accuracy: {:.5f}'.format(thm.accuracy(out[test_idx], labels[test_idx])))
        print('Test F1: {:.5f}'.format(thm.f1_score(out[test_idx], labels[test_idx])))
        print('Test Recall: {:.5f}'.format(thm.recall(out[test_idx], labels[test_idx])))

    metrics = {
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'train_acc': train_accs,
        'valid_acc': valid_accs
    }

    plot_metrics(args.dataset, args.model, metrics, args.epochs, os.path.join(root_path, 'figures'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--data-path', type=str, default='data', help='path to data directory')    
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    
    parser.add_argument('--model', type=str, required=True, help='model name', choices=['GCN', 'GAT', 'SGC', 'DGCN'])
    parser.add_argument('--nhids', type=int, default=16, help='number of hidden units')
    parser.add_argument('--nheads', type=int, default=8, help='number of heads (GAT)')
    parser.add_argument('--K', type=int, default=1, help='number of hops (SGC)')
    parser.add_argument('--proj-dim', type=int, default=8, help='dimension of projection (DGCN)')
    parser.add_argument('--skip-connection', choices=['add', 'concat'], default='add', help='skip connection type (DGCN)')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    

    args = parser.parse_args()
    main(args)