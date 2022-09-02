import os
import argparse
import pickle
import logging
import torch as th
import torchmetrics.functional as thm
from tqdm import tqdm
from project.utils.datasets import node_classification_datasets as load_dataset
from project.utils.models import DGCN
from project.utils.plotting import plot_metrics
from project.utils.normalize import row_normalize
import wandb
import time


root_path = os.path.dirname('.')


def main(args):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    wandb.init(project='gnn-benchmark', name="dgcn-{}".format(args.dataset), config={
        'dataset': args.dataset,
    })
    
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


    # feats = row_normalize(feats)

    train_idx = th.nonzero(data.train_mask, as_tuple=False).squeeze()
    valid_idx = th.nonzero(data.val_mask, as_tuple=False).squeeze() 
    test_idx = th.nonzero(data.test_mask, as_tuple=False).squeeze()
    
    args.nfeats = feats.shape[1]
    args.nout = (labels.max() + 1).item()

    model = DGCN(args.nfeats, nhids, args.nout, proj_dim, args.skip_connection).to(args.device)
    
    softmax = th.nn.LogSoftmax(dim=1)
    criterion = th.nn.NLLLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    
    experiment_test_accuracies = []
    experiment_test_f1 = []
    experiment_test_recall = []
    
    logger.info("="*80)
    logger.info(f'Training DGCN on {args.dataset}')

    wandb.watch(model)

    for _ in tqdm(range(1), desc='Experiments', leave=False):
        tick = time.time()
        for epoch in tqdm(range(args.epochs), desc='Training', leave=False):
            model.train()
            optimizer.zero_grad()
            out = model(edge_index, feats)
            loss = criterion(softmax(out[train_idx]), labels[train_idx])
            train_acc = thm.accuracy(softmax(out[train_idx]), labels[train_idx])
            
            wandb.log({
                'train/loss': loss,
                'train/acc': train_acc,
            })

            loss.backward()
            optimizer.step()
        tock = time.time()
        wandb.log({
            'train/time': tock - tick,
        })
        model.eval()
        with th.no_grad():
            out = model(edge_index, feats)
            loss = criterion(softmax(out[test_idx]), labels[test_idx])
            wandb.log({
                'test/loss': loss,
                'test/acc': thm.accuracy(softmax(out[test_idx]), labels[test_idx]),
            })
            experiment_test_accuracies.append(thm.accuracy(softmax(out[test_idx]), labels[test_idx]).item())
            experiment_test_f1.append(thm.f1_score(softmax(out[test_idx]), labels[test_idx]).item())
            experiment_test_recall.append(thm.recall(softmax(out[test_idx]), labels[test_idx]).item())

    logger.info('Training complete')
    logger.info('Test accuracy (mean): {:.4f} %'.format(th.tensor(experiment_test_accuracies).mean().item()*100))
    logger.info('Test accuracy std: {:.4f}'.format(th.tensor(experiment_test_accuracies).std().item()*100))
    logger.info('Test F1: {:.4f}'.format(th.tensor(experiment_test_f1).mean().item()))
    logger.info('Test Recall: {:.4f}'.format(th.tensor(experiment_test_recall).mean().item()))
    logger.info("="*80)

    wandb.finish()


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
    
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparameters')

    args = parser.parse_args()
    main(args)