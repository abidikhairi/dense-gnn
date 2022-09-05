import argparse
import torch as th
import torchmetrics.functional as thm
import torch_geometric.transforms as tgt
from project.models.gnn import LinkPredictor
from project.utils.datasets import node_classification_datasets as load_dataset


def train(model, loss_fn, optimizer, data, device):
    model.train()
    data.to(device)
    optimizer.zero_grad()
    logits = model(data.edge_label_index, data.x).view(-1)
    loss = loss_fn(logits, data.edge_label)
    
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, loss_fn, data, device):
    model.eval()
    data.to(device)

    logits = model(data.edge_label_index, data.x).view(-1)
    loss = loss_fn(logits, data.edge_label)
    auroc = thm.auroc(logits, data.edge_label.long())
    
    return loss.item(), auroc


@th.no_grad()
def test(model, loss_fn, data, device):
    model.eval()
    data.to(device)

    logits = model(data.edge_label_index, data.x).view(-1)
    loss = loss_fn(logits, data.edge_label).item()
    auroc = thm.auroc(logits, data.edge_label.long()).item()
    
    return loss, auroc


def main(args):
    device = th.device(args.device)

    data = load_dataset(args.data_path, args.dataset)
    transform = tgt.RandomLinkSplit(num_test=0.1, num_val=0.4, neg_sampling_ratio=1)
    train_data, val_data, test_data = transform(data)
    
    nfeats = data.num_node_features
    nhids = args.nhids
    proj_dim = args.proj_dim

    predictor = LinkPredictor(nfeats, nhids, proj_dim, 'concat').to(device)
    
    loss_fn = th.nn.BCELoss()
    optimizer = th.optim.Adam(predictor.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in range(args.epochs):
        train_loss = train(predictor, loss_fn, optimizer, train_data, device)

        valid_loss, valid_auroc = evaluate(predictor, loss_fn, val_data, device)

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Val AUROC: {valid_auroc:.4f}')
        
    test_loss, test_auroc = test(predictor, loss_fn, test_data, device)

    print(f'Test Loss: {test_loss:.4f}, Test AUROC: {test_auroc * 100:.2f} %')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--data-path', type=str, default='data', help='path to data directory')    
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    
    parser.add_argument('--nhids', type=int, default=16, help='number of hidden units')
    parser.add_argument('--proj-dim', type=int, default=8, help='dimension of projection (DGCN)')
    parser.add_argument('--skip-connection', choices=['add', 'concat'], default='concat', help='skip connection type (DGCN)')

    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparameters')

    args = parser.parse_args()
    main(args)