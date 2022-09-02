import argparse
import pickle
import dgl
import torch as th
import torch.nn.functional as F
import torchmetrics.functional as thm
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
import dgl.data as datasets
from project.models.gnn import DGCNGraphDGL


def main(args):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    dataset = datasets.FakeNewsDataset(name=args.dataset, feature_name='bert')
    trainset, _, testset = dgl.data.utils.split_dataset(dataset, frac_list=[0.9, 0, 0.1])

    nhid = args.nhids
    nout = args.nhids
    proj_dim = args.proj_dim
    
    n_feats = 768
    n_classes = dataset.num_classes
    batch_size = args.batch_size
    epochs = args.epochs

    trainloader = GraphDataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = GraphDataLoader(testset, batch_size=batch_size, shuffle=False)

    with open('tuning/Cora.pkl', 'rb') as f:
            params = pickle.load(f)

            lr = params['lr']
            weight_decay = params['weight_decay']

    model = DGCNGraphDGL(n_feats, nhid, nout, n_classes, proj_dim).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = th.nn.CrossEntropyLoss()
    
    model.train()

    for epoch in tqdm(range(epochs), desc='Training', leave=False):
        batch_train_loss = []
        batch_train_acc = []
        for batched_graph, labels in trainloader:
            batched_graph = batched_graph.to(device)
            labels = labels.flatten().long().to(device)
            feats = dataset.feature[batched_graph.ndata['_ID']].float().to(device)
            #feats = batched_graph.ndata['node_attr'].float().to(device)

            output = model(batched_graph, feats)
            loss = loss_fn(output, labels)
            
            loss.backward()
            optimizer.zero_grad()

            batch_train_loss.append(loss.item())
            batch_train_acc.append(thm.accuracy(F.softmax(output, dim=1).cpu(), labels.cpu()))
        # epoch loss is here
        # epoch acc is here
        epoch_train_loss = th.tensor(batch_train_loss).mean().item()
        epoch_train_acc = th.tensor(batch_train_acc).mean().item()  
         
    epoch_test_acc = []

    with th.no_grad():
        model.eval()
        for batched_graph, labels in testloader:
            batched_graph = batched_graph.to(device)
            labels = labels.flatten().long().to(device)
            feats = dataset.feature[batched_graph.ndata['_ID']].float().to(device)
            output = model(batched_graph, feats)
            
            loss = loss_fn(output, labels)
            
            epoch_test_acc.append(thm.accuracy(F.softmax(output, dim=1).cpu(), labels.cpu()))
        # epoch acc is here
        epoch_test_acc = th.tensor(epoch_test_acc).mean().item()

    print(f"Epoch {epoch} | Test Acc: {epoch_test_acc * 100:.4f} %")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, help='dataset name', choices=["politifact", "gossipcop"])
    parser.add_argument('--data-path', type=str, default='data', help='path to data directory')    
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    
    parser.add_argument('--nhids', type=int, default=30, help='number of hidden units')
    parser.add_argument('--proj-dim', type=int, default=12, help='dimension of projection (DGCN)')
    
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    
    parser.add_argument('--tuned', action='store_true', help='use tuned hyperparameters')

    args = parser.parse_args()
    main(args)