import optuna
import pickle
import dgl
import torch as th
import torch.nn.functional as F
import torchmetrics.functional as thm
import dgl.data as datasets 
from dgl.dataloading import GraphDataLoader
from project.models.gnn import DGCNGraphDGL
from project.utils.datasets import node_classification_datasets as load_dataset


def mask_to_index(mask):
    return th.nonzero(mask, as_tuple=False).squeeze()


def build_model(params):
    nfeats = params['nfeats']
    nout = params['nout']
    nhids = params['nhids']
    proj_dim = params['proj_dim']

    model = DGCNGraphDGL(nfeats, nhids, nout, 2, proj_dim)

    return model


def build_optimizer(model, params):
    lr = params['lr']
    weight_decay = params['weight_decay']
    
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer


def train_and_evaluate(model, optimizer, trainloader, testloader, dataset):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    loss_fn = th.nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(100):
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
    
    return 0.0


def objective(trial, dataset):
    params = {
        'batch_size': trial.suggest_int('batch_size', 4, 16),
        'nhids': trial.suggest_int('nhids', 8, 32),
        'proj_dim': trial.suggest_int('proj_dim', 4, 16),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-1),
    }
    batch_size = params['batch_size']

    dataset = datasets.FakeNewsDataset(name='politifact', feature_name='bert')
    trainset, _, testset = dgl.data.utils.split_dataset(dataset, frac_list=[0.9, 0, 0.1])
    trainloader = GraphDataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = GraphDataLoader(testset, batch_size=batch_size, shuffle=False)

    params['nout'] = params['nhids']
    params['nfeats'] = 768

    model = build_model(params)
    optimizer = build_optimizer(model, params)

    accuracy = train_and_evaluate(model, optimizer, trainloader, testloader, dataset)

    return accuracy


def main():
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

    study.optimize(lambda trial: objective(trial, 'politifact'), n_trials=30)

    print('-----------------------------------------------------')
    print('Number of finished trials: ', len(study.trials))
    print(f'Best trial (politifact):')
    
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print('\t{}: {}'.format(key, value))
    
    print('-----------------------------------------------------')

    with open(f'tuning/politifact.pkl', 'wb') as f:
        pickle.dump(best_trial.params, f)


if __name__ == '__main__':
    main()
