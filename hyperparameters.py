import optuna
import torch as th
import torchmetrics.functional as thm

from project.models.gnn import DGCN
from project.utils.datasets import node_classification_datasets as load_dataset


def mask_to_index(mask):
    return th.nonzero(mask, as_tuple=False).squeeze()


def build_model(params):
    nfeats = params['nfeats']
    nout = params['nout']
    nhids = params['nhids']
    proj_dim = params['proj_dim']
    skip_connection = params['skip_connection']

    if skip_connection == 'add':
        proj_dim = nhids

    model = DGCN(nfeats, nhids, nout, proj_dim, skip_connection=skip_connection)

    return model


def build_optimizer(model, params):
    lr = params['lr']
    weight_decay = params['weight_decay']
    
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer


def train_and_evaluate(model, optimizer, edge_index, n_feats, labels, train_idx, test_idx):
    loss_fn = th.nn.NLLLoss()
    model.train()

    for _ in range(100):
        optimizer.zero_grad()
        z = model(edge_index, n_feats)
        logits = th.log_softmax(z, dim=1)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with th.no_grad():
        z = model(edge_index, n_feats)
        logits = th.softmax(z, dim=1)
        
        acc = thm.accuracy(logits[test_idx], labels[test_idx]).item()
    
    return acc


def objective(trial):
    params = {
        'nhids': trial.suggest_int('nhids', 8, 32),
        'proj_dim': trial.suggest_int('proj_dim', 8, 16),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-1),
        'skip_connection': trial.suggest_categorical('skip_connection', ['add', 'concat'])
    }

    dataset = load_dataset('./data', 'LastFM')

    n_feats = dataset.x
    labels = dataset.y
    edge_index = dataset.edge_index

    train_idx = mask_to_index(dataset.train_mask)
    test_idx = mask_to_index(dataset.test_mask)
    
    params['nout'] = (labels.max() + 1).item()
    params['nfeats'] = n_feats.shape[1]

    model = build_model(params)
    optimizer = build_optimizer(model, params)

    accuracy = train_and_evaluate(model, optimizer, edge_index, n_feats, labels, train_idx, test_idx)

    return accuracy


def main():
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())

    study.optimize(objective, n_trials=100)

    print('-----------------------------------------------------')
    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    best_trial = study.best_trial
    for key, value in best_trial.params.items():
        print('\t{}: {}'.format(key, value))
    
    print('-----------------------------------------------------')


if __name__ == '__main__':
    main()
