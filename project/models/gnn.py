"""encoding: utf-8"""
import math
import dgl
import torch as th
import torch.nn as nn
import torch_geometric.nn as tgn


class DGCNEncoder(nn.Module):
    """
        Dense Graph Convolutional Networks (DGCN)
    """
    def __init__(self, nfeats, nhids, proj_dim=4, skip_connection = 'add') -> None:
        super().__init__()

        if skip_connection not in ['add', 'concat']:
            raise ValueError('skip_connection must be either "add" or "concat"')

        if skip_connection == 'add':
            raise ValueError('remove because of hyperparameter tuning')

        self.skip_connection = skip_connection
        self.nhids = nhids + proj_dim

        self.projection = nn.Linear(nfeats, proj_dim, bias=False)

        self.conv1 = tgn.GCNConv(nfeats, nhids)
        self.conv2 = tgn.GCNConv(self.nhids, self.nhids)


    def forward(self, edge_index, x):
        x_proj = self.projection(x)

        x = th.relu(self.conv1(x, edge_index))
        x = th.dropout(x, p=0.5, train=self.training)

        x = th.cat([x, x_proj], dim=1) if self.skip_connection == 'concat' else x + x_proj
                
        x = self.conv2(x, edge_index)

        return th.cat([x, x_proj], dim=1)
        

class DGCN(nn.Module):
    """
        Dense Graph Convolutional Networks (DGCN)
    """
    def __init__(self, nfeats, nhids, nout, proj_dim=4, skip_connection = 'add') -> None:
        super().__init__()

        if skip_connection not in ['add', 'concat']:
            raise ValueError('skip_connection must be either "add" or "concat"')

        if skip_connection == 'add':
            raise ValueError('remove because of hyperparameter tuning')

        self.skip_connection = skip_connection
        self.nhids = nhids + proj_dim

        self.projection = nn.Linear(nfeats, proj_dim)

        self.conv1 = tgn.GCNConv(nfeats, nhids)
        self.conv2 = tgn.GCNConv(self.nhids, nout)


    def forward(self, edge_index, x):
        x_proj = self.projection(x)

        x = th.relu(self.conv1(x, edge_index))
        x = th.dropout(x, p=0.5, train=self.training)

        x = th.cat([x, x_proj], dim=1) if self.skip_connection == 'concat' else x + x_proj
                
        x = self.conv2(x, edge_index)

        return x



class MultiLayerDGCN(nn.Module):
    def __init__(self, nfeats, nhids, nout, proj_dim=4, n_layers=2) -> None:
        super().__init__()

        self.nhids = nhids + proj_dim

        self.projection = nn.Linear(nfeats, proj_dim)
        
        layers = []
        layers.append(tgn.GCNConv(nfeats, nhids))
        for _ in range(n_layers - 2):
            layers.append(tgn.GCNConv(self.nhids, nhids))
        
        layers.append(tgn.GCNConv(self.nhids, nout))
        self.layers = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.projection.weight.size(1))
        self.projection.weight.data.normal_(-stdv, stdv)


    def forward(self, edge_index, x):
        x_proj = self.projection(x)

        x = th.relu(self.layers[0](x, edge_index))
        x = th.dropout(x, p=0.5, train=self.training)
        x = th.cat([x, x_proj], dim=1)

        for i in range(1, len(self.layers) - 1):
            x = th.relu(self.layers[i](x, edge_index))
            x = th.cat([x, x_proj], dim=1)
            x = th.dropout(x, p=0.5, train=self.training)
        
        x = self.layers[-1](x, edge_index)
        return x
