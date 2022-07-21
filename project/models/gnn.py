"""encoding: utf-8"""
import torch as th
import torch.nn as nn
import torch_geometric.nn as tgn


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
