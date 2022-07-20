"""encoding: utf-8"""
import torch as th
import torch.nn as nn
import torch_geometric.nn as tgn


class GCN(nn.Module):
    """
        Graph Convolutional Networks (GCN)
    """
    def __init__(self, nfeats, nhids, nout) -> None:
        super().__init__()

        self.conv1 = tgn.GCNConv(nfeats, nhids)
        self.conv2 = tgn.GCNConv(nhids, nout)

    def forward(self, edge_index, x):
        x = th.relu(self.conv1(x, edge_index))
        x = th.dropout(x, p=0.5, train=self.training)
        x = self.conv2(x, edge_index)

        return x


class GAT(nn.Module):
    """
        Graph Attention Networks (GAT)
    """
    def __init__(self, nfeats, nhids, nout, nheads) -> None:
        super().__init__()

        self.conv1 = tgn.GATConv(nfeats, nhids, nheads)
        self.conv2 = tgn.GATConv(nhids * nheads, nout, 1)
        self.activation = th.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, edge_index, x):
        x = self.activation(self.conv1(x, edge_index))
        x = th.dropout(x, p=0.6, train=self.training)
        x = self.conv2(x, edge_index)

        return x


class SGC(nn.Module):
    """
        Simplifying Graph Convolutional (SGC)
    """
    def __init__(self, nfeats, nouts, K=1) -> None:
        super().__init__()

        self.conv = tgn.SGConv(nfeats, nouts, K)

    
    def forward(self, edge_index, x):
        x = self.conv(x, edge_index)

        return x


class DGCN(nn.Module):
    """
        Dense Graph Convolutional Networks (DGCN)
    """
    def __init__(self, nfeats, nhids, nout, proj_dim=4, skip_connection = 'add') -> None:
        super().__init__()

        if skip_connection not in ['add', 'concat']:
            raise ValueError('skip_connection must be either "add" or "concat"')

        self.skip_connection = skip_connection
        self.nhids = nhids if self.skip_connection == 'add' else nhids + proj_dim

        self.projection = nn.Linear(nfeats, proj_dim)

        self.conv1 = tgn.GCNConv(nfeats, nhids)
        self.conv2 = tgn.GCNConv(nhids, nout)


    def forward(self, edge_index, x):
        x_porj = self.projection(x) if self.skip_connection == 'concat' else x

        x = th.relu(self.conv1(x, edge_index))
        x = th.dropout(x, p=0.5, train=self.training)
        
        x = th.cat([x, x_porj]) if self.skip_connection == 'concat' else x + x_porj
        
        x = self.conv2(x, edge_index)

        return x
