import os
import torch as th
import torch_geometric.transforms as tgt
from torch_geometric.datasets import Planetoid, Amazon, FacebookPagePage, LastFMAsia
from torch_geometric.utils import to_dense_adj


def main():
    path = './data'

    names = ['cora', 'amazon-computer', 'amazon-photo', 'facebook', 'lastfm']
    
    for name in names:
        if name == 'cora':
            dataset = Planetoid(path, name='Cora', transform=tgt.NormalizeFeatures())
        elif name == 'lastfm':
            dataset = LastFMAsia(path)
        elif name == 'amazon-photo':
            dataset = Amazon(path, name='Photo')
        elif name == 'facebook':
            dataset = FacebookPagePage(path)
        elif name == 'amazon-computer':
            dataset = Amazon(path, name='Computers')

        transform = tgt.RandomLinkSplit(num_test=0.1, num_val=0.4, neg_sampling_ratio=1)
        train_data, val_data, test_data = transform(dataset[0])

        for split, data in [('train', train_data), ('valid', val_data), ('test', test_data)]:
            edge_index = data.edge_label_index
            adj_mat = to_dense_adj(edge_index).squeeze(0)
            edge_labels = data.edge_label
            feats = data.x

            os.makedirs('./data/link-predictions/{}'.format(name), exist_ok=True)

            th.save(edge_index, f'./data/link-predictions/{name}/{split}_edge_index.pt')
            th.save(adj_mat, f'./data/link-predictions/{name}/{split}_adj_mat.pt')
            th.save(edge_labels, f'./data/link-predictions/{name}/{split}_edge_labels.pt')
            th.save(feats, f'./data/link-predictions/{name}/{split}_feats.pt')


if __name__ == '__main__':
    main()
