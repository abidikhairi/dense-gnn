from prettytable import PrettyTable
from project.utils.datasets import node_classification_datasets as load_dataset

if __name__ == '__main__':
    table = PrettyTable(field_names=['Dataset', '# Nodes', '# Edges', '# Node Features', '# Classes'])

    datasets = ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "Facebook", "LastFM"]

    for dataset in datasets:
        data = load_dataset(root='./data', name=dataset)


        n_feats = data.x.shape[1]
        n_nodes = data.x.shape[0]
        n_edges = data.edge_index.shape[1]
        n_labels = data.y.max().item() + 1

        table.add_row([dataset, n_nodes, n_edges, n_feats, n_labels])

    print(table)