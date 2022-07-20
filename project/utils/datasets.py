import torch_geometric.datasets as tgd
import torch_geometric.transforms as tgt


def node_classification_datasets(root, name):
    if name not in ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "Facebook", "LastFM"]:
        raise ValueError("Dataset {} not supported".format(name))

    if name in ["Cora", "CiteSeer", "PubMed"]:
        return tgd.Planetoid(root=root, name=name)[0]
    
    if name in ["Computers", "Photo"]:
        return tgd.Amazon(root, name)

    if name == "Facebook":
        data = tgd.FacebookPagePage(root=root)[0]
        tgt.RandomNodeSplit()(data)
        
        return data

    if name == "LastFM":
        return tgd.LastFMAsia(root=root)
