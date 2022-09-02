import torch_geometric.datasets as tgd
import torch_geometric.transforms as tgt
from torch.utils.data import random_split


def graph_classification_datasets(root: str, name: str):
    if name not in ["enzymes", "politifact", "gossipcop", "proteins"]:
        raise ValueError("Dataset {} not supported".format(name))

    if name in ["enzymes", "proteins"]:
        dataset = tgd.TUDataset(root=root, name=name.upper())
        train_size = int(0.8 * len(dataset))
        
        trainset = dataset[:train_size]
        testset = dataset[train_size:]
        
        return (trainset, testset)


    if name in ["politifact", "gossipcop"]:
        trainset = tgd.UPFD(root=root, name=name, feature='bert', split="train")
        testset = tgd.UPFD(root=root, name=name, feature='bert', split="test")
        
        return (trainset, testset)


def node_classification_datasets(root, name):
    if name not in ["Cora", "CiteSeer", "PubMed", "Computers", "Photo", "Facebook", "LastFM"]:
        raise ValueError("Dataset {} not supported".format(name))

    if name in ["Cora", "CiteSeer", "PubMed"]:
        return tgd.Planetoid(root=root, name=name)[0]
    
    if name in ["Computers", "Photo"]:
        data = tgd.Amazon(root, name)[0]
        tgt.RandomNodeSplit()(data)

        return data

    if name == "Facebook":
        data = tgd.FacebookPagePage(root=root)[0]
        tgt.RandomNodeSplit()(data)
        
        return data

    if name == "LastFM":
        data = tgd.LastFMAsia(root=root)[0]
        tgt.RandomNodeSplit()(data)

        return data
