import torch as th
import torch_geometric.nn as tgn


class Node2Vec:
    def __init__(self, edge_index, embedding_dim, walk_length, context_size, walks_per_node, p, q, num_negative_samples) -> None:
        self.model = tgn.models.Node2Vec(edge_index, embedding_dim, walk_length, context_size, walks_per_node, p, q, num_negative_samples)


    def train(self, epochs, batch_size=128, lr=0.01):
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        loader = self.model.loader(batch_size=batch_size, shuffle=True)
        
        optimizer = th.optim.Adam(self.model.parameters(), lr=lr)

        self.model.to(device)
        self.model.train()

        for epoch in range(epochs):
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()

                pos_rw, neg_rw = pos_rw.to(device), neg_rw.to(device)

                loss = self.model.loss(pos_rw, neg_rw)
                
                loss.backward()
                optimizer.step()
                
        return self.model


    def test(self, train_x, train_y, test_x, test_y):
        import sklearn.metrics as metrics
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(solver='lbfgs', multi_class='auto')\
                .fit(train_x.detach().cpu().numpy(),train_y.detach().cpu().numpy())
        
        test_acc = metrics.accuracy_score(test_y.detach().cpu().numpy(), clf.predict(test_x.detach().cpu().numpy()))
        test_recall = metrics.recall_score(test_y.detach().cpu().numpy(), clf.predict(test_x.detach().cpu().numpy()), average='micro')
        test_f1 = metrics.f1_score(test_y.detach().cpu().numpy(), clf.predict(test_x.detach().cpu().numpy()), average='micro')
        
        return test_acc, test_recall, test_f1


    def embeddings(self, nodes=None):
        if nodes is not None:
            return self.model.embeddings(nodes)
        return self.model.embedding.weight.data
