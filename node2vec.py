import argparse
import logging
import torch as th

from project.models.skipgrapm import Node2Vec
from project.utils.datasets import node_classification_datasets as load_dataset


def main(args):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    data = load_dataset(args.data_path, args.dataset)

    data.to(device)

    labels = data.y
    edge_index = data.edge_index

    train_idx = th.nonzero(data.train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(data.test_mask, as_tuple=False).squeeze()

    node2vec = Node2Vec(edge_index, args.vector_size, args.walk_length, args.window_size, args.num_walks, args.p, args.q, args.negative)

    logger.info("="*80)
    logger.info(f'Training Node2Vec on {args.dataset}')

    node2vec.train(args.epochs, args.batch_size, args.lr)

    n_feats = node2vec.embeddings()

    train_x = n_feats[train_idx]
    train_y = labels[train_idx]

    test_x = n_feats[test_idx]
    test_y = labels[test_idx]

    experiment_test_accuracies = []
    experiment_test_f1 = []
    experiment_test_recall = []

    for _ in range(10):
        test_acc, test_recall, test_f1 = node2vec.test(train_x, train_y, test_x, test_y)

        experiment_test_accuracies.append(test_acc)
        experiment_test_f1.append(test_f1)
        experiment_test_recall.append(test_recall)

    logger.info('Training complete')
    logger.info('Test accuracy (mean): {:.4f} %'.format(th.tensor(experiment_test_accuracies).mean().item()*100))
    logger.info('Test accuracy std: {:.4f}'.format(th.tensor(experiment_test_accuracies).std().item()*100))
    logger.info('Test F1: {:.4f}'.format(th.tensor(experiment_test_f1).mean().item()))
    logger.info('Test Recall: {:.4f}'.format(th.tensor(experiment_test_recall).mean().item()))
    logger.info("="*80)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cora', help='dataset name')
    parser.add_argument('--data-path', type=str, default='data', help='path to data directory')

    parser.add_argument('--num-walks', type=int, default=10, help='number of random walks')
    parser.add_argument('--walk-length', type=int, default=10, help='length of random walks')
    parser.add_argument('--window-size', type=int, default=5, help='window size')
    parser.add_argument('--negative', type=int, default=5, help='number of negative samples')
    parser.add_argument('--p', type=float, default=1, help='p parameter for negative sampling')
    parser.add_argument('--q', type=float, default=1, help='q parameter for negative sampling')
    parser.add_argument('--vector-size', type=int, default=128, help='dimension of embeddings')

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    args = parser.parse_args()
    main(args)
