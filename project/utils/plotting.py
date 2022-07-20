import os
import matplotlib.pyplot as plt


def plot_metrics(dataset, model, metrics, epochs, save_path):
    """
    Plot the metrics for a given dataset and model.
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot(range(epochs), metrics['train_acc'], label='training accuracy')
    plt.plot(range(epochs), metrics['valid_acc'], label='validation accuracy')

    plt.legend()
    plt.savefig(os.path.join(save_path, f'{dataset}-{model}-accuracy.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    
    plt.plot(range(epochs), metrics['train_loss'], label='training loss')
    plt.plot(range(epochs), metrics['valid_loss'], label='validation loss')
    
    plt.legend()
    plt.savefig(os.path.join(save_path, f'{dataset}-{model}-loss.png'))
    plt.close()