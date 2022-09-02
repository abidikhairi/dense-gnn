import torch as th
import torchmetrics.functional as thm


def train(model, optimizer, loss_fn, trainset, batch_size, device):
    model.train()
    
    total_loss = []
    total_acc = []
    labels = []
    preds = []

    for batch_idx, data in enumerate(trainset):
        data = data.to(device)

        output = model(data).unsqueeze(0)
        loss = loss_fn(output, data.y)

        loss.backward()

        labels.append(data.y.cpu())
        preds.append(output.flatten().cpu())

        if batch_idx % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss.append(loss.item())
            total_acc.append(thm.accuracy(th.vstack(preds), th.stack(labels)))
            
            labels = []
            preds = []
    
    return th.tensor(total_loss).mean().item(), th.tensor(total_acc).mean().item()


@th.no_grad()
def test(model, loss_fn, testset, batch_size, device):
    model.eval()

    total_loss = []
    total_acc = []
    labels = []
    preds = []
    
    for batch_idx, data in enumerate(testset):
        data = data.to(device)

        output = model(data).unsqueeze(0)
        loss = loss_fn(output, data.y)

        labels.append(data.y.cpu())
        preds.append(output.flatten().cpu())

        total_loss.append(loss.item())
    
    test_loss = th.tensor(total_loss).mean().item()
    test_acc = thm.accuracy(th.vstack(preds), th.stack(labels)).item()

    return test_loss, test_acc
