import torch
import torch.nn as nn

# 一个epoch的训练单元
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_mae = 0.0
    for X, y in dataloader:
        X = X.to(device) # (B, 4, 7, 1025)
        y = y.to(device) # (B, 1)
        

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        running_mae += torch.abs(pred - y).sum().item() 

    epoch_mae = running_mae / len(dataloader.dataset)

    return epoch_mae

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_mae = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device) # (B, 4, 7, 1025)
            y = y.to(device) # (B, 1)

            pred = model(X)
            

            running_mae += torch.abs(pred - y).sum().item()

    avg_mae = running_mae / len(dataloader.dataset)
    
    return avg_mae