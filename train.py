import torch
import torch.nn as nn

from torch.utils.data import dataloader, Dataset
from torch.optim import Adam

from model import SimpleClassifier
from utils import get_dataset

def train(model, epochs, train_dataloader, val_dataloader, optimizer, criterion):
    train_history = []
    val_history = []
    
    for epoch in epochs:

        train_loss = 0.0
        # train_acc = 0.0
        val_loss = 0.0
        # val_acc = 0.0

        for batch in train_dataloader:
            src, y = batch

            optimizer.zero_grad()
            y_hat = model(src)
            loss = criterion(y, y_hat)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()/len(train_dataloader) 

        with torch.no_grad():
            model.eval()
            
            for batch in val_dataloader:
                src, y = batch
                
                y_hat = model(src)
                loss = criterion(y, y_hat)

                val_loss += loss.item()/len(val_dataloader)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(f"[Epoch {epoch}] train loss:{train_loss}, val loss:{val_loss}")

    return train_history, val_history


if __name__ == "__main__":
    model = SimpleClassifier()
    epochs = 50
    batch_size = 16

    optimizer = Adam(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train(model, epochs)