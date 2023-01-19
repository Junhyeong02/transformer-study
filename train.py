import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import SimpleClassifier
from utils import get_dataset
from data import CultureDataset

def custom_loss(tgt, out):
    def mae(true, pred):
        return torch.mean(torch.abs(true - pred))
    
    growth_mae = mae(out[:, 0], tgt[:,0])
    stem_mae = mae(out[:, 1], tgt[:, 1])
    flowering_mae = mae(out[:, 2], tgt[:, 2])

    return (growth_mae*0.1 + stem_mae + flowering_mae)/3

def train(model, epochs, train_dataloader, val_dataloader, optimizer, criterion):
    train_history = []
    val_history = []
    
    for epoch in range(1, epochs+1):

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

    train_input_df = pd.read_csv("./data/train_processed_input.csv")
    train_output_df = pd.read_csv("./data/train_processed_output.csv")

    culturedataset = CultureDataset(train_input_df, train_output_df)

    print(culturedataset[0][0].shape)
    train_dataset, val_dataset = get_dataset(culturedataset, 0.8)

    print(len(train_dataset), len(val_dataset))

    # model
    num_encoder_layer = 6
    max_len = culturedataset[0][0].shape[0]
    embed_dim = culturedataset[0][0].shape[1]
    num_heads = 9
    d_model = culturedataset[0][0].shape[1]
    dim_ffn = 256
    output_dim = 3

    model = SimpleClassifier(num_encoder_layer, max_len, embed_dim, num_heads, d_model, dim_ffn, output_dim)
    
    # train
    epochs = 50
    batch_size = 16
    learning_rate = 1e-2

    optimizer = Adam(model.parameters(), lr = learning_rate)
    criterion = custom_loss

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)    
    
    train_history, val_history = train(model, epochs, train_dataloader, val_dataloader, optimizer, custom_loss)

    plt.plot(train_history, label="Train Loss") 
    plt.plot(val_history, 'r', label="Val Loss") 
    plt.title('ConvLSTM training & val Loss', fontsize=20) 
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    plt.show()