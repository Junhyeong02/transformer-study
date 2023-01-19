import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class CultureDataset(Dataset):
    def __init__(self, input_df:pd.DataFrame, output_df:pd.DataFrame = None, train_mode = True):
        self.input_df = input_df
        self.output_df = output_df
        self.train_mode = train_mode

        self.input_data = []
        self.label_data = []
        
        for key, value in self.input_df.groupby("Sample_no", as_index = False).groups.items():
            x = self.input_df.loc[value].drop(["Sample_no", "Week"], axis = 1).to_numpy()
            self.input_data.append(torch.tensor(x, dtype=torch.float32))
            
            if self.train_mode:
                y = np.squeeze(self.output_df[self.output_df["Sample_no"] == key].drop(["Sample_no", "Date", "Week"], axis = 1).to_numpy())
            else:
                y = np.zeros(3)

            self.label_data.append(torch.tensor(y, dtype=torch.float32))


    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.label_data[idx]

        return x, y

    def __len__(self):
        return len(self.input_data)

