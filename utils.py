import torch
from torch.utils.data import random_split

def get_attn_decoder_mask(tensor):
    mask = torch.ones_like(tensor, requires_grad = False)
    mask = torch.tril(mask)

    return mask

def get_dataset(dataset, split_ratio):

    len_train_dataset = int(len(dataset)/ split_ratio)
    len_test_dataset = len(dataset) - len_train_dataset
    
    train_dataset, test_dataset =  random_split(dataset, [len_train_dataset, len_test_dataset])

    return train_dataset, test_dataset

if __name__ == "__main__":
    size = 20
    tensor = torch.ones(size, size)
    mask = get_attn_decoder_mask(tensor)

    print(mask)
    print(mask.shape)