import torch

def get_attn_decoder_mask(tensor):
    mask = torch.ones_like(tensor, requires_grad = False)
    mask = torch.tril(mask)

    return mask

if __name__ == "__main__":
    size = 20
    tensor = torch.ones(size, size)
    mask = get_attn_decoder_mask(tensor)

    print(mask)
    print(mask.shape)