import torch
import torch.nn as nn

from net import MultiHeadAttention
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, embed_dim, requires_grad = False)
        
        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        _2i = torch.arange(0, embed_dim, step=2).float()

        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / embed_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_dim)))
        
    def forward(self, x):
        _, seq_len, _ = x.size() 
        
        return self.encoding[:seq_len, :]


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_model, dim_ffn):
        super(EncoderLayer, self).__init__()
        self.MHA = MultiHeadAttention(embed_dim, num_heads, d_model)
        self.FFN = nn.Sequential(nn.Linear(embed_dim, dim_ffn), nn.ReLU(), nn.Linear(dim_ffn, embed_dim))
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.MHA(x, x, x)
        x = self.layernorm(x)

        x = x + self.FFN(x)
        x = self.layernorm(x)

        return x

if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8
    d_model = 512
    dim_ffn = 2048

    batch_size = 16
    input_len = 20

    tensor = torch.rand(batch_size, input_len, embed_dim)

    pe_layer = PositionalEncoding(embed_dim, input_len)
    tensor = tensor + pe_layer(tensor)
    
    """
    plt.pcolormesh(pe_layer.encoding.numpy(), cmap='RdBu')
    plt.xlabel('Depth')
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
    """

    print(pe_layer(tensor))
    print(tensor.shape)

    encoder_layer = EncoderLayer(embed_dim, num_heads, d_model, dim_ffn)
    output = encoder_layer(tensor)

    print(output.shape)