import torch
import torch.nn as nn

from net import MultiHeadAttention

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

    encoder_layer = EncoderLayer(embed_dim, num_heads, d_model, dim_ffn)
    output = encoder_layer(tensor)

    print(output.shape)