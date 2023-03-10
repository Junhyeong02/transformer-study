import torch
import torch.nn as nn

from net import MultiHeadAttention
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super(PositionalEncoding, self).__init__()

        self.embed_dim = embed_dim
        self.max_len = max_len

        self.pos = torch.arange(0, self.max_len).unsqueeze(dim=1)
        self._2i = torch.arange(0, self.embed_dim, step=2).float()        
        
    def forward(self, x):
        _, seq_len, _ = x.size() 
        self.encoding = torch.zeros_like(x[0], requires_grad = False)
    
        self.encoding[:, ::2] = torch.sin(self.pos / (10000 ** (self._2i / self.embed_dim)))
        self.encoding[:, 1::2] = torch.cos(self.pos / (10000 ** (self._2i / self.embed_dim)))
        
        return self.encoding[:seq_len, :]

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()

        self.layer1 = nn.Linear(embed_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_model, dim_ffn):
        super(EncoderLayer, self).__init__()
        self.MHA = MultiHeadAttention(embed_dim, num_heads, d_model)
        self.FFN = PositionWiseFeedForward(embed_dim, dim_ffn)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src):
        x = src + self.dropout(self.MHA(src, src, src))
        x = self.layernorm1(x)

        x = x + self.dropout(self.FFN(x))
        x = self.layernorm2(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_model, dim_ffn, tgt_mask):
        super(DecoderLayer, self).__init__()
        self.maskedMHA = MultiHeadAttention(embed_dim, num_heads, d_model)
        self.MHA = MultiHeadAttention(embed_dim, num_heads, d_model)
        self.FFN = PositionWiseFeedForward(embed_dim, dim_ffn)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.mask = tgt_mask

    def forward(self, tgt, k, v):
        q = tgt + self.dropout(self.maskedMHA(tgt, tgt, tgt, self.mask))
        q = self.layernorm1(q)
        
        x = q + self.dropout(self.MHA(q, k, v))
        x = self.layernorm2(x)

        x = x + self.dropout(self.FFN(x))
        x = self.layernorm3(x)

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