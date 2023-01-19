import torch
import torch.nn as nn

from module import EncoderLayer, DecoderLayer, PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, num_encoder_layer, max_len, embed_dim, num_heads, d_model, dim_ffn):
        super(TransformerEncoder, self).__init__()

        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.module_list = nn.ModuleList(EncoderLayer(embed_dim, num_heads, d_model, dim_ffn) for _ in range(num_encoder_layer))

    def forward(self, src):
        x = src + self.positional_encoding(src)
        for module in self.module_list:
            x = module(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_decoder_layer, max_len, embed_dim, num_heads, d_model, dim_ffn, tgt_mask):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        self.module_list = nn.ModuleList(DecoderLayer(embed_dim, num_heads, d_model, dim_ffn, tgt_mask) for _ in num_decoder_layer)

    def forward(self, src, tgt):
        tgt = tgt + self.positional_encoding(tgt)
        for module in self.module_list:
            pass

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        

    def forward(self, src, tgt):
        return 