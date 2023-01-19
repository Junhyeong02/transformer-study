import torch
import torch.nn as nn

from transformer import TransformerEncoder

class SimpleClassifier(nn.Module):
    def __init__(self, num_encoder_layer, max_len, embed_dim, num_heads, d_model, dim_ffn, output_dim):
        super(SimpleClassifier, self).__init__()
        self.encoder = TransformerEncoder(num_encoder_layer, max_len, embed_dim, num_heads, d_model, dim_ffn)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

if __name__ == "__main__":
    batch_size = 16
    num_encoder_layer = 6
    max_len = 20
    embed_dim = 128
    num_heads = 8
    d_model = 128
    dim_ffn = 2048
    output_dim = 10

    model = SimpleClassifier(num_encoder_layer, max_len, embed_dim, num_heads, d_model, dim_ffn, output_dim)

    tensor = torch.Tensor(batch_size, max_len, embed_dim)
    output = model(tensor)

    print(model)
    print(output.shape)

