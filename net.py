import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0

        self.wq = nn.Linear(self.embed_dim, self.d_model)
        self.wk = nn.Linear(self.embed_dim, self.d_model)
        self.wv = nn.Linear(self.embed_dim, self.d_model)

        self.wo = nn.Linear(d_model, embed_dim)
        
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, mask = None):
        # q, k, v : (batch, len, embed_dim)

        batch_size = q.shape[0]
        input_len = q.shape[1]
        
        assert q.shape[2] == self.embed_dim

        query = self.wq(q)
        key = self.wq(k)
        value = self.wv(v)

        query_split = query.reshape((batch_size, input_len, self.num_heads, self.d_model//self.num_heads)).permute((0, 2, 1, 3))
        key_split = key.reshape((batch_size, input_len, self.num_heads, self.d_model//self.num_heads)).permute((0, 2, 1, 3))
        value_split = value.reshape((batch_size, input_len, self.num_heads, self.d_model//self.num_heads)).permute((0, 2, 1, 3))

        qk_matmal = torch.matmul(query_split, torch.transpose(key_split, 2, 3))

        logit = qk_matmal/self.d_model**0.5

        if mask is not None:
            logit += (1-mask) * -1e9

        scaled_dot = self.softmax(logit)
        att_concat = torch.matmul(scaled_dot, value_split).permute((0, 2, 1, 3)).reshape(batch_size, input_len, -1)

        return self.wo(att_concat)


if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8
    d_model = 512

    batch_size = 16
    input_len = 20

    net = MultiHeadAttention(embed_dim, num_heads, d_model)
    tensor = torch.rand(batch_size, input_len, embed_dim)

    output = net(tensor, tensor, tensor)
    print(output.shape)