import torch
from torch import nn
import numpy as np


# Baseline RNN network that tries to estimate LLR noise through time
# inspired by Artemasov et. al.
class BaselineRNN(nn.Module):
    def __init__(self, n=128, k=64, d_expand=5, depth=4, T=5):
        super().__init__()
        self.input_size = n + (n - k)
        self.rnn = nn.RNN(self.input_size, n * d_expand, depth, batch_first=False)
        self.proj = nn.Linear(n * d_expand * T, n)
        self.T = T

    def forward(self, x, s):
        x_ = torch.concat([torch.abs(x), s], axis=1)
        B = x.shape[0]
        xs = x_.repeat(self.T, 1, 1)
        outputs = self.rnn(xs)[0]
        outputs = outputs.permute(1, 0, 2).flatten(start_dim=1)
        z = self.proj(outputs)
        out = x - torch.sign(x) * z
        return out

# Baseline stacked GRU network that tries to estimate LLR noise through time
# used by Artemasov et. al.
class BaselineGRU(nn.Module):
    def __init__(self, n=128, k=64, d_expand=5, depth=4, T=5):
        super().__init__()
        self.input_size = n + (n - k)
        self.rnn = nn.GRU(self.input_size, n * d_expand, depth, batch_first=False)
        self.proj = nn.Linear(n * d_expand * T, n)
        self.T = T

    def forward(self, x, s):
        x_ = torch.concat([torch.abs(x), s], axis=1)
        B = x.shape[0]
        xs = x_.repeat(self.T, 1, 1)
        outputs = self.rnn(xs)[0]
        outputs = outputs.permute(1, 0, 2).flatten(start_dim=1)
        z = self.proj(outputs)
        out = x - torch.sign(x) * z
        return out

# Inspired by Error Correction Code Transformer (Choukroun and Wolf 2022), although it's a pretty
# standard transformer block. I threw a GELU in there though!
class TransformerBlock(nn.Module):
    def __init__(self, k=64, n=128, d=64, num_heads=8):
        super().__init__()
        self.n = n
        self.k = k
        self.N = n + (n - k)
        self.d = d
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(self.d, num_heads, batch_first=True)
        self.ln = nn.LayerNorm((self.N, self.d))
        self.ffn = nn.Sequential(
            nn.LayerNorm((self.N, self.d)),
            nn.Linear(self.d, 4*self.d),
            nn.GELU(),
            nn.Linear(4*self.d, self.d)
        )
        self.W_QKV = nn.Parameter(data=torch.ones(self.d, self.d * 3), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.W_QKV)

    def forward(self, x, mask=None):
        out = self.ln(x)
        qkv = x @ self.W_QKV
        Q, K, V = qkv[:, :, 0:self.d], qkv[:, :, self.d:2*self.d], qkv[:, :, self.d*2:]
        out = self.mha(Q, K, V, attn_mask=mask)[0]
        out = out + x
        out_2 = self.ffn(out)
        out_2 = out + out_2
        return out_2
        
class BaselineTransformer(nn.Module):
    def __init__(self, k=64, n=128, d=64, n_blocks=2, ecct_mask=None):
        super().__init__()
        self.N = n + (n - k)
        self.k = k
        self.n = n
        self.d = d
        self.n_blocks = n_blocks
        # just learn the embedding, i think positional reliability encoding as it's
        # described in the ECCT paper isn't necessary due to the nature of LLRs and soft
        # syndrome
        self.blocks = nn.ModuleList([TransformerBlock(k=k, n=n, d=d) for _ in range(n_blocks)])
        self.embedding = nn.Parameter(data=torch.ones(self.N, self.d), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.embedding)
        if ecct_mask is None:
            self.mask = None
        else:
            self.register_buffer('mask', self.create_mask(ecct_mask))
        self.fc_1 = nn.Linear(self.d, 1)
        self.fc_2 = nn.Linear(self.N, self.n)
        self.ln = nn.LayerNorm((self.N, self.d))

    # Algorithm from Error Correction Code Transformer (Choukroun and Wolf 2022)
    def create_mask(self, H):
        l, n = H.shape
        k = n-l
        mask = torch.eye(2 * self.n - self.k)
        for i in range(0, n-k):
            idx = np.argwhere(H[i] == 1)
            for j in idx:
                mask[n+i, j] = 1 
                mask[j, n+i] = 1
                for k in idx:
                    mask[j,k] = 1 
                    mask[k, j] = 1
        return mask == 0 # torch expects inverse mask

    def forward(self, x, s):
        x_ = torch.concat([torch.abs(x), s], axis=1)
        B = x.shape[0]
        out = x_.view(B, self.N, 1) * self.embedding
        for l in self.blocks:
            out = l(out, mask=self.mask)
        out = self.ln(out)
        out = self.fc_1(out)
        out = out.squeeze(-1)
        out = self.fc_2(out)
        out = x - torch.sign(x) * out
        return out