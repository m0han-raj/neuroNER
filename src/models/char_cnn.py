import torch
import torch.nn as nn

class CharCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=30, out_ch=50):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(emb_dim, out_ch, 3)

    def forward(self, x):
        B, T, C = x.size()
        x = self.emb(x).view(B*T, C, -1).transpose(1,2)
        x = torch.relu(self.conv(x))
        x = torch.max(x, 2)[0]
        return x.view(B, T, -1)
