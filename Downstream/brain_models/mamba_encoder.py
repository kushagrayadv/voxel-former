import torch
import torch.nn as nn

class MambaMixer(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class MambaEncoder(nn.Module):
    def __init__(self, dim, depth, dropout):
        super().__init__()

        self.input_proj = nn.Linear()

    def forward(self, x):
        return x