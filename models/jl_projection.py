import torch
import math

class JLProjection(torch.nn.Module):
    def __init__(self, p, k):
        super().__init__()
        # fixed random matrix
        R = torch.randn(p, k) / math.sqrt(k)
        self.register_buffer('R', R)

    def forward(self, x):
        return x @ self.R
