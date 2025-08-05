import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

class SwarmLDA(nn.Module):
    def __init__(self, d, M, p_sub=None, n_min=200, beta_grid=[0.15,0.3,0.45]):
        super().__init__()
        self.d = d
        self.M = M
        self.p_sub = p_sub or int(np.sqrt(d))+70
        self.n_min = n_min
        self.beta_grid = beta_grid
        # aggregation layer
        self.W = nn.Linear(M, d, bias=False)  # placeholder dims

    def forward(self, z, labels=None):
        # stub: return random scores
        n,K = z.shape[0], labels.max().item()+1
        scores = torch.randn(n,K)
        return scores

class Aggregator(nn.Module):
    def __init__(self, M, K):
        super().__init__()
        self.W = nn.Linear(M, K)

    def forward(self, P):
        # P: (n, M, K) preliminary probabilities
        # collapse: weighted sum
        M,K = P.shape[1], P.shape[2]
        flat = P.view(P.size(0), M*K)
        out = self.W(flat)
        return out

def distill_student(z, teacher_probs, beta):
    # fit one LDA: placeholder
    return torch.zeros_like(teacher_probs)
