import torch
from torch.utils.data import Dataset

class SingleCellDataset(Dataset):
    def __init__(self, X, y=None, jl_proj=None):
        '''
        X: numpy array or torch tensor (n x p)
        y: labels or None
        jl_proj: nn.Module for JL projection
        '''
        self.X = torch.from_numpy(X).float() if not torch.is_tensor(X) else X
        self.y = torch.from_numpy(y).long() if y is not None else None
        self.jl = jl_proj

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.jl:
            x = self.jl(x)
        if self.y is None:
            return x
        return x, self.y[idx]
