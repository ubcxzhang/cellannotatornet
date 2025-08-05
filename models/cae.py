import torch
import torch.nn as nn
from losses.cae_losses import reconstruction_loss, fisher_loss

class CAE(nn.Module):
    def __init__(self, p, d, hidden=[256,128], dropout=0.1):
        super().__init__()
        layers = []
        prev = p
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers, nn.Linear(prev,d))
        # decoder symmetric
        layers = []
        prev = d
        for h in reversed(hidden):
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev,p)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

    def loss(self, x, recon_x, z, labels, lambda1, lambda2):
        lrec = reconstruction_loss(recon_x, x)
        lvar = fisher_loss(z, labels)
        return lambda1*lrec + lambda2*lvar, lrec, lvar
