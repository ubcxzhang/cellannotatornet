import torch

def reconstruction_loss(recon_x, x):
    return torch.nn.functional.mse_loss(recon_x, x)

def fisher_loss(z, labels, eps=1e-8):
    # simple fisher-like loss
    K = labels.max().item()+1
    mu = z.mean(0)
    Vw=0; Vb=0
    for t in range(K):
        mask = labels==t
        zt = z[mask]
        if zt.size(0)==0: continue
        mut = zt.mean(0)
        Vw += ((zt-mut)**2).sum()
        Vb += mask.float().sum() * ((mut-mu)**2).sum()
    return Vw/(Vb+eps)
