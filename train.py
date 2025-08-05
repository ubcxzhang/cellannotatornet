import argparse
import torch
from torch.utils.data import DataLoader
import config
from data.dataset import SingleCellDataset
from data.transforms import rare_augmentation
from models.jl_projection import JLProjection
from models.cae import CAE
from models.classifier import SwarmLDA, Aggregator

def train_cae(ref_loader, query_loader, device):
    p = config.jl_dim if config.use_jl else ref_loader.dataset.X.shape[1]
    jl = JLProjection(p, config.jl_dim) if config.use_jl else None
    cae = CAE(config.jl_dim if config.use_jl else p, config.latent_dim).to(device)
    opt = torch.optim.Adam(cae.parameters(), lr=config.learning_rate)
    for ep in range(config.epochs_cae):
        for (xr, yr), xq in zip(ref_loader, query_loader):
            xr, yr, xq = xr.to(device), yr.to(device), xq.to(device)
            x = torch.cat([xr, xq],0)
            recon, z = cae(x)
            labels = torch.cat([yr, -torch.ones_like(yr)],0).long()  # -1 for query
            loss, lrec, lvar = cae.loss(x, recon, z, yr, config.lambda1, config.lambda2)
            opt.zero_grad(); loss.backward(); opt.step()
    torch.save(cae.state_dict(), 'cae.pt')
    return cae

def train_classifier(cae, ref_loader, device):
    # placeholder: user to implement
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['end2end','cae','classifier'], required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load data placeholder np arrays
    import numpy as np
    Xr = np.random.rand(1000, 40000)
    yr = np.random.randint(0,10,1000)
    Xq = np.random.rand(1000, 40000)
    jr = JLProjection(40000, config.jl_dim) if config.use_jl else None
    ref_ds = SingleCellDataset(Xr, yr, jl_proj=jr)
    query_ds = SingleCellDataset(Xq, None, jl_proj=jr)
    ref_loader = DataLoader(ref_ds, batch_size=config.batch_ref, shuffle=True)
    query_loader = DataLoader(query_ds, batch_size=config.batch_ref, shuffle=True)
    if args.mode in ('end2end','cae'):
        cae = train_cae(ref_loader, query_loader, device)
    if args.mode in ('end2end','classifier'):
        train_classifier(cae, ref_loader, device)

if __name__=='__main__':
    main()
