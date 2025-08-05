import argparse
import torch
from data.dataset import SingleCellDataset
from models.cae import CAE
from models.classifier import SwarmLDA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_query', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    # load query data
    import numpy as np
    Xq = np.random.rand(1000,40000)  # placeholder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cae = CAE(40000,128).to(device)
    cae.load_state_dict(torch.load(args.model_path))
    ds = SingleCellDataset(Xq, None)
    loader = torch.utils.data.DataLoader(ds, batch_size=256)
    preds = []
    for x in loader:
        x = x.to(device)
        recon, z = cae(x)
        preds.append(torch.argmax(z,1).cpu().numpy())
    import pandas as pd
    df = pd.DataFrame({'pred': np.concatenate(preds)})
    df.to_csv(args.output, index=False)

if __name__=='__main__':
    main()
