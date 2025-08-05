import torch
import numpy as np

def rare_augmentation(z, labels, n_min):
    '''
    z: tensor (n x d), labels: tensor (n), returns augmented indices and data
    '''
    unique = labels.unique().tolist()
    out_z, out_y = [], []
    for k in unique:
        idx = (labels==k).nonzero().view(-1)
        count = len(idx)
        if count >= n_min:
            sel = idx[torch.randperm(count)[:int(0.8*count)]]
            out_z.append(z[sel])
            out_y += [k]*len(sel)
        else:
            # all + synth
            out_z.append(z[idx])
            out_y += [k]*count
            # synth
            if count>0:
                weights = np.random.dirichlet(np.ones(count), int(0.8*n_min)-count)
                for w in weights:
                    mix = (w[:,None]*z[idx].numpy()).sum(0)
                    out_z.append(torch.from_numpy(mix).float())
                    out_y.append(k)
    return torch.cat(out_z,0), torch.tensor(out_y, dtype=torch.long)
