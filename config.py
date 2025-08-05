# Hyperparameters for CellAnnotatorNet

# CAE
lambda1 = 1.0
lambda2 = 0.1
learning_rate = 1e-3
batch_ref = 200
batch_query_ratio = 1.0  # q = batch_ref * ratio * (nQ/nR)
epochs_cae = 10000
latent_dim = 128

# JL projection
use_jl = True
jl_dim = 2000

# Batch adversary
adv_weight = 0.5
adv_ramp_ratio = 0.3

# Classifier
M = 10
feature_subsamp = None  # if None, uses sqrt(d)+70
n_min = 200
beta_grid = [0.15,0.30,0.45]

# Distillation
distill = True
