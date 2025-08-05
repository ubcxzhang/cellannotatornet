# CellAnnotatorNet

A PyTorch-based end-to-end framework for single-cell RNA-seq annotation integrating a 
Categorical Auto-encoder (CAE) and a three-stage classification module with optional 
knowledge distillation.

## Repository structure

```
cell_annotator_net/
├── README.md                # this file
├── requirements.txt         # Python dependencies
├── setup.py                 # package installer
├── config.py                # hyper-parameters
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Dataset & JLP projection
│   └── transforms.py        # subsampling & augmentation
├── models/
│   ├── __init__.py
│   ├── jl_projection.py     # JLProjection module
│   ├── cae.py               # CAE architecture
│   └── classifier.py        # SwarmLDA & aggregator & distillation
├── losses/
│   ├── __init__.py
│   └── cae_losses.py        # reconstruction & fisher losses
├── utils.py                 # logging, checkpoints, seed
├── train.py                 # training entrypoint
└── infer.py                 # inference & distillation
```

## Installation

```bash
git clone <repo_url>
cd cell_annotator_net
pip install -r requirements.txt
```

## Usage

### Training

- **End-to-end** (CAE then classifier):
  ```bash
  python train.py --mode end2end --config config.py
  ```
- **CAE only**:
  ```bash
  python train.py --mode cae --config config.py
  ```
- **Classifier only** (requires pretrained CAE saved embeddings):
  ```bash
  python train.py --mode classifier --config config.py
  ```

### Inference

```bash
python infer.py --model_path <checkpoint> --data_query <query_file> --output predictions.csv
```

See detailed options with `python train.py --help` and `python infer.py --help`.
