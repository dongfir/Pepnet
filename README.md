# PepNet

Residue-level peptide–protein binding prediction via a cross-modal graph network (sequence encoder + structural GNN + cross-modal attention + residual fusion + cosine-margin head).

## 1) Requirements

Python >= 3.10
torch==2.7.1+cu118
torch-geometric==2.6.1
numpy==2.3.1
scikit-learn==1.7.1
tqdm==4.67.1
pandas

## 2) Install

```bash
pip install -r requirements.txt
```

## 3) Downloads (Releases)

Weights

https://github.com/dongfir/Pepnet/releases/download/v0.1.0/pepnet_model.pt
https://github.com/dongfir/Pepnet/releases/download/v0.1.0/pepnet_model4_seed42.pt

Datasets

https://github.com/dongfir/Pepnet/releases/download/v0.1.0/biolip1.pt
https://github.com/dongfir/Pepnet/releases/download/v0.1.0/gnn_crossmodal_with_mask1.pt

## 4) Place files
datasets

data/biolip1.pt

data/gnn_crossmodal_with_mask1.pt

optional pretrained weights

results/models/pepnet_model.pt

results/models/pepnet_model4_seed42.pt


## 5)Run

Unified entry
```bash
python main.py
```

Menu

1 Train (default split 9:1)

2 Evaluate (isotonic/temperature calibration + MCC threshold sweep)

3 Predict (export residue-level CSV for the validation subset)


Direct scripts
```bash
python train/train.py
python train/eval.py
python predict.py
```

## 6)Dataset format (.pt)

Dataset format (.pt)

Expected: a sequence (e.g., list) of torch_geometric.data.Data saved to one .pt file.

Required fields

x: float32 [N, 256] (256 must match config.in_dim)

edge_index: int64 [2, E]

y: float32 or int64 [N] with values 0 or 1

Optional fields

pdb_id: str or list[str]

protein_chain_id: str or list[str]

residue_ids: Tensor or List [N] (if missing, fallback 0..N-1)

Sanity check
```bash
import torch
ds = torch.load("./data/biolip1.pt", weights_only=False)
d0 = ds[0]
print("len:", len(ds))
print("x:", d0.x.shape, d0.x.dtype)
print("edge_index:", d0.edge_index.shape, d0.edge_index.dtype)
print("y:", d0.y.shape, d0.y.dtype)
```

## 7)Prediction CSV

Prediction CSV

Default output

results/predictions/predicted_pepnet_biolip1.csv

Columns

pdb_id

chain_id

residue_index

true_label

predicted_prob (sigmoid(logit) in [0,1])

```bash
pdb_id,chain_id,residue_index,true_label,predicted_prob
1abc,A,42,1,0.9132
1abc,A,43,0,0.1875
1abc,A,44,1,0.7349
```

## 8)Reproducibility notes

Seeds for torch, random, numpy are fixed in config.py.

Evaluation matches training: isotonic or temperature calibration plus MCC threshold sweep (default scan [0.28, 0.55], step 0.002).

EMA with decay 0.999; evaluation prefers EMA weights when available.

predict.py prints absolute data_path, model_path, save_dir.






