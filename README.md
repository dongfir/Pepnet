PepNet

Residue-level peptide–protein prediction via a cross-modal graph network.

<p align="center"> <img alt="Python >= 3.10" src="https://img.shields.io/badge/python-%3E%3D3.10-blue" /> <img alt="torch 2.7.1+cu118" src="https://img.shields.io/badge/torch-2.7.1%2Bcu118-EE4C2C" /> <img alt="torch_geometric 2.6.1" src="https://img.shields.io/badge/torch__geometric-2.6.1-0F9D58" /> </p>

Summary. PepNet uses dual encoders (sequence + structural GNN), cross-modal attention, and a residual fusion block. The classifier head is noise-robust (cosine-margin + gate-based adaptive fusion + sample-wise weights). Training supports AMP, EMA, cosine LR with warmup, and gradient clipping. Evaluation mirrors training via isotonic/temperature calibration and an MCC-based threshold search for easy reproducibility.

1) Installation

Option A (recommended)

pip install -r requirements.txt


Option B (pinned versions)

# choose the proper CUDA wheel for torch if needed
pip install torch==2.7.1+cu118
pip install torch-geometric==2.6.1
pip install numpy==2.3.1
pip install scikit-learn==1.7.1
pip install tqdm==4.67.1
pip install pandas
# optional for plotting:
# pip install matplotlib seaborn


No HHblits / ProtTrans required; datasets are preprocessed .pt files.

2) Pretrained Weights & Datasets

Download from GitHub Releases

Weights

pepnet_model.pt

pepnet_model4_seed42.pt

Datasets

biolip1.pt

gnn_crossmodal_with_mask1.pt

Quick fetch

<details> <summary>Windows PowerShell</summary>
New-Item -ItemType Directory -Force -Path .\data, .\results\models | Out-Null
iwr https://github.com/dongfir/Pepnet/releases/download/v0.1.0/biolip1.pt                   -OutFile .\data\biolip1.pt
iwr https://github.com/dongfir/Pepnet/releases/download/v0.1.0/gnn_crossmodal_with_mask1.pt -OutFile .\data\gnn_crossmodal_with_mask1.pt
iwr https://github.com/dongfir/Pepnet/releases/download/v0.1.0/pepnet_model.pt              -OutFile .\results\models\pepnet_model.pt
iwr https://github.com/dongfir/Pepnet/releases/download/v0.1.0/pepnet_model4_seed42.pt      -OutFile .\results\models\pepnet_model4_seed42.pt

</details> <details> <summary>Linux/macOS</summary>
mkdir -p data results/models
wget -O data/biolip1.pt https://github.com/dongfir/Pepnet/releases/download/v0.1.0/biolip1.pt
wget -O data/gnn_crossmodal_with_mask1.pt https://github.com/dongfir/Pepnet/releases/download/v0.1.0/gnn_crossmodal_with_mask1.pt
wget -O results/models/pepnet_model.pt https://github.com/dongfir/Pepnet/releases/download/v0.1.0/pepnet_model.pt
wget -O results/models/pepnet_model4_seed42.pt https://github.com/dongfir/Pepnet/releases/download/v0.1.0/pepnet_model4_seed42.pt

</details>

Expected layout after download:

data/
  ├─ biolip1.pt
  └─ gnn_crossmodal_with_mask1.pt
results/
  └─ models/
      ├─ pepnet_model.pt
      └─ pepnet_model4_seed42.pt

3) Dataset format (.pt)

Load via torch.load(path, weights_only=False). It should return an indexable sequence (e.g., a list) of torch_geometric.data.Data objects. Each item contains:

Field	Shape / dtype	Required	Notes
x	FloatTensor [N, 256]	✔︎	Residue node features; 256 must match config.in_dim.
edge_index	LongTensor [2, E]	✔︎	COO edge indices (0-based).
y	Float/LongTensor [N]	✔︎	Residue-level labels (0/1).
pdb_id	str or list[str]	–	Exported to CSV if present.
protein_chain_id	str or list[str]	–	Exported to CSV if present.
residue_ids	Tensor/List [N]	–	Residue indices; fallback to 0..N-1.

Sanity check

import torch
ds = torch.load("./data/biolip1.pt", weights_only=False)
d0 = ds[0]
print("len:", len(ds))
print("x:", d0.x.shape, d0.x.dtype)
print("edge_index:", d0.edge_index.shape, d0.edge_index.dtype)
print("y:", d0.y.shape, d0.y.dtype)

4) Run
(a) Configure paths (optional)

Override defaults using environment variables:

PEPNET_DATA_PATH (default ./data/biolip1.pt)

PEPNET_MODEL_PATH (default ./results/models/pepnet_model.pt)

PEPNET_TRAIN_SCRIPT_PATH (default ./train/train.py)

PEPNET_EVAL_SCRIPT_PATH (default ./train/eval.py)

PEPNET_PREDICT_SCRIPT_PATH (default ./predict.py)

PEPNET_PREDICTION_PATH (default ./results/predictions)

Examples

# Biolip
$env:PEPNET_DATA_PATH  = "$PWD\data\biolip1.pt"
$env:PEPNET_MODEL_PATH = "$PWD\results\models\pepnet_model.pt"

# Self-built
$env:PEPNET_DATA_PATH  = "$PWD\data\gnn_crossmodal_with_mask1.pt"
$env:PEPNET_MODEL_PATH = "$PWD\results\models\pepnet_model4_seed42.pt"

# Biolip
export PEPNET_DATA_PATH="$PWD/data/biolip1.pt"
export PEPNET_MODEL_PATH="$PWD/results/models/pepnet_model.pt"

# Self-built
export PEPNET_DATA_PATH="$PWD/data/gnn_crossmodal_with_mask1.pt"
export PEPNET_MODEL_PATH="$PWD/results/models/pepnet_model4_seed42.pt"

(b) One-click entry
python main.py


Menu:

1 — Train (AdamW + AMP + EMA + cosine LR with warmup; default 9:1 split)

2 — Evaluate (isotonic/temperature calibration + MCC threshold sweep; returns 7 metrics)

3 — Predict (exports residue-level CSV on the validation split)

5) Prediction output (CSV)

Saved to ./results/predictions/ (e.g., predicted_pepnet_biolip1.csv).

Columns

Column	Description
pdb_id	PDB identifier
chain_id	Protein chain ID
residue_index	Residue index (prefers residue_ids; falls back to 0..N-1)
true_label	Ground truth (0/1)
predicted_prob	Probability sigmoid(logit) in [0, 1]

Example

pdb_id,chain_id,residue_index,true_label,predicted_prob
1abc,A,42,1,0.9132
1abc,A,43,0,0.1875
1abc,A,44,1,0.7349

6) Reproducibility notes

Fixed seeds for torch, random, and numpy are set in config.py.

Evaluation mirrors training: isotonic/temperature calibration + MCC threshold search
(default scan [0.28, 0.55] with step 0.002).

EMA (decay = 0.999) is used during training; evaluation prefers EMA weights if present.

predict.py prints absolute data_path, model_path, and save_dir to aid debugging.

7) Project layout
Pepnet/
├─ data/                          # .pt datasets (keep .gitkeep if empty)
├─ models/                        # model components
├─ results/
│  ├─ models/                     # saved weights (.pt)
│  └─ predictions/                # prediction CSVs
├─ train/
│  ├─ train.py  ├─ eval.py
│  ├─ calibration.py ├─ thresholds.py ├─ ema.py
├─ config.py  ├─ main.py  ├─ predict.py
└─ requirements.txt  └─ README.md

8) Citation

If you use PepNet in your research, please cite this repository (and your paper if available).
