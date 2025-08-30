PepNet

Residue-level peptide–protein prediction via a cross-modal graph network.

Install Dependencies

Option A (recommended)
pip install -r requirements.txt

Option B (pin common versions)

pip install torch==2.7.1+cu118
pip install torch-geometric==2.6.1
pip install numpy==2.3.1
pip install scikit-learn==1.7.1
pip install tqdm==4.67.1
pip install pandas
Optional for plotting
pip install matplotlib seaborn


No HHblits/ProtTrans required; data are loaded from .pt directly.


Description

PepNet uses dual encoders (sequence & structural GNN), cross-modal attention, and a residual fusion block. The classifier head is noise-robust (cosine-margin + gate-based adaptive fusion + sample-wise weights).


Training supports EMA, AMP mixed precision, cosine decay with warmup, and gradient clipping. Evaluation applies isotonic/temperature calibration and MCC-based threshold search, aligned with training.


Datasets

Put your processed dataset at:

./data/biolip1.pt


Or override via environment variable:


Windows PowerShell

$env:PEPNET_DATA_PATH="D:\path\to\biolip1.pt"


Linux/macOS

export PEPNET_DATA_PATH=/data/biolip1.pt

.pt file structure (PyG Data per graph)


torch.load(path, weights_only=False) should return an indexable sequence (e.g., list) of torch_geometric.data.Data. Fields:

Field	Shape / dtype	Required	Notes (CN)
x	FloatTensor [N, 256]		
edge_index	LongTensor [2, E]		
y	Float/LongTensor [N]		
pdb_id	str or list[str]	
protein_chain_id	str or list[str]	
residue_ids	Tensor/List [N]	

Quick sanity check

import torch
ds = torch.load("./data/biolip1.pt", weights_only=False)
d0 = ds[0]
print("len:", len(ds))
print("x:", d0.x.shape, d0.x.dtype)
print("edge_index:", d0.edge_index.shape, d0.edge_index.dtype)
print("y:", d0.y.shape, d0.y.dtype)

Instructions for Running the Model
1) Configure paths (optional)



PEPNET_MODEL_PATH (default ./results/models/pepnet_model4_seed77777.pt)

PEPNET_TRAIN_SCRIPT_PATH (default ./train/train.py)

PEPNET_EVAL_SCRIPT_PATH (default ./train/eval.py)

PEPNET_PREDICT_SCRIPT_PATH (default ./predict.py)

PEPNET_PREDICTION_PATH (default ./results/predictions)

2) Train / Evaluate / Predict (one-click entry)



python main.py
Choose / ：
1 Train  （AdamW + AMP + EMA + cosine warmup；default 9:1 split ）
2 Evaluate（Isotonic/Temp calibration + MCC threshold sweep；7 metrics ）
3 Predict （Export residue-level CSV for the validation set ）

Prediction Output (CSV)

File generated at:

./results/predictions/predicted_pepnet_biolip1.csv


Columns:

pdb_id — PDB identifier 

chain_id — Protein chain ID 

residue_index — Residue index 

true_label — Ground-truth (0/1)

predicted_prob — Probability after sigmoid(logit) (0–1)

Example

pdb_id,chain_id,residue_index,true_label,predicted_prob
1abc,A,42,1,0.9132
1abc,A,43,0,0.1875
1abc,A,44,1,0.7349
