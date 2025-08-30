PepNet

Residue-level peptide–protein prediction via a cross-modal graph network.
（残基级肽-蛋白预测：跨模态图网络）
Install Dependencies

Option A (recommended)
pip install -r requirements.txt

Option B (pin common versions)
（手动指定常用版本；请按你的 CUDA 版本选择合适的 torch 轮子）
pip install torch==2.7.1+cu118
pip install torch-geometric==2.6.1
pip install numpy==2.3.1
pip install scikit-learn==1.7.1
pip install tqdm==4.67.1
pip install pandas
# Optional for plotting（可选出图依赖）
# pip install matplotlib seaborn


No HHblits/ProtTrans required; data are loaded from .pt directly.
（无需 HHblits 或 ProtTrans 等外部工具；数据用 .pt 直接读取）

Description

PepNet uses dual encoders (sequence & structural GNN), cross-modal attention, and a residual fusion block. The classifier head is noise-robust (cosine-margin + gate-based adaptive fusion + sample-wise weights).
（序列编码器 + 结构 GNN；跨模态注意力与残差融合；分类头含余弦间隔、门控自适应融合与样本级权重，鲁棒性更好）

Training supports EMA, AMP mixed precision, cosine decay with warmup, and gradient clipping. Evaluation applies isotonic/temperature calibration and MCC-based threshold search, aligned with training.
（训练支持 EMA/AMP/余弦退火+预热/梯度裁剪；评估用等渗/温度校准与基于 MCC 的阈值搜索，与训练严格对齐）

Datasets

Put your processed dataset at:

./data/biolip1.pt


Or override via environment variable:
（也可用环境变量覆盖）

Windows PowerShell

$env:PEPNET_DATA_PATH="D:\path\to\biolip1.pt"


Linux/macOS

export PEPNET_DATA_PATH=/data/biolip1.pt

.pt file structure (PyG Data per graph)

（.pt 输入结构：每个样本为一个 PyG Data）
torch.load(path, weights_only=False) should return an indexable sequence (e.g., list) of torch_geometric.data.Data. Fields:

Field	Shape / dtype	Required	Notes (CN)
x	FloatTensor [N, 256]		残基特征，列维需与 config.py 的 in_dim=256 一致
edge_index	LongTensor [2, E]		图的 COO 边索引（从 0 开始）
y	Float/LongTensor [N]		残基级二分类 0/1
pdb_id	str or list[str]	–	可选；存在则写入预测 CSV
protein_chain_id	str or list[str]	–	可选；存在则写入预测 CSV
residue_ids	Tensor/List [N]	–	可选；若缺省则用 0..N-1

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

（可改 config.py，也可用环境变量覆盖）

PEPNET_MODEL_PATH (default ./results/models/pepnet_model4_seed77777.pt)

PEPNET_TRAIN_SCRIPT_PATH (default ./train/train.py)

PEPNET_EVAL_SCRIPT_PATH (default ./train/eval.py)

PEPNET_PREDICT_SCRIPT_PATH (default ./predict.py)

PEPNET_PREDICTION_PATH (default ./results/predictions)

2) Train / Evaluate / Predict (one-click entry)

（一键入口）

python main.py
# Choose / 选择：
# 1 Train  （AdamW + AMP + EMA + cosine warmup；default 9:1 split / 默认 9:1）
# 2 Evaluate（Isotonic/Temp calibration + MCC threshold sweep；7 metrics / 7 指标）
# 3 Predict （Export residue-level CSV for the validation set / 导出验证集逐残基 CSV）

Prediction Output (CSV)

File generated at:

./results/predictions/predicted_pepnet_biolip1.csv


Columns:

pdb_id — PDB identifier (无该字段则为 unknown)

chain_id — Protein chain ID (无该字段则为 unknown)

residue_index — Residue index (优先 residue_ids，否则 0..N-1)

true_label — Ground-truth (0/1)

predicted_prob — Probability after sigmoid(logit) (0–1)

Example

pdb_id,chain_id,residue_index,true_label,predicted_prob
1abc,A,42,1,0.9132
1abc,A,43,0,0.1875
1abc,A,44,1,0.7349
