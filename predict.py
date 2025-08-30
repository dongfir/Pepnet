# predict.py
import os
import sys
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from models.pepnet_model import PepNet
from config import Config as cfg
from tqdm import tqdm

def main():
    print("[INFO] Python:", sys.executable)
    print("[INFO] CWD:", os.path.abspath(os.getcwd()))
    print("[INFO] data_path:", os.path.abspath(cfg.data_path))
    print("[INFO] model_path:", os.path.abspath(cfg.model_save_path))
    print("[INFO] save_dir:", os.path.abspath(cfg.prediction_save_path))
    os.makedirs(cfg.prediction_save_path, exist_ok=True)

    device = torch.device(cfg.device)

    # 加载模型
    model = PepNet(
        in_channels=cfg.in_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        use_global_task=cfg.use_global, use_mask_task=cfg.use_mask,
        dropout=cfg.dropout, head_type=getattr(cfg, "head_type", "cosine")
    )
    state = torch.load(cfg.model_save_path, map_location=device)
    model.load_state_dict(state if isinstance(state, dict) else state["model"])
    model.to(device).eval()
    print("[INFO] 模型已加载。")

    # 加载数据集
    dataset = torch.load(cfg.data_path, weights_only=False)
    split = int((1 - getattr(cfg, "val_ratio", 0.1)) * len(dataset))
    val_dataset = dataset[split:]
    print(f"[INFO] 数据集大小：total={len(dataset)}, val={len(val_dataset)}")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Predicting"):
            data = data.to(device)
            out = model(data)
            logits = out["node_logits"].squeeze()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_true = data.y.detach().cpu().numpy()

            # 取元数据
            def _safe(item, default="unknown"):
                try:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        return item[0]
                    return item
                except Exception:
                    return default

            pdb_id  = _safe(getattr(data, "pdb_id", "unknown"))
            chain_id = _safe(getattr(data, "protein_chain_id", "unknown"))
            residue_ids = getattr(data, "residue_ids", None)
            if residue_ids is None:
                residue_ids = list(range(len(y_true)))
            else:
                # 转为可索引的 list
                try:
                    residue_ids = residue_ids.tolist()
                except Exception:
                    residue_ids = list(residue_ids)

            for i, (prob, label) in enumerate(zip(probs, y_true)):
                results.append({
                    "pdb_id": str(pdb_id),
                    "chain_id": str(chain_id),
                    "residue_index": int(residue_ids[i]) if i < len(residue_ids) else int(i),
                    "true_label": int(label),
                    "predicted_prob": float(prob),
                })

    df = pd.DataFrame(results)
    save_path = os.path.abspath(os.path.join(cfg.prediction_save_path, "predicted_pepnet_biolip1.csv"))
    df.to_csv(save_path, index=False)
    print(f"[INFO] ✅ 已保存模型预测结果到: {save_path}")
    print(f"[INFO] 行数: {len(df)}")

if __name__ == "__main__":
    main()
