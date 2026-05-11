"""ECG Transformer 评估入口。

用法:
    python -m src.evaluate --ckpt runs/ecg_tiny/best.pt
    python -m src.evaluate --ckpt runs/ecg_tiny/best.pt --split test --batch-size 256

输出: accuracy / macro F1 / 每类 F1 / classification_report / 混淆矩阵
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data import MITBIHDataset
from src.data.mitbih import CLASS_NAMES, NUM_CLASSES
from src.model import ECGClassifier
from src.train import evaluate
from src.utils import confusion_matrix_str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt 路径")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--data-dir", type=str, default=None,
                   help="覆盖 ckpt 中保存的 data_dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"[ckpt] saved at epoch {ckpt.get('epoch', '?')}, "
          f"train metrics: {ckpt.get('metrics', {})}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGClassifier.from_config(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])

    data_dir = args.data_dir or cfg["data"]["data_dir"]
    ds = MITBIHDataset(data_dir, split=args.split)
    print(f"[data] split={args.split}, n={len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    use_amp = device.type == "cuda"
    metrics, y_true, y_pred = evaluate(model, loader, device, use_amp=use_amp)

    print("\n=== Metrics ===")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  per-class F1: " + ", ".join(
        f"{n}={f:.3f}" for n, f in zip(CLASS_NAMES, metrics["per_class_f1"])
    ))

    print("\n=== Classification report ===")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    print("=== Confusion matrix (rows=truth, cols=pred) ===")
    print(confusion_matrix_str(y_true, y_pred, CLASS_NAMES))


if __name__ == "__main__":
    main()
