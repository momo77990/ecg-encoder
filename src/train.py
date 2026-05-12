"""ECG Transformer 训练入口。

用法:
    python -m src.train --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import MITBIHDataset, compute_class_weights
from src.data.mitbih import CLASS_NAMES, NUM_CLASSES
from src.model import ECGClassifier
from src.utils import (
    compute_metrics,
    confusion_matrix_str,
    cosine_with_warmup,
    count_params,
    ensure_dir,
    load_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument(
        "--smoke",
        action="store_true"
    )
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for sig, label in loader:
        sig = sig.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(sig)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        targets.append(label.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    metrics = compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES)
    return metrics, y_true, y_pred


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    # ---------- 数据 ----------
    train_ds = MITBIHDataset(cfg["data"]["data_dir"], split="train")
    test_ds = MITBIHDataset(cfg["data"]["data_dir"], split="test")
    print(f"[data] train={len(train_ds)} test={len(test_ds)}")

    if args.smoke:
        # smoke: 分层抽样保证每类都有样本（顺序前缀会全是 N 类）
        from torch.utils.data import Subset

        rng = np.random.default_rng(cfg.get("seed", 42))
        n_per_class_train = max(1, cfg["data"]["batch_size"] * 64 // NUM_CLASSES)
        n_per_class_test = max(1, 2048 // NUM_CLASSES)

        def stratified_indices(labels: np.ndarray, n_per_class: int) -> list[int]:
            idx: list[int] = []
            for c in range(NUM_CLASSES):
                pool = np.where(labels == c)[0]
                if len(pool) == 0:
                    continue
                pick = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
                idx.extend(pick.tolist())
            rng.shuffle(idx)
            return idx

        train_ds = Subset(train_ds, stratified_indices(train_ds.labels, n_per_class_train))
        test_ds = Subset(test_ds, stratified_indices(test_ds.labels, n_per_class_test))
        print(f"[smoke] train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # ---------- 模型 ----------
    model = ECGClassifier.from_config(cfg["model"]).to(device)
    print(f"[model] {count_params(model):,} params")

    # ---------- 损失 ----------
    if cfg["data"].get("use_class_weights", True):
        if isinstance(train_ds, torch.utils.data.Subset):
            base_labels = train_ds.dataset.labels[train_ds.indices]
        else:
            base_labels = train_ds.labels
        class_weights = compute_class_weights(base_labels, num_classes=NUM_CLASSES).to(device)
        print(f"[loss] class_weights = {class_weights.tolist()}")
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---------- 优化器 + 调度 ----------
    epochs = 1 if args.smoke else cfg["train"]["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=cfg["train"]["warmup_ratio"],
    )

    use_amp = cfg["train"].get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------- 输出目录 ----------
    log_dir = cfg["train"]["log_dir"]
    ckpt_dir = cfg["train"].get("ckpt_dir", log_dir)
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # ---------- 训练循环 ----------
    best_macro_f1 = -1.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for sig, label in pbar:
            sig = sig.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(sig)
                loss = criterion(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item()) * sig.size(0)
            n_seen += sig.size(0)
            global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, n_seen)
        elapsed = time.time() - t0

        metrics, _, _ = evaluate(model, test_loader, device, use_amp=use_amp)
        msg = (
            f"[epoch {epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"test_acc={metrics['accuracy']:.4f} "
            f"test_macro_f1={metrics['macro_f1']:.4f} "
            f"per_class_f1={[f'{x:.3f}' for x in metrics['per_class_f1']]} "
            f"({elapsed:.1f}s)"
        )
        print(msg)
        writer.add_scalar("eval/accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("eval/macro_f1", metrics["macro_f1"], epoch)
        for i, f in enumerate(metrics["per_class_f1"]):
            writer.add_scalar(f"eval/f1_{CLASS_NAMES[i]}", f, epoch)

        # 保存 best
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            ckpt_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                ckpt_path,
            )
            print(f"  ↳ saved best ckpt → {ckpt_path} (macro_f1={best_macro_f1:.4f})")

    # 训练结束: 输出最终混淆矩阵
    metrics, y_true, y_pred = evaluate(model, test_loader, device, use_amp=use_amp)
    print("\n[final] confusion matrix on test:")
    print(confusion_matrix_str(y_true, y_pred, CLASS_NAMES))
    print(f"[final] best test macro_f1 = {best_macro_f1:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
