"""Phase 2 训练入口：冻结 encoder + 冻结 LLM，只训 projector。

用法:
    python -m src.train_lm --config config_lm.yaml
    python -m src.train_lm --config config_lm.yaml --smoke
"""
from __future__ import annotations

# 先于 transformers import 设置 offline，避免代理问题
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import CAPTIONS, MITBIHDataset, caption_to_label
from src.data.mitbih import CLASS_NAMES, NUM_CLASSES
from src.model import ECGClassifier, MLPProjector
from src.model.ecg_lm import ECGLanguageModel
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
    p.add_argument("--config", type=str, default="config_lm.yaml")
    p.add_argument("--smoke", action="store_true", help="1 epoch + 32 batch")
    return p.parse_args()


# ---------------- 模型加载 ----------------


def load_frozen_encoder(ckpt_path: str, device: torch.device) -> "torch.nn.Module":
    """从 phase 1 ckpt 加载 ECGEncoder（丢弃 classification head）。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classifier = ECGClassifier.from_config(ckpt["config"]["model"])
    classifier.load_state_dict(ckpt["model"])
    encoder = classifier.encoder
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[encoder] loaded from {ckpt_path} (epoch={ckpt.get('epoch')}, "
          f"trained macro_f1={ckpt['metrics']['macro_f1']:.4f})")
    return encoder


def load_frozen_llm(name: str, dtype: str, device: torch.device):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in llm.parameters())
    print(f"[llm] {name} loaded ({n_params/1e9:.2f}B params, dtype={dtype}, hidden={llm.config.hidden_size})")
    return llm, tokenizer


# ---------------- 评测 ----------------


@torch.no_grad()
def evaluate_text(
    model: ECGLanguageModel,
    loader: DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    """对 loader 逐 batch 生成文本，反向解析回类别，算指标。

    注意：调用方应确保 loader 的样本类别分布合理（已分层采样或 shuffle），
    否则 max_samples 限制下可能只看到一类。
    """
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    sample_outputs: list[str] = []
    n_done = 0
    pbar = tqdm(loader, desc="eval", leave=False)
    for sig, label in pbar:
        sig = sig.to(device, non_blocking=True)
        texts = model.generate(sig, max_new_tokens=24)
        # 第一个 batch 收集前 5 条做 sanity 输出
        if not sample_outputs:
            for txt, lab in zip(texts[:5], label[:5].tolist()):
                sample_outputs.append(f"true={CLASS_NAMES[int(lab)]} | gen='{txt}'")
        for txt, lab in zip(texts, label.tolist()):
            preds.append(caption_to_label(txt))
            targets.append(int(lab))
        n_done += sig.size(0)
        if max_samples is not None and n_done >= max_samples:
            break
    y_true = np.array(targets)
    y_pred = np.array(preds)

    # 没匹配上 (-1) 的算作错误：替换成 num_classes 这个不存在的类别 id
    y_pred_safe = np.where(y_pred < 0, NUM_CLASSES, y_pred)

    metrics = compute_metrics(y_true, y_pred_safe, num_classes=NUM_CLASSES)
    metrics["match_rate"] = float((y_pred >= 0).mean())
    return metrics, y_true, y_pred_safe, sample_outputs


def stratified_subset_indices(
    labels: np.ndarray,
    n_per_class: int,
    seed: int = 42,
    num_classes: int = NUM_CLASSES,
) -> list[int]:
    """从 labels 中每类抽 n_per_class 个，shuffle 返回索引列表。"""
    rng = np.random.default_rng(seed)
    idx: list[int] = []
    for c in range(num_classes):
        pool = np.where(labels == c)[0]
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)
        idx.extend(pick.tolist())
    rng.shuffle(idx)
    return idx


# ---------------- 主函数 ----------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    # ---------- 加载冻结的 encoder + LLM ----------
    encoder = load_frozen_encoder(cfg["encoder_ckpt"], device)
    llm, tokenizer = load_frozen_llm(cfg["llm"]["name"], cfg["llm"]["dtype"], device)

    # ---------- 构造 projector ----------
    proj_in = encoder.d_model
    proj_out = llm.config.hidden_size
    projector = MLPProjector(
        in_dim=proj_in,
        out_dim=proj_out,
        hidden_dim=cfg["projector"].get("hidden_dim", proj_out),
    ).to(device)
    print(f"[projector] {count_params(projector):,} params (in={proj_in}, out={proj_out})")

    model = ECGLanguageModel(
        encoder=encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        system_msg=cfg["system_msg"],
        user_prompt=cfg["user_prompt"],
        captions=CAPTIONS,
    )

    # ---------- 数据 ----------
    train_ds = MITBIHDataset(cfg["data"]["data_dir"], split="train")
    test_ds = MITBIHDataset(cfg["data"]["data_dir"], split="test")
    print(f"[data] train={len(train_ds)} test={len(test_ds)}")

    if args.smoke:
        rng = np.random.default_rng(cfg.get("seed", 42))
        # 分层 small subset
        n_per_class = max(1, cfg["data"]["batch_size"] * 32 // NUM_CLASSES)

        def stratified(labels: np.ndarray, n: int) -> list[int]:
            idx: list[int] = []
            for c in range(NUM_CLASSES):
                pool = np.where(labels == c)[0]
                if len(pool) == 0:
                    continue
                idx.extend(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
            rng.shuffle(idx)
            return idx

        train_ds = Subset(train_ds, stratified(train_ds.labels, n_per_class))
        test_ds = Subset(test_ds, stratified(test_ds.labels, max(1, 256 // NUM_CLASSES)))
        print(f"[smoke] train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # eval set: 用分层抽样保证每类都有样本
    if not args.smoke:
        eval_n_per_class = cfg["data"].get("eval_max_samples", 2000) // NUM_CLASSES
        eval_indices = stratified_subset_indices(
            test_ds.labels if not isinstance(test_ds, Subset) else test_ds.dataset.labels[test_ds.indices],
            n_per_class=eval_n_per_class,
            seed=cfg.get("seed", 42),
        )
        eval_ds = Subset(test_ds, eval_indices) if not isinstance(test_ds, Subset) else Subset(test_ds.dataset, [test_ds.indices[i] for i in eval_indices])
    else:
        eval_ds = test_ds                                   # smoke 时已 stratified
    test_loader = DataLoader(
        eval_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    print(f"[eval] using {len(eval_ds)} samples (stratified)")

    # ---------- 优化器 + 调度 ----------
    epochs = 1 if args.smoke else cfg["train"]["epochs"]
    total_steps = len(train_loader) * epochs
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = cosine_with_warmup(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=cfg["train"]["warmup_ratio"],
    )

    log_dir = cfg["train"]["log_dir"]
    ckpt_dir = cfg["train"].get("ckpt_dir", log_dir)
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)
    writer = SummaryWriter(log_dir=log_dir)

    eval_max = cfg["data"].get("eval_max_samples", 2000)
    best_macro_f1 = -1.0
    global_step = 0

    for epoch in range(epochs):
        projector.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
        for sig, label in pbar:
            sig = sig.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(sig, label)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * sig.size(0)
            n_seen += sig.size(0)
            global_step += 1
            if global_step % 20 == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss = running_loss / max(1, n_seen)
        elapsed = time.time() - t0

        # 评测
        metrics, _, _, samples = evaluate_text(
            model, test_loader, device, max_samples=None     # 用整个 stratified eval_ds
        )
        msg = (
            f"[epoch {epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"test_acc={metrics['accuracy']:.4f} "
            f"test_macro_f1={metrics['macro_f1']:.4f} "
            f"match_rate={metrics['match_rate']:.4f} "
            f"per_class_f1={[f'{x:.3f}' for x in metrics['per_class_f1']]} "
            f"({elapsed:.1f}s)"
        )
        print(msg)
        print("[samples]")
        for s in samples:
            print("   ", s)

        writer.add_scalar("eval/accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("eval/macro_f1", metrics["macro_f1"], epoch)
        writer.add_scalar("eval/match_rate", metrics["match_rate"], epoch)
        for i, f in enumerate(metrics["per_class_f1"]):
            writer.add_scalar(f"eval/f1_{CLASS_NAMES[i]}", f, epoch)

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            ckpt_path = os.path.join(ckpt_dir, "best_projector.pt")
            torch.save(
                {
                    "projector": model.projector_state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                ckpt_path,
            )
            print(f"  ↳ saved best projector → {ckpt_path} (macro_f1={best_macro_f1:.4f})")

    # 最终结果
    metrics, y_true, y_pred, samples = evaluate_text(
        model, test_loader, device, max_samples=None
    )
    print("\n[final] confusion matrix on test:")
    print(confusion_matrix_str(y_true, y_pred, CLASS_NAMES + ["NoMatch"]))
    print(f"[final] best test macro_f1 = {best_macro_f1:.4f}")
    print(f"[final] match_rate = {metrics['match_rate']:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
