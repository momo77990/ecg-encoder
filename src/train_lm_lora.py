"""Phase 3 训练入口：在 Phase 2 best projector 基础上，给 LLM 加 LoRA 微调。

冻结 ECG encoder；训练 projector + Qwen attention LoRA。

用法:
    python -m src.train_lm_lora --config config_lm_lora.yaml
    python -m src.train_lm_lora --config config_lm_lora.yaml --smoke
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import CAPTIONS, MITBIHDataset
from src.data.mitbih import CLASS_NAMES, NUM_CLASSES
from src.model import MLPProjector
from src.model.ecg_lm import ECGLanguageModel
from src.train_lm import (
    evaluate_text,
    load_frozen_encoder,
    load_frozen_llm,
    stratified_subset_indices,
)
from src.utils import (
    confusion_matrix_str,
    cosine_with_warmup,
    count_params,
    ensure_dir,
    load_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config_lm_lora.yaml")
    p.add_argument("--smoke", action="store_true", help="1 epoch + 32 batch")
    return p.parse_args()


def add_lora(llm: torch.nn.Module, lora_cfg: dict) -> torch.nn.Module:
    """给 Qwen attention 层注入 LoRA，只 target q/k/v/o projections。"""
    config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, config)
    llm.print_trainable_parameters()
    return llm


def load_projector_from_phase2(
    ckpt_path: str,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> MLPProjector:
    projector = MLPProjector(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    projector.load_state_dict(ckpt["projector"])
    print(f"[projector] initialized from {ckpt_path} "
          f"(macro_f1={ckpt.get('metrics', {}).get('macro_f1', 'n/a')})")
    return projector


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}")

    # ---------- 冻结 encoder + 加载 LLM ----------
    encoder = load_frozen_encoder(cfg["encoder_ckpt"], device)
    llm, tokenizer = load_frozen_llm(cfg["llm"]["name"], cfg["llm"]["dtype"], device)

    # ---------- 给 LLM 注入 LoRA ----------
    llm = add_lora(llm, cfg["lora"])
    llm.train()
    # PEFT 后 base model 参数仍 frozen；LoRA adapter requires_grad=True

    # ---------- projector 从 Phase 2 best 初始化 ----------
    projector = load_projector_from_phase2(
        cfg["projector"]["init_from"],
        in_dim=encoder.d_model,
        out_dim=llm.base_model.model.config.hidden_size if hasattr(llm, "base_model") else llm.config.hidden_size,
        hidden_dim=cfg["projector"].get("hidden_dim", 1536),
        device=device,
    )
    print(f"[projector] trainable params = {count_params(projector):,}")

    model = ECGLanguageModel(
        encoder=encoder,
        projector=projector,
        llm=llm,
        tokenizer=tokenizer,
        system_msg=cfg["system_msg"],
        user_prompt=cfg["user_prompt"],
        captions=CAPTIONS,
    )

    # ECGLanguageModel.__init__ 会 freeze llm，因此这里重新打开 LoRA adapter 的 requires_grad
    for name, p in model.llm.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    model.llm.train()

    lora_params = [p for n, p in model.llm.named_parameters() if "lora_" in n and p.requires_grad]
    print(f"[lora] trainable params = {sum(p.numel() for p in lora_params):,}")

    # ---------- 数据 ----------
    train_ds = MITBIHDataset(cfg["data"]["data_dir"], split="train")
    test_ds = MITBIHDataset(cfg["data"]["data_dir"], split="test")
    print(f"[data] train={len(train_ds)} test={len(test_ds)}")

    if args.smoke:
        n_per_class = max(1, cfg["data"]["batch_size"] * 32 // NUM_CLASSES)
        train_idx = stratified_subset_indices(train_ds.labels, n_per_class=n_per_class, seed=cfg.get("seed", 42))
        test_idx = stratified_subset_indices(test_ds.labels, n_per_class=max(1, 256 // NUM_CLASSES), seed=cfg.get("seed", 42))
        train_ds = Subset(train_ds, train_idx)
        test_ds = Subset(test_ds, test_idx)
        print(f"[smoke] train={len(train_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    if not args.smoke:
        eval_n_per_class = cfg["data"].get("eval_max_samples", 2000) // NUM_CLASSES
        eval_idx = stratified_subset_indices(test_ds.labels, n_per_class=eval_n_per_class, seed=cfg.get("seed", 42))
        eval_ds = Subset(test_ds, eval_idx)
    else:
        eval_ds = test_ds
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    print(f"[eval] using {len(eval_ds)} samples (stratified)")

    # ---------- optimizer: projector 小 LR, LoRA 较大 LR ----------
    optimizer = torch.optim.AdamW(
        [
            {"params": projector.parameters(), "lr": cfg["train"]["projector_lr"]},
            {"params": lora_params, "lr": cfg["train"]["lora_lr"]},
        ],
        weight_decay=cfg["train"]["weight_decay"],
    )

    epochs = 1 if args.smoke else cfg["train"]["epochs"]
    total_steps = len(train_loader) * epochs
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

    best_macro_f1 = -1.0
    global_step = 0
    for epoch in range(epochs):
        projector.train()
        model.llm.train()
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
            torch.nn.utils.clip_grad_norm_(list(projector.parameters()) + lora_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * sig.size(0)
            n_seen += sig.size(0)
            global_step += 1
            if global_step % 20 == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/projector_lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/lora_lr", scheduler.get_last_lr()[1], global_step)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, n_seen)
        elapsed = time.time() - t0

        metrics, y_true, y_pred, samples = evaluate_text(model, eval_loader, device, max_samples=None)
        print(
            f"[epoch {epoch+1}/{epochs}] train_loss={train_loss:.4f} "
            f"test_acc={metrics['accuracy']:.4f} "
            f"test_macro_f1={metrics['macro_f1']:.4f} "
            f"match_rate={metrics['match_rate']:.4f} "
            f"per_class_f1={[f'{x:.3f}' for x in metrics['per_class_f1']]} "
            f"({elapsed:.1f}s)"
        )
        print("[samples]")
        for s in samples:
            print("   ", s)

        writer.add_scalar("eval/accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("eval/macro_f1", metrics["macro_f1"], epoch)
        writer.add_scalar("eval/match_rate", metrics["match_rate"], epoch)

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            ckpt_path = os.path.join(ckpt_dir, "best_lora.pt")
            torch.save(
                {
                    "projector": model.projector_state_dict(),
                    "lora": {k: v.cpu() for k, v in model.llm.state_dict().items() if "lora_" in k},
                    "config": cfg,
                    "epoch": epoch,
                    "metrics": metrics,
                },
                ckpt_path,
            )
            print(f"  ↳ saved best LoRA ckpt → {ckpt_path} (macro_f1={best_macro_f1:.4f})")

    metrics, y_true, y_pred, _ = evaluate_text(model, eval_loader, device, max_samples=None)
    print("\n[final] confusion matrix on eval:")
    print(confusion_matrix_str(y_true, y_pred, CLASS_NAMES + ["NoMatch"]))
    print(f"[final] best test macro_f1 = {best_macro_f1:.4f}")
    print(f"[final] match_rate = {metrics['match_rate']:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
