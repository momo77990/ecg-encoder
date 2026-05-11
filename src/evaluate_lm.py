"""Phase 2 评估入口：从 ECG 生成文本，反向解析回类别，算分类指标。

用法:
    python -m src.evaluate_lm --ckpt runs/ecg_lm/best_projector.pt
    python -m src.evaluate_lm --ckpt runs/ecg_lm/best_projector.pt --num-samples 500
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data import CAPTIONS, MITBIHDataset, caption_to_label
from src.data.mitbih import CLASS_NAMES, NUM_CLASSES
from src.model import ECGClassifier, MLPProjector
from src.model.ecg_lm import ECGLanguageModel
from src.train_lm import evaluate_text, load_frozen_encoder, load_frozen_llm, stratified_subset_indices
from src.utils import confusion_matrix_str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="best_projector.pt 路径")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=2000,
                   help="评测样本数上限。0 = 全量")
    p.add_argument("--show-examples", type=int, default=20,
                   help="额外打印多少个 (生成文本, 真实标签) 例子")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"[ckpt] saved at epoch {ckpt.get('epoch')}, "
          f"train metrics: {ckpt.get('metrics')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 加载冻结的 encoder + LLM ----------
    encoder = load_frozen_encoder(cfg["encoder_ckpt"], device)
    llm, tokenizer = load_frozen_llm(cfg["llm"]["name"], cfg["llm"]["dtype"], device)

    # ---------- 若 ckpt 含 LoRA，注入并加载 LoRA 权重 ----------
    has_lora = "lora" in ckpt and len(ckpt["lora"]) > 0
    if has_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = cfg["lora"]
        config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, config)
        # 加载训好的 LoRA 权重（部分 state_dict）
        missing, unexpected = llm.load_state_dict(ckpt["lora"], strict=False)
        loaded = len(ckpt["lora"]) - len(unexpected)
        print(f"[lora] injected and loaded {loaded} LoRA tensors "
              f"(r={lora_cfg['r']}, alpha={lora_cfg['alpha']}, "
              f"target={lora_cfg['target_modules']})")
    hidden_size = (llm.base_model.model.config.hidden_size if has_lora else llm.config.hidden_size)

    # ---------- projector + 加载训练好的权重 ----------
    projector = MLPProjector(
        in_dim=encoder.d_model,
        out_dim=hidden_size,
        hidden_dim=cfg["projector"].get("hidden_dim", hidden_size),
    ).to(device)
    projector.load_state_dict(ckpt["projector"])
    print(f"[projector] loaded {sum(p.numel() for p in projector.parameters()):,} params")

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
    ds = MITBIHDataset(cfg["data"]["data_dir"], split=args.split)
    print(f"[data] split={args.split}, n={len(ds)}")

    # 分层取子集（默认）：每类抽 num_samples / NUM_CLASSES 个
    if args.num_samples > 0 and args.num_samples < len(ds):
        from torch.utils.data import Subset
        n_per_class = max(1, args.num_samples // NUM_CLASSES)
        idx = stratified_subset_indices(ds.labels, n_per_class=n_per_class)
        ds = Subset(ds, idx)
        print(f"[data] stratified subset: {len(ds)} samples ({n_per_class}/class)")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    metrics, y_true, y_pred, samples = evaluate_text(
        model, loader, device, max_samples=None
    )

    # ---------- 输出 ----------
    n = len(y_true)
    print(f"\n=== Metrics (n={n}) ===")
    print(f"  match_rate (生成可解析率): {metrics['match_rate']:.4f}")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  per-class F1: " + ", ".join(
        f"{n}={f:.3f}" for n, f in zip(CLASS_NAMES, metrics["per_class_f1"])
    ))

    print("\n=== Classification report ===")
    # 把 pred=NUM_CLASSES (NoMatch) 排除在外 — sklearn 自动忽略不在 labels 里的
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    print("=== Confusion matrix (rows=truth, cols=pred) ===")
    print(confusion_matrix_str(y_true, y_pred, CLASS_NAMES + ["NoMatch"]))

    # ---------- 例子 ----------
    if args.show_examples > 0:
        print(f"\n=== Sample generations (first {args.show_examples}) ===")
        # 重新跑一次小 batch 拿原文
        small_loader = DataLoader(
            ds, batch_size=args.show_examples, shuffle=False, num_workers=0
        )
        sig, label = next(iter(small_loader))
        sig = sig.to(device)
        with torch.no_grad():
            texts = model.generate(sig, max_new_tokens=24)
        for i, (txt, lab) in enumerate(zip(texts, label.tolist())):
            parsed = caption_to_label(txt)
            parsed_name = CLASS_NAMES[parsed] if 0 <= parsed < NUM_CLASSES else "NoMatch"
            ok = "✓" if parsed == int(lab) else "✗"
            print(f"  [{i:2d}] {ok} true={CLASS_NAMES[int(lab)]} pred={parsed_name}")
            print(f"       gen='{txt}'")


if __name__ == "__main__":
    main()
