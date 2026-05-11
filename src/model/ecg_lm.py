"""ECG → LLM 桥接模块（LLaVA 风格）。

冻结 encoder + 冻结 LLM，只训练 projector。

数据流：
    ECG (B, 1, 187)
      → encoder (frozen) → (B, 25, 256)
      → projector (trainable) → (B, 25, llm_hidden)
      → 拼接到 [prefix_text, ECG_embeds, suffix_text, target_text] 的 inputs_embeds
      → LLM forward (frozen) → CE loss on target tokens only
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .ecg_encoder import ECGEncoder
from .projector import MLPProjector


# ChatML 模板片段（Qwen2.5 / Qwen2 系列原生支持）。
# {sys} 替换成 system message；user prompt 紧跟在 ECG 之后。
_PREFIX_TEMPLATE = "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n"
_SUFFIX_TEMPLATE = "\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
_TARGET_SUFFIX = "<|im_end|>"        # 加在每个 caption 末尾


class ECGLanguageModel(nn.Module):
    """ECG encoder + projector + LLM 的顶层包装。

    encoder 和 llm 在初始化时已被设为 eval() 且 requires_grad=False。
    """
    def __init__(
        self,
        encoder: ECGEncoder,
        projector: MLPProjector,
        llm: nn.Module,
        tokenizer,
        system_msg: str = "You are a cardiologist analyzing single-heartbeat ECG signals.",
        user_prompt: str = "Describe this heartbeat.",
        captions: Optional[dict[int, str]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.llm = llm
        self.tokenizer = tokenizer
        self.captions_map = captions  # {label: caption_str}

        # 冻结 encoder 和 LLM
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad = False

        # ---------------- 预 tokenize 固定文本 ----------------
        prefix_str = _PREFIX_TEMPLATE.format(sys=system_msg)
        suffix_str = _SUFFIX_TEMPLATE.format(user_prompt=user_prompt)

        # 对应的 token ids（不加 special tokens，模板里已有 <|im_start|> 等）
        prefix_ids = tokenizer(prefix_str, add_special_tokens=False, return_tensors="pt").input_ids[0]
        suffix_ids = tokenizer(suffix_str, add_special_tokens=False, return_tensors="pt").input_ids[0]
        self.register_buffer("prefix_ids", prefix_ids, persistent=False)
        self.register_buffer("suffix_ids", suffix_ids, persistent=False)
        self.prefix_len = int(prefix_ids.shape[0])
        self.suffix_len = int(suffix_ids.shape[0])

        # 预 tokenize 5 个 caption（含末尾 <|im_end|>）
        self.target_ids: dict[int, torch.Tensor] = {}
        max_len = 0
        if captions is not None:
            for k, txt in captions.items():
                ids = tokenizer(
                    txt + _TARGET_SUFFIX,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids[0]
                self.target_ids[int(k)] = ids
                max_len = max(max_len, int(ids.shape[0]))
        self.max_target_len = max_len

        # pad token: Qwen2.5 自带 pad_token；兜底用 eos
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = int(tokenizer.pad_token_id)

    # ---------------- 工具方法 ----------------

    @property
    def llm_dtype(self) -> torch.dtype:
        return next(self.llm.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.llm.parameters()).device

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """通过 LLM 的 embedding 层把 token ids 转成向量。"""
        return self.llm.get_input_embeddings()(ids.to(self.device))

    def _encode_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        """ECG → projector 输出 embeds。

        ecg: (B, 1, 187) → encoder → (B, 25, 256) → projector → (B, 25, llm_hidden)
        encoder 不参与梯度。projector 参与。最终 cast 到 LLM dtype。
        """
        with torch.no_grad():
            feats = self.encoder(ecg)         # (B, 25, 256), float32
        # 保留 fp32 跑 projector，结尾 cast 到 LLM dtype（通常 BF16）
        embeds = self.projector(feats.float())            # (B, 25, llm_hidden)
        return embeds.to(self.llm_dtype)

    # ---------------- 训练 forward ----------------

    def forward(
        self,
        ecg: torch.Tensor,
        labels_int: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """构造 inputs_embeds + labels 跑 LLM，返回 LM loss。

        ecg:        (B, 1, T)
        labels_int: (B,) int64 类别 id (0-4)
        return: {"loss": scalar, "logits": (B, L, V)}
        """
        if self.captions_map is None:
            raise RuntimeError("captions must be provided to use forward()")

        B = ecg.size(0)
        device = self.device

        # 1. ECG → embeds
        ecg_embeds = self._encode_ecg(ecg)                       # (B, N_ecg, H)
        N_ecg = ecg_embeds.size(1)

        # 2. prefix / suffix → embeds (broadcast 到 batch)
        prefix_emb = self._embed_tokens(self.prefix_ids).unsqueeze(0).expand(B, -1, -1)  # (B, P, H)
        suffix_emb = self._embed_tokens(self.suffix_ids).unsqueeze(0).expand(B, -1, -1)  # (B, S, H)

        # 3. 每条样本的 target token ids，right-pad 到 max_target_len
        max_t = self.max_target_len
        tgt_ids_padded = torch.full((B, max_t), self.pad_token_id, dtype=torch.long, device=device)
        tgt_attn = torch.zeros((B, max_t), dtype=torch.long, device=device)
        labels_for_target = torch.full((B, max_t), -100, dtype=torch.long, device=device)
        for i, lab in enumerate(labels_int.tolist()):
            ids = self.target_ids[int(lab)].to(device)
            L = ids.shape[0]
            tgt_ids_padded[i, :L] = ids
            tgt_attn[i, :L] = 1
            labels_for_target[i, :L] = ids        # 仅这些位置参与 loss

        target_emb = self._embed_tokens(tgt_ids_padded)          # (B, max_t, H)

        # 4. 拼接 inputs_embeds
        inputs_embeds = torch.cat(
            [prefix_emb, ecg_embeds, suffix_emb, target_emb], dim=1
        )                                                         # (B, P + N_ecg + S + max_t, H)
        total_len = inputs_embeds.size(1)

        # 5. attention_mask: prefix+ECG+suffix 全 1, target 段 tgt_attn
        attn_prefix = torch.ones((B, self.prefix_len + N_ecg + self.suffix_len), dtype=torch.long, device=device)
        attn_mask = torch.cat([attn_prefix, tgt_attn], dim=1)     # (B, total_len)

        # 6. labels: prefix+ECG+suffix 段全 -100，target 段为 labels_for_target
        labels = torch.full((B, total_len), -100, dtype=torch.long, device=device)
        labels[:, self.prefix_len + N_ecg + self.suffix_len :] = labels_for_target

        # 7. LLM forward (frozen, but grads flow)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )
        return {"loss": outputs.loss, "logits": outputs.logits}

    # ---------------- 推理 generate ----------------

    @torch.no_grad()
    def generate(
        self,
        ecg: torch.Tensor,
        max_new_tokens: int = 32,
    ) -> list[str]:
        """从 ECG 生成文本。返回 list[str]，长度 = batch size。"""
        B = ecg.size(0)
        ecg_embeds = self._encode_ecg(ecg)
        N_ecg = ecg_embeds.size(1)
        prefix_emb = self._embed_tokens(self.prefix_ids).unsqueeze(0).expand(B, -1, -1)
        suffix_emb = self._embed_tokens(self.suffix_ids).unsqueeze(0).expand(B, -1, -1)

        inputs_embeds = torch.cat([prefix_emb, ecg_embeds, suffix_emb], dim=1)
        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        # HF generate(inputs_embeds=) 只返回新生成的 token，不含输入
        out_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # 解码（去掉 special tokens 让输出更干净）
        texts = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return texts

    # ---------------- ckpt 友好 ----------------

    def projector_state_dict(self) -> dict:
        """只返回 projector 权重，用于 best ckpt 保存。"""
        return self.projector.state_dict()

    def load_projector_state_dict(self, state_dict: dict) -> None:
        self.projector.load_state_dict(state_dict)
