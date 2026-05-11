"""5 类心拍 → 自然语言模板 + 反向解析。

LLaVA 风格的 caption 监督信号。所有模板长度接近，避免标签长度泄露信息。
"""
from __future__ import annotations


CAPTIONS: dict[int, str] = {
    0: "This heartbeat shows a normal sinus pattern.",
    1: "This heartbeat shows a supraventricular ectopic beat.",
    2: "This heartbeat shows a ventricular ectopic beat.",
    3: "This heartbeat shows a fusion of ventricular and normal beats.",
    4: "This heartbeat shows an unclassifiable or paced pattern.",
}


# 关键词→类别映射，按从最特异到最一般的顺序匹配。
# 注意 "supraventricular" 必须早于 "ventricular"，否则后者会先命中。
_KEYWORDS_PRIORITY: list[tuple[int, str]] = [
    (1, "supraventricular"),
    (3, "fusion"),
    (4, "unclassifiable"),
    (4, "paced"),
    (4, "unknown"),
    (2, "ventricular"),       # 必须放在 supraventricular 后
    (0, "normal"),
    (0, "sinus"),             # normal 同义词
]


def label_to_caption(label: int) -> str:
    """label (0-4) → 模板句子。"""
    return CAPTIONS[int(label)]


def caption_to_label(text: str) -> int:
    """生成文本反向解析回类别 id。

    规则：按 _KEYWORDS_PRIORITY 顺序找第一个出现在 text 里的关键词，
    返回对应的 label。都没命中返回 -1。
    """
    text_lower = text.lower()
    for label, kw in _KEYWORDS_PRIORITY:
        if kw in text_lower:
            return label
    return -1
