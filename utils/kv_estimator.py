# -*- coding: utf-8 -*-
# utils/kv_estimator.py
from dataclasses import dataclass

@dataclass
class KVParams:
    n_layers: int = 24
    n_heads: int = 32
    head_dim: int = 128
    dtype_bytes: int = 2    # fp16
    # 视觉 token/多模态补偿（若有图像）
    vis_tokens_per_image: int = 0

def kv_bytes_from_tokens(n_tokens: int, p: KVParams) -> int:
    # 2 表示 K 和 V；粗略估计足以做“相对比较”
    return int(n_tokens * p.n_layers * p.n_heads * p.head_dim * 2 * p.dtype_bytes)

def kv_bytes_for_groups(token_counts: list[int], p: KVParams) -> int:
    return sum(kv_bytes_from_tokens(t, p) for t in token_counts)
