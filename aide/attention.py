# 注意力预算 ε 的 logit-penalty/mask

import torch
import math
from typing import Optional, Tuple


def enforce_epsilon_budget(attn_logits, related_mask, epsilon: float):
    """
    若无关项总质量 > ε * 相关项总质量，对无关项 subtract penalty 直到满足；
    由此保证相关注意力份额 ≥ 1/(1+ε)。
    """
    if not torch.any(related_mask):
        # 如果没有相关项，直接返回原始logits
        return attn_logits
    
    # 计算相关项和无关项的logsumexp
    rel = torch.logsumexp(attn_logits[related_mask], dim=-1)
    irr = torch.logsumexp(attn_logits[~related_mask], dim=-1)
    
    # 计算超出预算的部分
    over = irr - (rel + math.log(epsilon))
    penalty = torch.clamp(over, min=0.0)
    
    # 对无关项应用惩罚
    attn_logits[~related_mask] -= penalty.unsqueeze(-1)
    
    return attn_logits


def compute_attention_penalty(attn_logits, related_mask, epsilon: float) -> float:
    """计算需要的注意力惩罚值"""
    if not torch.any(related_mask):
        return 0.0
    
    rel = torch.logsumexp(attn_logits[related_mask], dim=-1)
    irr = torch.logsumexp(attn_logits[~related_mask], dim=-1)
    
    over = irr - (rel + math.log(epsilon))
    return torch.clamp(over, min=0.0).item()


def create_related_mask(attn_logits, target_indices, context_size: int) -> torch.Tensor:
    """创建相关项掩码"""
    batch_size, seq_len = attn_logits.shape[:2]
    related_mask = torch.zeros_like(attn_logits, dtype=torch.bool)
    
    for i, target_idx in enumerate(target_indices):
        if target_idx < seq_len:
            related_mask[i, target_idx] = True
    
    return related_mask


def apply_attention_mask(attn_logits, mask: torch.Tensor, mask_value: float = -1e9):
    """应用注意力掩码"""
    attn_logits = attn_logits.masked_fill(~mask, mask_value)
    return attn_logits


def normalize_attention_weights(attn_logits: torch.Tensor) -> torch.Tensor:
    """归一化注意力权重"""
    return torch.softmax(attn_logits, dim=-1)


def compute_attention_share(attn_weights: torch.Tensor, related_mask: torch.Tensor) -> float:
    """计算相关项的注意力份额"""
    if not torch.any(related_mask):
        return 0.0
    
    related_attention = attn_weights[related_mask].sum()
    total_attention = attn_weights.sum()
    
    return (related_attention / total_attention).item() if total_attention > 0 else 0.0


def validate_epsilon_constraint(attn_weights: torch.Tensor, related_mask: torch.Tensor, epsilon: float) -> bool:
    """验证是否满足ε约束"""
    if not torch.any(related_mask):
        return True
    
    related_share = compute_attention_share(attn_weights, related_mask)
    min_share = 1.0 / (1.0 + epsilon)
    
    return related_share >= min_share


def adaptive_epsilon_enforcement(attn_logits, related_mask, epsilon: float, max_iterations: int = 10):
    """自适应ε约束强制执行"""
    original_logits = attn_logits.clone()
    
    for iteration in range(max_iterations):
        # 应用ε约束
        attn_logits = enforce_epsilon_budget(attn_logits, related_mask, epsilon)
        
        # 计算注意力权重
        attn_weights = normalize_attention_weights(attn_logits)
        
        # 验证约束
        if validate_epsilon_constraint(attn_weights, related_mask, epsilon):
            break
        
        # 如果仍未满足，增加惩罚强度
        if iteration < max_iterations - 1:
            attn_logits = original_logits - (iteration + 1) * 0.1
    
    return attn_logits


def compute_attention_metrics(attn_logits: torch.Tensor, related_mask: torch.Tensor, epsilon: float) -> dict:
    """计算注意力相关指标"""
    # 应用ε约束
    constrained_logits = enforce_epsilon_budget(attn_logits.clone(), related_mask, epsilon)
    
    # 计算注意力权重
    original_weights = normalize_attention_weights(attn_logits)
    constrained_weights = normalize_attention_weights(constrained_logits)
    
    # 计算指标
    original_share = compute_attention_share(original_weights, related_mask)
    constrained_share = compute_attention_share(constrained_weights, related_mask)
    
    # 计算惩罚值
    penalty = compute_attention_penalty(attn_logits, related_mask, epsilon)
    
    # 验证约束
    constraint_satisfied = validate_epsilon_constraint(constrained_weights, related_mask, epsilon)
    
    return {
        'original_share': original_share,
        'constrained_share': constrained_share,
        'penalty': penalty,
        'constraint_satisfied': constraint_satisfied,
        'min_required_share': 1.0 / (1.0 + epsilon),
        'epsilon': epsilon
    }
