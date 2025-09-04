# 三元代理 V/H/C（Δlogit/注意力噪声/KV字节）

import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F


def delta_logit_gain(model, group, target_text, L=3) -> float:
    """加入组后目标/候选子目标的 Δ-logit；若无法做早层，退化用 CLIP 相似度。"""
    if not group.tokens or not target_text:
        return 0.0
    
    try:
        # 尝试使用早层接口计算Δ-logit
        return compute_early_layer_delta_logit(model, group, target_text, L)
    except (AttributeError, NotImplementedError):
        # 退化到CLIP相似度计算
        return compute_clip_similarity_fallback(group, target_text)


def compute_early_layer_delta_logit(model, group, target_text, L):
    """使用早层接口计算Δ-logit增益"""
    # 这里应该调用模型的早层接口
    # 由于大多数模型没有直接暴露早层接口，这里提供一个框架实现
    
    # 构建包含和不包含组的输入
    base_tokens = encode_target_only(target_text)
    enhanced_tokens = base_tokens + group.tokens
    
    # 计算logit差异（简化实现）
    base_logit = compute_target_logit(model, base_tokens, target_text)
    enhanced_logit = compute_target_logit(model, enhanced_tokens, target_text)
    
    delta_logit = enhanced_logit - base_logit
    return max(0.0, delta_logit)  # 确保非负


def compute_target_logit(model, tokens, target_text):
    """计算目标词的logit值（简化实现）"""
    # 在实际实现中，这里应该调用模型的forward pass
    # 并提取目标词的logit值
    
    # 简化：基于token和目标词的匹配度估算logit
    target_words = target_text.lower().split()
    max_logit = 0.0
    
    for token in tokens:
        token_lower = token.lower()
        for target_word in target_words:
            if target_word in token_lower or token_lower in target_word:
                # 计算匹配度
                match_ratio = min(len(target_word), len(token_lower)) / max(len(target_word), len(token_lower))
                logit = match_ratio * 2.0  # 缩放到合理范围
                max_logit = max(max_logit, logit)
    
    return max_logit


def compute_clip_similarity_fallback(group, target_text):
    """使用CLIP相似度作为Δ-logit的退化方案"""
    if not group.tokens or not target_text:
        return 0.0
    
    # 计算组内tokens与目标文本的平均相似度
    similarities = []
    target_words = target_text.lower().split()
    
    for token in group.tokens:
        token_lower = token.lower()
        max_sim = 0.0
        
        for target_word in target_words:
            # 简单的字符串相似度
            if target_word in token_lower or token_lower in target_word:
                sim = min(len(target_word), len(token_lower)) / max(len(target_word), len(token_lower))
                max_sim = max(max_sim, sim)
        
        # 特殊token的相似度调整
        if "GOAL:" in token:
            max_sim = 1.0
        elif "FRONTIER" in token:
            max_sim = max_sim * 0.8
        elif "HISTORY" in token:
            max_sim = max_sim * 0.6
        
        similarities.append(max_sim)
    
    # 返回平均相似度作为Δ-logit的代理
    return np.mean(similarities) if similarities else 0.0


def attention_noise_increment(model, group, target_text, L=3) -> float:
    """组对"与目标无关集合"的注意力增量比值（早层 softmax 统计）"""
    if not group.tokens or not target_text:
        return 0.0
    
    try:
        # 尝试使用早层接口计算注意力噪声
        return compute_early_layer_attention_noise(model, group, target_text, L)
    except (AttributeError, NotImplementedError):
        # 退化到启发式计算
        return compute_heuristic_attention_noise(group, target_text)


def compute_early_layer_attention_noise(model, group, target_text, L):
    """使用早层接口计算注意力噪声增量"""
    # 这里应该调用模型的早层注意力接口
    # 计算组对无关token的注意力分布变化
    
    # 构建无关token集合
    irrelevant_tokens = generate_irrelevant_tokens(target_text)
    
    # 计算基线注意力分布
    base_attention = compute_attention_distribution(model, [], irrelevant_tokens, L)
    
    # 计算加入组后的注意力分布
    enhanced_attention = compute_attention_distribution(model, group.tokens, irrelevant_tokens, L)
    
    # 计算注意力增量
    attention_increment = np.mean(enhanced_attention - base_attention)
    return max(0.0, attention_increment)


def generate_irrelevant_tokens(target_text):
    """生成与目标无关的token集合"""
    # 简化的无关token生成
    irrelevant_tokens = [
        "WALL", "FLOOR", "CEILING", "DOOR", "WINDOW",
        "TABLE", "LAMP", "PICTURE", "BOOK", "PEN"
    ]
    
    # 移除与目标相关的token
    target_words = target_text.lower().split()
    filtered_tokens = []
    
    for token in irrelevant_tokens:
        is_relevant = False
        for target_word in target_words:
            if target_word in token.lower():
                is_relevant = True
                break
        if not is_relevant:
            filtered_tokens.append(token)
    
    return filtered_tokens


def compute_attention_distribution(model, group_tokens, irrelevant_tokens, L):
    """计算注意力分布（简化实现）"""
    # 在实际实现中，这里应该调用模型的注意力机制
    
    # 简化：基于token类型和数量估算注意力分布
    total_tokens = len(group_tokens) + len(irrelevant_tokens)
    if total_tokens == 0:
        return np.zeros(len(irrelevant_tokens))
    
    # 分配注意力权重
    attention_weights = np.ones(len(irrelevant_tokens)) * 0.1  # 基础权重
    
    # 根据组tokens调整注意力
    for token in group_tokens:
        if "GOAL:" in token:
            # 目标相关token会减少对无关token的注意力
            attention_weights *= 0.8
        elif "FRONTIER" in token:
            # 前沿token会适度减少注意力
            attention_weights *= 0.9
        elif "HISTORY" in token:
            # 历史token影响较小
            attention_weights *= 0.95
    
    return attention_weights


def compute_heuristic_attention_noise(group, target_text):
    """启发式计算注意力噪声"""
    if not group.tokens or not target_text:
        return 0.0
    
    # 基于组内token类型估算注意力噪声
    noise_score = 0.0
    target_words = target_text.lower().split()
    
    for token in group.tokens:
        token_lower = token.lower()
        
        # 检查token是否与目标相关
        is_target_related = False
        for target_word in target_words:
            if target_word in token_lower or token_lower in target_word:
                is_target_related = True
                break
        
        if is_target_related:
            # 目标相关token减少噪声
            if "GOAL:" in token:
                noise_score -= 0.2
            elif "FRONTIER" in token:
                noise_score -= 0.1
        else:
            # 无关token增加噪声
            if "HISTORY" in token:
                noise_score += 0.05
            else:
                noise_score += 0.1
    
    return max(0.0, noise_score)


def estimate_cost_bytes(group) -> int:
    """估算组的成本字节数"""
    return group.kv_bytes + len(group.sig)


def encode_target_only(target_text):
    """仅编码目标文本的tokens"""
    if not target_text:
        return []
    
    return [f"GOAL:{target_text}"]


def score_groups(groups, model, target_text, cfg):
    """为所有组计算三元代理指标"""
    for g in groups:
        g.v_hat = delta_logit_gain(model, g, target_text, cfg.early_layers)
        g.h_hat = attention_noise_increment(model, g, target_text, cfg.early_layers)
        g.c_hat = estimate_cost_bytes(g)
