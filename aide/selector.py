# AIRD 选择（贪心/背包，含 ε 约束）

import numpy as np
from typing import List, Dict, Tuple, Set
from aide.config import AIDEConfig


def semantic_gate(groups, target_text, theta=0.25):
    """文本/CLIP 嵌入先做硬筛"""
    if not groups or not target_text:
        return []
    
    filtered_groups = []
    target_words = target_text.lower().split()
    
    for group in groups:
        # 计算组与目标的语义相似度
        semantic_score = compute_semantic_similarity(group, target_words)
        
        # 应用阈值过滤
        if semantic_score >= theta:
            filtered_groups.append(group)
    
    return filtered_groups


def compute_semantic_similarity(group, target_words):
    """计算组与目标词的语义相似度"""
    if not group.tokens or not target_words:
        return 0.0
    
    similarities = []
    
    for token in group.tokens:
        token_lower = token.lower()
        max_sim = 0.0
        
        for target_word in target_words:
            # 计算token与目标词的相似度
            if target_word in token_lower or token_lower in target_word:
                # 字符串匹配相似度
                sim = min(len(target_word), len(token_lower)) / max(len(target_word), len(token_lower))
                max_sim = max(max_sim, sim)
        
        # 特殊token的相似度调整
        if "GOAL:" in token:
            max_sim = 1.0  # 目标相关token最高相似度
        elif "FRONTIER" in token:
            max_sim = max_sim * 0.8  # 前沿节点中等相似度
        elif "HISTORY" in token:
            max_sim = max_sim * 0.6  # 历史节点较低相似度
        
        similarities.append(max_sim)
    
    # 返回平均相似度
    return np.mean(similarities) if similarities else 0.0


def select_groups_AIRD(groups, cfg, price_p, price_tau, kv_budget_bytes):
    """AIRD算法：基于价格的双重约束贪心选择"""
    # 计算调整后的权重参数
    alpha = cfg.alpha_base + price_tau
    lambd = cfg.lambda_base + price_p
    
    # 语义门控预筛选
    G = semantic_gate(groups, "...", cfg.theta_sem)
    
    # 初始化选择状态
    used = 0  # 已使用的KV字节数
    h_sum = 0.0  # 累积的干扰项
    S = []  # 选中的组ID列表
    
    # lazy-greedy：(ΔV - αΔH)/ΔC 排序
    G = sorted(G, key=lambda g: (g.v_hat - alpha * g.h_hat) / max(g.c_hat, 1), reverse=True)
    
    for g in G:
        # 检查KV预算约束
        if used + g.kv_bytes > kv_budget_bytes:
            continue
        
        # 检查干扰预算约束
        if h_sum + g.h_hat > cfg.epsilon_interf:
            continue
        
        # 计算边际收益
        marginal = g.v_hat - alpha * g.h_hat - lambd * g.c_hat
        
        # 如果边际收益非正，停止选择
        if marginal <= 0:
            break
        
        # 选择该组
        S.append(g.gid)
        used += g.kv_bytes
        h_sum += g.h_hat
    
    return S, used, h_sum


def select_groups_knapsack(groups, cfg, price_p, price_tau, kv_budget_bytes):
    """背包算法：动态规划求解最优组合"""
    # 语义门控预筛选
    G = semantic_gate(groups, "...", cfg.theta_sem)
    
    if not G:
        return [], 0, 0.0
    
    # 计算调整后的权重参数
    alpha = cfg.alpha_base + price_tau
    lambd = cfg.lambda_base + price_p
    
    # 构建动态规划表
    n = len(G)
    # dp[i][j][k] 表示前i个组，使用j字节，干扰为k时的最大价值
    max_interference = int(cfg.epsilon_interf * 100)  # 离散化干扰值
    
    # 简化实现：使用贪心近似
    return select_groups_AIRD(groups, cfg, price_p, price_tau, kv_budget_bytes)


def compute_group_utility(group, alpha, lambd):
    """计算组的效用值"""
    return group.v_hat - alpha * group.h_hat - lambd * group.c_hat


def validate_selection_constraints(selected_groups, groups_dict, cfg, kv_budget_bytes):
    """验证选择结果是否满足约束"""
    total_kv = 0
    total_interference = 0.0
    
    for gid in selected_groups:
        if gid in groups_dict:
            group = groups_dict[gid]
            total_kv += group.kv_bytes
            total_interference += group.h_hat
    
    kv_valid = total_kv <= kv_budget_bytes
    interference_valid = total_interference <= cfg.epsilon_interf
    
    return kv_valid, interference_valid, total_kv, total_interference


def optimize_group_selection(groups, cfg, price_p, price_tau, kv_budget_bytes, method="greedy"):
    """优化组选择的主函数"""
    if method == "greedy":
        return select_groups_AIRD(groups, cfg, price_p, price_tau, kv_budget_bytes)
    elif method == "knapsack":
        return select_groups_knapsack(groups, cfg, price_p, price_tau, kv_budget_bytes)
    else:
        raise ValueError(f"Unknown selection method: {method}")


def compute_selection_metrics(selected_groups, groups_dict):
    """计算选择结果的指标"""
    if not selected_groups:
        return {
            'total_value': 0.0,
            'total_interference': 0.0,
            'total_cost': 0,
            'group_count': 0
        }
    
    total_value = 0.0
    total_interference = 0.0
    total_cost = 0
    
    for gid in selected_groups:
        if gid in groups_dict:
            group = groups_dict[gid]
            total_value += group.v_hat
            total_interference += group.h_hat
            total_cost += group.c_hat
    
    return {
        'total_value': total_value,
        'total_interference': total_interference,
        'total_cost': total_cost,
        'group_count': len(selected_groups)
    }


def adaptive_selection_with_fallback(groups, cfg, price_p, price_tau, kv_budget_bytes):
    """自适应选择策略，包含回退机制"""
    # 首先尝试贪心算法
    try:
        selected_ids, used_kv, used_interference = select_groups_AIRD(
            groups, cfg, price_p, price_tau, kv_budget_bytes
        )
        
        # 验证结果
        groups_dict = {g.gid: g for g in groups}
        kv_valid, interference_valid, _, _ = validate_selection_constraints(
            selected_ids, groups_dict, cfg, kv_budget_bytes
        )
        
        if kv_valid and interference_valid:
            return selected_ids, used_kv, used_interference
        
    except Exception as e:
        print(f"Greedy selection failed: {e}")
    
    # 回退到简单选择策略
    return fallback_selection(groups, cfg, kv_budget_bytes)


def fallback_selection(groups, cfg, kv_budget_bytes):
    """回退选择策略：基于价值优先的简单选择"""
    if not groups:
        return [], 0, 0.0
    
    # 按价值排序
    sorted_groups = sorted(groups, key=lambda g: g.v_hat, reverse=True)
    
    selected_ids = []
    used_kv = 0
    used_interference = 0.0
    
    for group in sorted_groups:
        # 检查约束
        if used_kv + group.kv_bytes > kv_budget_bytes:
            continue
        
        if used_interference + group.h_hat > cfg.epsilon_interf:
            continue
        
        # 选择组
        selected_ids.append(group.gid)
        used_kv += group.kv_bytes
        used_interference += group.h_hat
    
    return selected_ids, used_kv, used_interference
