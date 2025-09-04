# 目标感知分组（早层亲和 + 空间连通）

import numpy as np
from typing import List, Dict, Tuple
from semantic_mapping import encode_map_to_tokens, group_signature


class Group:
    def __init__(self, gid, tokens):
        self.gid = gid
        self.tokens = tokens
        self.kv_bytes = 0
        self.sig = b""
        self.v_hat = 0.0  # Value proxy (Δlogit)
        self.h_hat = 0.0  # Interference proxy (attention noise)
        self.c_hat = 0.0  # Cost proxy (KV bytes)
        self.has_new = True


def early_qk_affinity(llm, tokens, target_text, L):
    """前 L 层近似目标词 Query 与 token Key 的亲和度；无早层接口时可用 CLIP 替代。"""
    if not tokens or not target_text:
        return np.zeros(len(tokens))
    
    # 简化实现：基于文本相似度的亲和度计算
    # 在实际实现中，这里应该调用LLM的早层接口获取Q-K亲和度
    
    affinities = []
    target_words = target_text.lower().split()
    
    for token in tokens:
        # 计算token与目标词的相似度
        token_lower = token.lower()
        max_sim = 0.0
        
        for target_word in target_words:
            # 简单的字符串匹配相似度
            if target_word in token_lower or token_lower in target_word:
                sim = min(len(target_word), len(token_lower)) / max(len(target_word), len(token_lower))
                max_sim = max(max_sim, sim)
        
        # 特殊token的亲和度调整
        if "GOAL:" in token:
            max_sim = 1.0  # 目标相关token最高亲和度
        elif "FRONTIER" in token:
            max_sim = max_sim * 0.8  # 前沿节点中等亲和度
        elif "HISTORY" in token:
            max_sim = max_sim * 0.6  # 历史节点较低亲和度
        
        affinities.append(max_sim)
    
    return np.array(affinities)


def spatial_semantic_cluster(tokens, aff, max_group_kv):
    """空间邻接 + 亲和度聚类，控制组级 KV 上限"""
    if not tokens:
        return []
    
    n_tokens = len(tokens)
    if n_tokens == 1:
        return [tokens]
    
    # 构建亲和度矩阵
    aff_matrix = np.zeros((n_tokens, n_tokens))
    for i in range(n_tokens):
        for j in range(n_tokens):
            if i == j:
                aff_matrix[i, j] = 1.0
            else:
                # 结合空间邻接性和语义亲和度
                spatial_sim = compute_spatial_similarity(tokens[i], tokens[j])
                semantic_sim = (aff[i] + aff[j]) / 2.0
                aff_matrix[i, j] = 0.7 * spatial_sim + 0.3 * semantic_sim
    
    # 层次聚类
    clusters = hierarchical_clustering(aff_matrix, threshold=0.5)
    
    # 控制组大小以符合KV限制
    final_clusters = []
    for cluster in clusters:
        cluster_tokens = [tokens[i] for i in cluster]
        estimated_kv = estimate_group_kv_bytes(cluster_tokens)
        
        if estimated_kv <= max_group_kv:
            final_clusters.append(cluster_tokens)
        else:
            # 如果组太大，进一步分割
            sub_clusters = split_large_cluster(cluster_tokens, max_group_kv)
            final_clusters.extend(sub_clusters)
    
    return final_clusters


def compute_spatial_similarity(token1, token2):
    """计算两个token的空间相似度"""
    # 提取坐标信息
    coords1 = extract_coordinates(token1)
    coords2 = extract_coordinates(token2)
    
    if coords1 is None or coords2 is None:
        return 0.0
    
    # 计算欧几里得距离
    distance = np.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)
    
    # 转换为相似度（距离越小，相似度越高）
    max_distance = 100.0  # 假设最大距离
    similarity = max(0.0, 1.0 - distance / max_distance)
    
    return similarity


def extract_coordinates(token):
    """从token中提取坐标信息"""
    if "CENTER:" in token:
        try:
            # 解析类似 "FRONTIER_CENTER:(120.5,80.2)" 的格式
            coords_str = token.split("CENTER:(")[1].split(")")[0]
            x, y = map(float, coords_str.split(","))
            return (x, y)
        except:
            return None
    return None


def hierarchical_clustering(aff_matrix, threshold=0.5):
    """简单的层次聚类实现"""
    n = len(aff_matrix)
    clusters = [[i] for i in range(n)]
    
    while len(clusters) > 1:
        best_merge = None
        best_sim = -1
        
        # 找到最相似的两个簇
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = compute_cluster_similarity(aff_matrix, clusters[i], clusters[j])
                if sim > best_sim:
                    best_sim = sim
                    best_merge = (i, j)
        
        # 如果相似度低于阈值，停止合并
        if best_sim < threshold:
            break
        
        # 合并簇
        i, j = best_merge
        clusters[i].extend(clusters[j])
        clusters.pop(j)
    
    return clusters


def compute_cluster_similarity(aff_matrix, cluster1, cluster2):
    """计算两个簇之间的平均相似度"""
    total_sim = 0.0
    count = 0
    
    for i in cluster1:
        for j in cluster2:
            total_sim += aff_matrix[i, j]
            count += 1
    
    return total_sim / count if count > 0 else 0.0


def estimate_group_kv_bytes(tokens):
    """估算组的KV缓存字节数"""
    if not tokens:
        return 0
    
    # 简化的估算：每个token约100字节
    base_bytes = len(tokens) * 100
    
    # 根据token类型调整
    for token in tokens:
        if "GOAL:" in token:
            base_bytes += 200  # 目标token需要更多上下文
        elif "FRONTIER" in token:
            base_bytes += 150  # 前沿信息
        elif "HISTORY" in token:
            base_bytes += 100  # 历史信息
    
    return base_bytes


def minhash_signature(tokens):
    """生成tokens的MinHash签名"""
    return group_signature(tokens)


def split_large_cluster(tokens, max_kv_bytes):
    """分割过大的簇"""
    if not tokens:
        return []
    
    # 简单策略：按token类型分组
    goal_tokens = [t for t in tokens if "GOAL:" in t]
    frontier_tokens = [t for t in tokens if "FRONTIER" in t]
    history_tokens = [t for t in tokens if "HISTORY" in t]
    other_tokens = [t for t in tokens if not any(x in t for x in ["GOAL:", "FRONTIER", "HISTORY"])]
    
    sub_clusters = []
    
    # 添加目标相关簇
    if goal_tokens:
        sub_clusters.append(goal_tokens)
    
    # 添加前沿相关簇
    if frontier_tokens:
        sub_clusters.append(frontier_tokens)
    
    # 添加历史相关簇
    if history_tokens:
        sub_clusters.append(history_tokens)
    
    # 添加其他token簇
    if other_tokens:
        sub_clusters.append(other_tokens)
    
    return sub_clusters


def build_or_update_groups(agent_state, new_tokens, target_text, cfg):
    """构建或更新分组"""
    aff = early_qk_affinity(agent_state.llm, new_tokens, target_text, cfg.early_layers)
    clusters = spatial_semantic_cluster(new_tokens, aff, cfg.max_group_kv_bytes)
    groups = []
    
    for cl in clusters:
        # Simple group ID assignment
        if 'group_index' not in agent_state:
            agent_state['group_index'] = {'counter': 0}
        gid = agent_state['group_index']['counter']
        agent_state['group_index']['counter'] += 1
        g = Group(gid, cl)
        g.kv_bytes = estimate_group_kv_bytes(cl)
        g.sig = minhash_signature(cl)
        groups.append(g)
    
    return groups
