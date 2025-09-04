# 组摘要消息（哈希/签名/去重合并）

import hashlib
import time
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class GroupSummary:
    gid: int
    sig: bytes
    kv_bytes: int
    v_hat: float
    h_hat: float


def compose_summaries(groups) -> List[GroupSummary]:
    """将组列表转换为摘要列表"""
    return [GroupSummary(g.gid, g.sig, g.kv_bytes, g.v_hat, g.h_hat) for g in groups]


def dedup_and_merge(agent_state, incoming_summaries):
    """按 sig 去重；若外部组更优(高V/低H)则更新本地目录/全局图，但不强制加载KV"""
    if not incoming_summaries:
        return []
    
    # 按签名分组
    sig_groups = defaultdict(list)
    for summary in incoming_summaries:
        sig_groups[summary.sig].append(summary)
    
    merged_summaries = []
    updated_groups = []
    
    for sig, summaries in sig_groups.items():
        if len(summaries) == 1:
            # 单个摘要，直接添加
            merged_summaries.append(summaries[0])
            continue
        
        # 多个相同签名的摘要，选择最优的
        best_summary = select_best_summary(summaries)
        merged_summaries.append(best_summary)
        
        # 检查是否需要更新本地组
        if should_update_local_group(agent_state, best_summary):
            updated_groups.append(best_summary)
    
    # 更新本地目录和全局图
    update_local_directory(agent_state, updated_groups)
    update_global_graph(agent_state, updated_groups)
    
    return merged_summaries


def select_best_summary(summaries: List[GroupSummary]) -> GroupSummary:
    """从多个相同签名的摘要中选择最优的"""
    if not summaries:
        return None
    
    if len(summaries) == 1:
        return summaries[0]
    
    # 计算综合评分：V - αH（α为权重参数）
    alpha = 1.0  # 可配置的权重参数
    best_score = float('-inf')
    best_summary = summaries[0]
    
    for summary in summaries:
        score = summary.v_hat - alpha * summary.h_hat
        
        # 如果评分相同，选择KV字节数较少的（更高效）
        if score > best_score or (score == best_score and summary.kv_bytes < best_summary.kv_bytes):
            best_score = score
            best_summary = summary
    
    return best_summary


def should_update_local_group(agent_state, summary: GroupSummary) -> bool:
    """判断是否应该更新本地组"""
    # 检查本地是否存在相同签名的组
    local_group = find_local_group_by_signature(agent_state, summary.sig)
    
    if local_group is None:
        # 本地没有该组，应该添加
        return True
    
    # 比较质量：外部组是否更优
    alpha = 1.0
    external_score = summary.v_hat - alpha * summary.h_hat
    local_score = local_group.v_hat - alpha * local_group.h_hat
    
    return external_score > local_score


def find_local_group_by_signature(agent_state, sig: bytes):
    """根据签名查找本地组"""
    # 这里需要根据实际的agent_state结构来实现
    # 假设agent_state有一个groups属性
    if hasattr(agent_state, 'groups'):
        for group in agent_state.groups:
            if hasattr(group, 'sig') and group.sig == sig:
                return group
    
    return None


def update_local_directory(agent_state, updated_summaries: List[GroupSummary]):
    """更新本地组目录"""
    if not updated_summaries:
        return
    
    # 这里需要根据实际的agent_state结构来实现
    # 假设agent_state有一个group_directory属性
    if hasattr(agent_state, 'group_directory'):
        for summary in updated_summaries:
            agent_state.group_directory[summary.gid] = {
                'sig': summary.sig,
                'kv_bytes': summary.kv_bytes,
                'v_hat': summary.v_hat,
                'h_hat': summary.h_hat,
                'update_time': time.time()
            }


def update_global_graph(agent_state, updated_summaries: List[GroupSummary]):
    """更新全局图"""
    if not updated_summaries:
        return
    
    # 这里需要根据实际的agent_state结构来实现
    # 假设agent_state有一个global_graph属性
    if hasattr(agent_state, 'global_graph'):
        for summary in updated_summaries:
            # 添加或更新全局图中的节点
            agent_state.global_graph.add_or_update_node(
                summary.gid,
                {
                    'sig': summary.sig,
                    'kv_bytes': summary.kv_bytes,
                    'v_hat': summary.v_hat,
                    'h_hat': summary.h_hat,
                    'last_update': time.time()
                }
            )


def compute_group_hash(summary: GroupSummary) -> str:
    """计算组的哈希值"""
    hash_input = f"{summary.gid}:{summary.sig.hex()}:{summary.kv_bytes}:{summary.v_hat}:{summary.h_hat}"
    return hashlib.sha256(hash_input.encode()).hexdigest()


def verify_group_signature(summary: GroupSummary, expected_sig: bytes) -> bool:
    """验证组签名"""
    return summary.sig == expected_sig


def compress_summaries(summaries: List[GroupSummary]) -> bytes:
    """压缩摘要列表为字节流"""
    if not summaries:
        return b''
    
    # 简单的序列化格式
    compressed = b''
    for summary in summaries:
        # 格式：gid(4) + sig_len(1) + sig + kv_bytes(4) + v_hat(4) + h_hat(4)
        gid_bytes = summary.gid.to_bytes(4, 'big')
        sig_len = len(summary.sig).to_bytes(1, 'big')
        kv_bytes = summary.kv_bytes.to_bytes(4, 'big')
        v_hat_bytes = int(summary.v_hat * 1000).to_bytes(4, 'big')  # 放大1000倍
        h_hat_bytes = int(summary.h_hat * 1000).to_bytes(4, 'big')  # 放大1000倍
        
        compressed += gid_bytes + sig_len + summary.sig + kv_bytes + v_hat_bytes + h_hat_bytes
    
    return compressed


def decompress_summaries(compressed_data: bytes) -> List[GroupSummary]:
    """从字节流解压缩摘要列表"""
    if not compressed_data:
        return []
    
    summaries = []
    offset = 0
    
    while offset < len(compressed_data):
        # 解析gid
        gid = int.from_bytes(compressed_data[offset:offset+4], 'big')
        offset += 4
        
        # 解析sig
        sig_len = int.from_bytes(compressed_data[offset:offset+1], 'big')
        offset += 1
        sig = compressed_data[offset:offset+sig_len]
        offset += sig_len
        
        # 解析kv_bytes
        kv_bytes = int.from_bytes(compressed_data[offset:offset+4], 'big')
        offset += 4
        
        # 解析v_hat
        v_hat_int = int.from_bytes(compressed_data[offset:offset+4], 'big')
        v_hat = v_hat_int / 1000.0  # 还原
        offset += 4
        
        # 解析h_hat
        h_hat_int = int.from_bytes(compressed_data[offset:offset+4], 'big')
        h_hat = h_hat_int / 1000.0  # 还原
        offset += 4
        
        summaries.append(GroupSummary(gid, sig, kv_bytes, v_hat, h_hat))
    
    return summaries


def compute_summary_statistics(summaries: List[GroupSummary]) -> Dict:
    """计算摘要统计信息"""
    if not summaries:
        return {
            'count': 0,
            'total_kv_bytes': 0,
            'avg_v_hat': 0.0,
            'avg_h_hat': 0.0,
            'unique_signatures': 0
        }
    
    total_kv_bytes = sum(s.kv_bytes for s in summaries)
    avg_v_hat = np.mean([s.v_hat for s in summaries])
    avg_h_hat = np.mean([s.h_hat for s in summaries])
    unique_signatures = len(set(s.sig for s in summaries))
    
    return {
        'count': len(summaries),
        'total_kv_bytes': total_kv_bytes,
        'avg_v_hat': avg_v_hat,
        'avg_h_hat': avg_h_hat,
        'unique_signatures': unique_signatures,
        'v_hat_range': (min(s.v_hat for s in summaries), max(s.v_hat for s in summaries)),
        'h_hat_range': (min(s.h_hat for s in summaries), max(s.h_hat for s in summaries))
    }


def filter_summaries_by_quality(summaries: List[GroupSummary], 
                               min_v_hat: float = 0.0, 
                               max_h_hat: float = float('inf')) -> List[GroupSummary]:
    """根据质量过滤摘要"""
    return [
        summary for summary in summaries
        if summary.v_hat >= min_v_hat and summary.h_hat <= max_h_hat
    ]


def merge_summaries_with_priority(local_summaries: List[GroupSummary], 
                                 remote_summaries: List[GroupSummary],
                                 local_priority: float = 0.5) -> List[GroupSummary]:
    """按优先级合并本地和远程摘要"""
    if not local_summaries and not remote_summaries:
        return []
    
    if not local_summaries:
        return remote_summaries
    
    if not remote_summaries:
        return local_summaries
    
    # 按签名分组
    all_summaries = local_summaries + remote_summaries
    sig_groups = defaultdict(list)
    
    for summary in all_summaries:
        sig_groups[summary.sig].append(summary)
    
    merged = []
    for sig, summaries in sig_groups.items():
        if len(summaries) == 1:
            merged.append(summaries[0])
        else:
            # 有多个相同签名的摘要，按优先级选择
            best_summary = select_summary_with_priority(summaries, local_priority)
            merged.append(best_summary)
    
    return merged


def select_summary_with_priority(summaries: List[GroupSummary], local_priority: float) -> GroupSummary:
    """按优先级选择摘要"""
    if not summaries:
        return None
    
    if len(summaries) == 1:
        return summaries[0]
    
    # 计算加权评分
    best_score = float('-inf')
    best_summary = summaries[0]
    
    for summary in summaries:
        # 基础评分：V - H
        base_score = summary.v_hat - summary.h_hat
        
        # 应用优先级权重（这里简化处理，实际可能需要更复杂的逻辑）
        weighted_score = base_score * local_priority
        
        if weighted_score > best_score:
            best_score = weighted_score
            best_summary = summary
    
    return best_summary
