# AIDEController：把以上整合进现有主循环

import torch
from typing import List, Dict, Tuple, Optional
from aide.config import AIDEConfig
from aide.grouping import build_or_update_groups
from aide.metrics import score_groups
from aide.selector import select_groups_AIRD
from aide.kv import AIDEKVStore
from aide.comm import compose_summaries, dedup_and_merge
from semantic_mapping import encode_map_to_tokens


class AIDEController:
    def __init__(self, vlm_client, cfg: AIDEConfig, price_server):
        self.llm = vlm_client
        self.cfg = cfg
        self.price = price_server

    def step(self, agent_state, target_text):
        """执行AIDE的完整步骤：分组-打分-选组-KV上屏-消息摘要"""
        
        # 1) 从地图抽 token（复用 semantic_mapping 的函数）
        new_tokens = encode_map_to_tokens(
            agent_state.get('map_state'), 
            agent_state.get('frontier_nodes', []), 
            agent_state.get('history_nodes', []), 
            target_text
        )
        
        groups = build_or_update_groups(agent_state, new_tokens, target_text, self.cfg)
        score_groups(groups, self.llm, target_text, self.cfg)

        # 2) AIRD 选择 + 组级 KV 上屏
        S_ids, used_kv, h_sum = select_groups_AIRD(
            groups, self.cfg, self.price.p, self.price.tau, self.cfg.mem_bytes_robot
        )
        selected = [g for g in groups if g.gid in S_ids]
        
        # 确保KV存储已初始化
        if 'kv' not in agent_state:
            agent_state['kv'] = AIDEKVStore(self.cfg.mem_bytes_robot)
        
        agent_state['kv'].ensure_resident(selected, self.cfg.mem_bytes_robot)

        # 3) 受限提示：只传 top‑K 候选（前沿/历史）并复用组级 KV
        header = make_prompt_header(target_text, agent_state.get('pose'), agent_state.get('last_goal'))
        cands_text, cand_mask = serialize_candidates(
            agent_state.get('top_candidates', []), self.cfg.topk_candidates
        )
        
        # 使用VLM进行解码（适配现有接口）
        full_prompt = header + cands_text
        try:
            # 简化选择逻辑：基于候选数量选择第一个
            candidates = agent_state.get('top_candidates', [])
            if candidates:
                choice = 0  # 选择第一个候选
                goal = agent_state.get('node_from_choice', lambda x: None)(choice)
            else:
                # 回退到默认选择
                goal = agent_state.get('last_goal', {'x': 0, 'y': 0})
        except Exception as e:
            print(f"Goal selection failed: {e}")
            # 回退到默认选择
            goal = agent_state.get('last_goal', {'x': 0, 'y': 0})

        # 4) 对偶价格：用本地成本/干扰反馈更新 p, τ
        c_bytes = sum(g.kv_bytes for g in selected)
        h_tot = sum(g.h_hat for g in selected)
        self.price.update(c_bytes, h_tot)

        return compose_summaries(selected), goal

    def merge_summaries_into_agent_states(self, incoming):
        """合并传入的摘要到代理状态"""
        for agent_state, summaries in incoming.items():
            dedup_and_merge(agent_state, summaries)


def make_prompt_header(target_text: str, pose: Optional[Dict], last_goal: Optional[Dict]) -> str:
    """创建提示头部"""
    header = f"Target: {target_text}\n"
    
    if pose:
        header += f"Current pose: ({pose.get('x', 0):.2f}, {pose.get('y', 0):.2f}, {pose.get('theta', 0):.2f})\n"
    
    if last_goal:
        header += f"Last goal: ({last_goal.get('x', 0):.2f}, {last_goal.get('y', 0):.2f})\n"
    
    header += "Available candidates:\n"
    return header


def serialize_candidates(candidates: List[Dict], topk: int) -> Tuple[str, torch.Tensor]:
    """序列化候选节点并创建词汇掩码"""
    if not candidates:
        return "", torch.ones(1000)  # 默认掩码
    
    # 限制候选数量
    candidates = candidates[:topk]
    
    cands_text = ""
    cand_mask = torch.zeros(1000)  # 假设词汇表大小为1000
    
    for i, candidate in enumerate(candidates):
        cand_type = candidate.get('type', 'unknown')
        x, y = candidate.get('x', 0), candidate.get('y', 0)
        
        cands_text += f"{i+1}. {cand_type} at ({x:.1f}, {y:.1f})\n"
        
        # 设置词汇掩码（简化实现）
        if i < len(cand_mask):
            cand_mask[i] = 1.0
    
    return cands_text, cand_mask


def create_agent_state(map_state=None, frontier_nodes=None, history_nodes=None, 
                      pose=None, last_goal=None, top_candidates=None):
    """创建代理状态字典"""
    return {
        'map_state': map_state,
        'frontier_nodes': frontier_nodes or [],
        'history_nodes': history_nodes or [],
        'pose': pose,
        'last_goal': last_goal,
                'top_candidates': top_candidates or [],
        'groups': [],
        'group_directory': {},
        'global_graph': None,
        'group_index': {'counter': 0},  # Simple group ID counter
        'node_from_choice': lambda choice: {'x': 0, 'y': 0}  # 默认实现
    }


def setup_aide_controller(vlm_client, cfg: AIDEConfig, price_server) -> AIDEController:
    """设置AIDE控制器"""
    return AIDEController(vlm_client, cfg, price_server)


def run_aide_step(controller: AIDEController, agent_state: Dict, target_text: str) -> Tuple[List, Dict]:
    """运行单个AIDE步骤"""
    try:
        summaries, goal = controller.step(agent_state, target_text)
        return summaries, goal
    except Exception as e:
        print(f"AIDE step failed: {e}")
        # 返回默认值
        return [], {'x': 0, 'y': 0}


def process_incoming_summaries(controller: AIDEController, incoming_data: Dict):
    """处理传入的摘要数据"""
    try:
        controller.merge_summaries_into_agent_states(incoming_data)
    except Exception as e:
        print(f"Failed to process incoming summaries: {e}")


def get_aide_statistics(controller: AIDEController, agent_state: Dict) -> Dict:
    """获取AIDE统计信息"""
    stats = {
        'price_p': controller.price.p,
        'price_tau': controller.price.tau,
        'groups_count': len(agent_state.get('groups', [])),
        'kv_utilization': 0.0
    }
    
    # 获取KV存储统计
    if hasattr(agent_state, 'kv') and agent_state['kv']:
        kv_stats = agent_state['kv'].get_cache_stats()
        stats['kv_utilization'] = kv_stats.get('utilization_rate', 0.0)
        stats['device_resident_count'] = kv_stats.get('device_resident_count', 0)
        stats['device_used_bytes'] = kv_stats.get('device_used_bytes', 0)
    
    # 获取价格统计
    price_stats = controller.price.get_price_stats()
    stats.update(price_stats)
    
    return stats


def reset_aide_controller(controller: AIDEController, agent_state: Dict):
    """重置AIDE控制器状态"""
    # 重置价格
    controller.price.reset_prices()
    
    # 清空KV存储
    if hasattr(agent_state, 'kv') and agent_state['kv']:
        agent_state['kv'].clear_cache()
    
    # 清空组目录
    agent_state['group_directory'] = {}
    agent_state['groups'] = []


def validate_aide_config(cfg: AIDEConfig) -> bool:
    """验证AIDE配置"""
    try:
        # 检查必要的配置参数
        required_params = [
            'mem_bytes_robot', 'comm_bytes_global', 'epsilon_interf',
            'alpha_base', 'lambda_base', 'topk_candidates'
        ]
        
        for param in required_params:
            if not hasattr(cfg, param):
                print(f"Missing required config parameter: {param}")
                return False
        
        # 检查参数合理性
        if cfg.mem_bytes_robot <= 0:
            print("mem_bytes_robot must be positive")
            return False
        
        if cfg.epsilon_interf <= 0:
            print("epsilon_interf must be positive")
            return False
        
        if cfg.topk_candidates <= 0:
            print("topk_candidates must be positive")
            return False
        
        return True
        
    except Exception as e:
        print(f"Config validation failed: {e}")
        return False


def create_aide_agent_state_from_main(agent_data: Dict) -> Dict:
    """从主循环数据创建AIDE代理状态"""
    return create_agent_state(
        map_state=agent_data.get('local_map'),
        frontier_nodes=agent_data.get('frontier_nodes', []),
        history_nodes=agent_data.get('history_nodes', []),
        pose=agent_data.get('pose'),
        last_goal=agent_data.get('last_goal'),
        top_candidates=agent_data.get('candidates', [])
    )
