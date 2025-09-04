# 分布式价格 p/τ 的更新（集中/去中心）

import time
import threading
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from aide.config import AIDEConfig


class PriceServer:
    def __init__(self, cfg: AIDEConfig):
        self.p = 0.0  # 价格 p (cost per byte)
        self.tau = 0.0  # 价格 τ (interference penalty)
        self.cfg = cfg
        
        # 分布式更新相关
        self.update_history = []  # 更新历史
        self.last_update_time = time.time()
        self.update_count = 0
        
        # 线程安全
        self._lock = threading.RLock()

    def update(self, total_cost_bytes: int, total_interf: float) -> Tuple[float, float]:
        """更新价格 p 和 τ"""
        with self._lock:
            # 计算价格更新
            p_update = self.cfg.lr_p * (total_cost_bytes - self.cfg.comm_bytes_global)
            tau_update = self.cfg.lr_tau * (total_interf - self.cfg.epsilon_interf)
            
            # 应用更新（确保非负）
            self.p = max(0.0, self.p + p_update)
            self.tau = max(0.0, self.tau + tau_update)
            
            # 记录更新历史
            self.update_history.append({
                'timestamp': time.time(),
                'p': self.p,
                'tau': self.tau,
                'cost_bytes': total_cost_bytes,
                'interference': total_interf,
                'p_update': p_update,
                'tau_update': tau_update
            })
            
            # 限制历史记录长度
            if len(self.update_history) > 1000:
                self.update_history = self.update_history[-500:]
            
            self.last_update_time = time.time()
            self.update_count += 1
            
            return self.p, self.tau

    def get_prices(self) -> Tuple[float, float]:
        """获取当前价格"""
        with self._lock:
            return self.p, self.tau

    def reset_prices(self):
        """重置价格"""
        with self._lock:
            self.p = 0.0
            self.tau = 0.0
            self.update_history.clear()
            self.update_count = 0

    def get_price_stats(self) -> Dict:
        """获取价格统计信息"""
        with self._lock:
            if not self.update_history:
                return {
                    'current_p': self.p,
                    'current_tau': self.tau,
                    'update_count': self.update_count,
                    'last_update': self.last_update_time
                }
            
            # 计算统计信息
            p_values = [entry['p'] for entry in self.update_history]
            tau_values = [entry['tau'] for entry in self.update_history]
            
            return {
                'current_p': self.p,
                'current_tau': self.tau,
                'p_mean': np.mean(p_values),
                'p_std': np.std(p_values),
                'p_min': np.min(p_values),
                'p_max': np.max(p_values),
                'tau_mean': np.mean(tau_values),
                'tau_std': np.std(tau_values),
                'tau_min': np.min(tau_values),
                'tau_max': np.max(tau_values),
                'update_count': self.update_count,
                'last_update': self.last_update_time,
                'history_length': len(self.update_history)
            }


class DecentralizedPriceServer(PriceServer):
    """去中心化价格服务器"""
    
    def __init__(self, cfg: AIDEConfig, node_id: str):
        super().__init__(cfg)
        self.node_id = node_id
        self.peer_prices = {}  # 其他节点的价格
        self.peer_weights = {}  # 其他节点的权重
        self.gossip_interval = 1.0  # gossip间隔（秒）
        self.last_gossip_time = time.time()

    def update_with_gossip(self, total_cost_bytes: int, total_interf: float, 
                          peer_updates: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
        """带gossip的价格更新"""
        with self._lock:
            # 基础价格更新
            p, tau = self.update(total_cost_bytes, total_interf)
            
            # 处理来自其他节点的更新
            for peer_id, (peer_p, peer_tau) in peer_updates.items():
                self.peer_prices[peer_id] = (peer_p, peer_tau)
                self.peer_weights[peer_id] = self.peer_weights.get(peer_id, 0.1)
            
            # 计算加权平均价格
            if self.peer_prices:
                weighted_p = self._compute_weighted_average('p')
                weighted_tau = self._compute_weighted_average('tau')
                
                # 应用gossip更新
                gossip_weight = 0.1  # gossip权重
                self.p = (1 - gossip_weight) * self.p + gossip_weight * weighted_p
                self.tau = (1 - gossip_weight) * self.tau + gossip_weight * weighted_tau
            
            return self.p, self.tau

    def _compute_weighted_average(self, price_type: str) -> float:
        """计算加权平均价格"""
        if not self.peer_prices:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for peer_id, (peer_p, peer_tau) in self.peer_prices.items():
            weight = self.peer_weights.get(peer_id, 0.1)
            price = peer_p if price_type == 'p' else peer_tau
            
            weighted_sum += weight * price
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_gossip_message(self) -> Dict:
        """生成gossip消息"""
        with self._lock:
            return {
                'node_id': self.node_id,
                'p': self.p,
                'tau': self.tau,
                'timestamp': time.time(),
                'update_count': self.update_count
            }

    def update_peer_weight(self, peer_id: str, weight: float):
        """更新对等节点权重"""
        with self._lock:
            self.peer_weights[peer_id] = max(0.0, min(1.0, weight))


class CentralizedPriceServer(PriceServer):
    """集中式价格服务器"""
    
    def __init__(self, cfg: AIDEConfig):
        super().__init__(cfg)
        self.client_updates = defaultdict(list)  # 客户端更新记录
        self.client_weights = {}  # 客户端权重
        self.aggregation_interval = 5.0  # 聚合间隔（秒）
        self.last_aggregation_time = time.time()

    def receive_client_update(self, client_id: str, total_cost_bytes: int, 
                            total_interf: float) -> Tuple[float, float]:
        """接收客户端更新"""
        with self._lock:
            # 记录客户端更新
            self.client_updates[client_id].append({
                'timestamp': time.time(),
                'cost_bytes': total_cost_bytes,
                'interference': total_interf
            })
            
            # 检查是否需要聚合
            if time.time() - self.last_aggregation_time > self.aggregation_interval:
                self._aggregate_client_updates()
            
            return self.p, self.tau

    def _aggregate_client_updates(self):
        """聚合客户端更新"""
        if not self.client_updates:
            return
        
        # 计算总成本和干扰
        total_cost = 0
        total_interference = 0.0
        total_weight = 0.0
        
        for client_id, updates in self.client_updates.items():
            if not updates:
                continue
            
            # 使用最新更新
            latest_update = updates[-1]
            weight = self.client_weights.get(client_id, 1.0)
            
            total_cost += latest_update['cost_bytes'] * weight
            total_interference += latest_update['interference'] * weight
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            avg_cost = total_cost / total_weight
            avg_interference = total_interference / total_weight
            
            # 更新价格
            self.update(avg_cost, avg_interference)
        
        # 清理旧更新
        cutoff_time = time.time() - self.aggregation_interval * 2
        for client_id in list(self.client_updates.keys()):
            self.client_updates[client_id] = [
                update for update in self.client_updates[client_id]
                if update['timestamp'] > cutoff_time
            ]
        
        self.last_aggregation_time = time.time()

    def set_client_weight(self, client_id: str, weight: float):
        """设置客户端权重"""
        with self._lock:
            self.client_weights[client_id] = max(0.0, weight)

    def get_client_stats(self) -> Dict:
        """获取客户端统计信息"""
        with self._lock:
            stats = {
                'client_count': len(self.client_updates),
                'total_updates': sum(len(updates) for updates in self.client_updates.values()),
                'last_aggregation': self.last_aggregation_time
            }
            
            for client_id, updates in self.client_updates.items():
                if updates:
                    stats[f'client_{client_id}_updates'] = len(updates)
                    stats[f'client_{client_id}_weight'] = self.client_weights.get(client_id, 1.0)
            
            return stats
