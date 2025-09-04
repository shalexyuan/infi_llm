# 组级 KV 缓存 ensure_resident + 组尾增量

import time
import threading
from typing import List, Dict, Set, Optional
from collections import OrderedDict
import numpy as np


class AIDEKVStore:
    def __init__(self, device_bytes: int):
        self.device_resident = set()  # 设备内存中的组ID集合
        self.slow_index = {}  # gid -> storage ptr (慢存储索引)
        self.meta = {}  # 组元数据
        self.device_used = 0  # 设备内存已使用字节数
        self.device_budget = device_bytes  # 设备内存总预算
        
        # 缓存管理相关
        self.access_times = {}  # gid -> last_access_time
        self.access_counts = {}  # gid -> access_count
        self.group_sizes = {}  # gid -> kv_bytes
        
        # 线程安全
        self._lock = threading.RLock()

    def ensure_resident(self, groups: list, kv_budget_bytes: int):
        """确保指定组在设备内存中，必要时进行换出"""
        with self._lock:
            # 计算需要加载的组
            need_groups = [g for g in groups if g.gid not in self.device_resident]
            need_bytes = sum(g.kv_bytes for g in need_groups)
            
            # 检查是否需要换出
            free_bytes = self.device_budget - self.device_used
            if need_bytes > free_bytes:
                # 选择换出受害者
                exclude_gids = {g.gid for g in groups}
                victims = self._choose_evictions(exclude=exclude_gids)
                
                # 执行换出
                for victim_gid in victims:
                    self._evict_to_slow(victim_gid)
                    free_bytes = self.device_budget - self.device_used
                    if need_bytes <= free_bytes:
                        break
            
            # 批量加载需要的组
            load_gids = [g.gid for g in need_groups]
            self._prefetch_from_slow(load_gids)
            
            # 处理增量更新
            for g in groups:
                if g.has_new:
                    self._append_delta_on_device(g.gid, g.tokens)
            
            # 更新设备内存中的组集合
            self.device_resident.update(load_gids)
            
            # 计算命中率
            hit_rate = 1 - len(load_gids) / max(len(groups), 1)
            
            return {
                "hit_rate": hit_rate,
                "loaded_groups": len(load_gids),
                "device_used": self.device_used,
                "device_budget": self.device_budget
            }

    def _choose_evictions(self, exclude: Set[int]) -> List[int]:
        """选择要换出的组（LRU + 大小考虑）"""
        candidates = []
        for gid in self.device_resident:
            if gid not in exclude:
                # 计算综合评分（访问时间 + 访问频率 + 大小）
                last_access = self.access_times.get(gid, 0)
                access_count = self.access_counts.get(gid, 0)
                size = self.group_sizes.get(gid, 0)
                
                # 评分公式：时间权重 + 频率权重 - 大小惩罚
                score = (time.time() - last_access) * 0.6 + access_count * 0.3 - size * 0.1
                candidates.append((gid, score))
        
        # 按评分排序，选择评分最低的（最不重要的）
        candidates.sort(key=lambda x: x[1])
        return [gid for gid, _ in candidates]

    def _evict_to_slow(self, gid: int):
        """将组从设备内存换出到慢存储"""
        if gid not in self.device_resident:
            return
        
        # 获取组大小
        group_size = self.group_sizes.get(gid, 0)
        
        # 更新慢存储索引
        self.slow_index[gid] = {
            'ptr': f"slow_storage_{gid}_{int(time.time())}",
            'size': group_size,
            'evict_time': time.time()
        }
        
        # 从设备内存移除
        self.device_resident.remove(gid)
        self.device_used -= group_size
        
        # 清理元数据
        if gid in self.access_times:
            del self.access_times[gid]
        if gid in self.access_counts:
            del self.access_counts[gid]
        if gid in self.group_sizes:
            del self.group_sizes[gid]

    def _prefetch_from_slow(self, gids: List[int]):
        """从慢存储批量预取组到设备内存"""
        for gid in gids:
            if gid in self.slow_index:
                # 从慢存储加载
                storage_info = self.slow_index[gid]
                group_size = storage_info['size']
                
                # 检查设备内存空间
                if self.device_used + group_size <= self.device_budget:
                    # 加载到设备内存
                    self.device_resident.add(gid)
                    self.device_used += group_size
                    self.group_sizes[gid] = group_size
                    
                    # 更新访问信息
                    self.access_times[gid] = time.time()
                    self.access_counts[gid] = self.access_counts.get(gid, 0) + 1
                    
                    # 从慢存储索引移除
                    del self.slow_index[gid]
                else:
                    # 设备内存不足，跳过加载
                    print(f"Warning: Insufficient device memory for group {gid}")
            else:
                # 新组，分配默认大小
                default_size = 1000  # 默认1KB
                if self.device_used + default_size <= self.device_budget:
                    self.device_resident.add(gid)
                    self.device_used += default_size
                    self.group_sizes[gid] = default_size
                    self.access_times[gid] = time.time()
                    self.access_counts[gid] = 1

    def _append_delta_on_device(self, gid: int, new_tokens: List[str]):
        """在设备内存中追加组的增量更新"""
        if gid not in self.device_resident:
            return
        
        # 计算增量大小
        delta_size = len(new_tokens) * 100  # 估算：每个token约100字节
        
        # 检查是否需要额外空间
        if self.device_used + delta_size > self.device_budget:
            # 需要换出其他组
            victims = self._choose_evictions(exclude={gid})
            for victim_gid in victims:
                self._evict_to_slow(victim_gid)
                if self.device_used + delta_size <= self.device_budget:
                    break
        
        # 追加增量
        self.group_sizes[gid] = self.group_sizes.get(gid, 0) + delta_size
        self.device_used += delta_size
        
        # 更新访问信息
        self.access_times[gid] = time.time()
        self.access_counts[gid] = self.access_counts.get(gid, 0) + 1

    def get_group_info(self, gid: int) -> Optional[Dict]:
        """获取组信息"""
        with self._lock:
            if gid in self.device_resident:
                return {
                    'location': 'device',
                    'size': self.group_sizes.get(gid, 0),
                    'last_access': self.access_times.get(gid, 0),
                    'access_count': self.access_counts.get(gid, 0)
                }
            elif gid in self.slow_index:
                return {
                    'location': 'slow_storage',
                    'size': self.slow_index[gid]['size'],
                    'evict_time': self.slow_index[gid]['evict_time']
                }
            else:
                return None

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        with self._lock:
            return {
                'device_resident_count': len(self.device_resident),
                'device_used_bytes': self.device_used,
                'device_budget_bytes': self.device_budget,
                'slow_storage_count': len(self.slow_index),
                'total_groups': len(self.device_resident) + len(self.slow_index),
                'utilization_rate': self.device_used / self.device_budget if self.device_budget > 0 else 0
            }

    def clear_cache(self):
        """清空缓存"""
        with self._lock:
            self.device_resident.clear()
            self.slow_index.clear()
            self.meta.clear()
            self.device_used = 0
            self.access_times.clear()
            self.access_counts.clear()
            self.group_sizes.clear()

    def prefetch_groups(self, gids: List[int]):
        """预取指定组到设备内存"""
        with self._lock:
            for gid in gids:
                if gid not in self.device_resident and gid in self.slow_index:
                    self._prefetch_from_slow([gid])

    def evict_group(self, gid: int):
        """主动换出指定组"""
        with self._lock:
            if gid in self.device_resident:
                self._evict_to_slow(gid)
