# -*- coding: utf-8 -*-
# utils/mem_profiler.py
import time, threading, queue, json, os
from dataclasses import dataclass
from typing import Optional, List
import psutil

try:
    import pynvml as N
    N.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False

@dataclass
class MemSample:
    t: float
    vram_used_mb: int
    proc_rss_mb: int

@dataclass
class MemReport:
    start_t: float
    end_t: float
    idle_vram_mb: int
    peak_vram_mb: int
    delta_vram_mb: int
    aumc_mb_s: float
    samples: List[MemSample]

class GpuMemProbe:
    """
    黑盒 GPU 显存采样器：不改 VLM 代码。通过 NVML 轮询 GPU memory.used，
    同时采集服务进程 RSS（如果提供 pid）。
    """
    def __init__(self, gpu_index: int = 0, service_pid: Optional[int] = None, interval_ms: int = 20):
        self.gpu_index = gpu_index
        self.service_pid = service_pid
        self.interval = max(5, interval_ms) / 1000.0
        self._running = False
        self._q = queue.Queue()
        self._thread = None
        self._device = None
        if _NVML_OK:
            self._device = N.nvmlDeviceGetHandleByIndex(gpu_index)

    def _poll(self):
        p = psutil.Process(self.service_pid) if self.service_pid else None
        while self._running:
            t = time.time()
            if _NVML_OK:
                mi = N.nvmlDeviceGetMemoryInfo(self._device)
                vram_mb = int(mi.used / (1024**2))
            else:
                vram_mb = -1
            rss_mb = int(p.memory_info().rss / (1024**2)) if p else -1
            self._q.put(MemSample(t, vram_mb, rss_mb))
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return time.time()

    def stop_and_report(self) -> MemReport:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        samples = []
        while not self._q.empty():
            samples.append(self._q.get())
        if not samples:
            now = time.time()
            return MemReport(now, now, 0, 0, 0, 0.0, [])
        idle = min(s.vram_used_mb for s in samples)      # 观测到的最小值视为“空闲”
        peak = max(s.vram_used_mb for s in samples)
        delta = max(0, peak - idle)
        # AUMC: 梯形法积分（(x_i + x_{i+1})/2 * Δt）
        aumc = 0.0
        for i in range(len(samples)-1):
            y1 = samples[i].vram_used_mb - idle
            y2 = samples[i+1].vram_used_mb - idle
            dt = samples[i+1].t - samples[i].t
            if dt > 0:
                aumc += (y1 + y2) * 0.5 * dt
        return MemReport(samples[0].t, samples[-1].t, idle, peak, delta, aumc, samples)

def size_bytes_of_request(json_payload: dict, binary_blobs: Optional[List[bytes]]=None) -> int:
    text_bytes = len(json.dumps(json_payload, ensure_ascii=False).encode("utf-8"))
    blob_bytes = sum(len(b) for b in (binary_blobs or []))
    return text_bytes + blob_bytes
