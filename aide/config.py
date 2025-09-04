# AIDE 参数与预算

from dataclasses import dataclass


@dataclass
class AIDEConfig:
    early_layers: int = 3
    max_group_kv_bytes: int = 1_000_000
    alpha_base: float = 1.0
    lambda_base: float = 1e-9
    theta_sem: float = 0.25
    epsilon_interf: float = 0.5
    mem_bytes_robot: int = 8_000_000_000
    comm_bytes_global: int = 2_000_000
    lr_p: float = 1e-3
    lr_tau: float = 1e-3
    topk_candidates: int = 6
