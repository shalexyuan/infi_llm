# ------------ selector.py ------------
from typing import List, Dict, Tuple
import math
import pdb

def _effective_cost(g: Dict, agent_id: str) -> float:
    """已驻留则增量成本≈0；否则用 g['C']。"""
    if "resident_agents" in g and agent_id in g["resident_agents"]:
        return 0.0
    return float(g["C"])

def _greedy_pack(groups: List[Dict], agent_id: str, lam_eff: float,
                 B_mem: float, eps_H: float, alpha_base: float,
                 w_dist: float, pose_xy: Tuple[float,float]) -> Tuple[List[int], float, float, float, float]:
    """贪心近似解 argmax Σ[U - lam_eff*C - w_dist*D], s.t. ΣC≤B_mem, ΣH≤eps_H"""
    rx, ry = pose_xy
    raw_entries = []
    max_dist = 0.0
    max_cost = 0.0
    for g in groups:
        U = float(g["V"]) - alpha_base * float(g["H"])
        C = _effective_cost(g, agent_id)
        D = math.hypot(float(g.get("cx", 0)) - rx, float(g.get("cy", 0)) - ry)
        raw_entries.append((g["g_id"], U, g["H"], C, D))
        if D > max_dist:
            max_dist = D
        if C > max_cost:
            max_cost = C

    scored = []
    dist_norm = max_dist + 1e-6
    cost_norm = max(max_cost, 1e-6)
    for g_id, U, H, C, D in raw_entries:
        D_norm = min(D / dist_norm, 1.0 - 1e-9)
        C_norm = C / cost_norm
        score = U - lam_eff * C_norm - w_dist * D_norm
        print(score,U,lam_eff,C_norm,D_norm)
        density = score / max(C_norm, 1e-6)  # 单位字节净效用
        scored.append((g_id, U, H, C_norm, D_norm, score, density))
    # lazy-greedy：先按 density 排，再按 score 校正
    scored.sort(key=lambda x: (x[6], x[5]), reverse=True)

    B_budget = B_mem / cost_norm if cost_norm > 0 else B_mem
    
    S, U_sum, C_sum, H_sum, D_sum = [], 0.0, 0.0, 0.0, 0.0
    print(f"C_sum: {C_sum}, H_sum: {H_sum}, D_sum: {D_sum}, B_budget: {B_budget}, B_mem: {B_mem}, cost_norm: {cost_norm}, eps_H: {eps_H}")
    print(scored)
    # pdb.set_trace()
    for g_id, U, H, C, D, score, density in scored:
        if score < 0:  # 边际<=0 直接停止（Dinkelbach条件）
            continue
        if C_sum + C > B_budget:         # 显存预算
            continue
        if H_sum + H > eps_H:         # 干扰预算（需要就设；不需要可设很大）
            continue
        S.append(g_id); U_sum += U; C_sum += C; H_sum += H; D_sum += D

    if not S and scored:
        best = max(scored, key=lambda x: x[6])
        if best[6] > 0:
            g_id, U, H, C, D, score, density = best
            S = [g_id]
            U_sum, C_sum, H_sum, D_sum = U, C, H, D

    return S, U_sum, C_sum, H_sum, D_sum

def select_mipb_for_agent(agent_id: str,
                          groups: List[Dict],
                          pose_xy: Tuple[float,float],
                          B_mem: float, eps_H: float,
                          alpha_base: float = 1.0,
                          price_p: float = 0.0,  # 来自价格协调器
                          w_dist: float = 0.0,
                          lam0: float = 0.0,
                          max_outer: int = 8,
                          tol: float = 1e-6) -> Dict:
    """
    返回：{
      "agent": agent_id,
      "selected": [g_id...],
      "U_sum": ..., "C_sum": ..., "H_sum": ...,
      "lambda_star": ...,
      "ranklist": [(g_id, score_at_lambda), ...]  # 用于“意向-竞价”与二轮回退
    }
    """
    lam = lam0
    last = None
    ranklist = None

    for _ in range(max_outer):
        lam_eff = lam + price_p
        S, U_sum, C_sum, H_sum, D_sum = _greedy_pack(
            groups, agent_id, lam_eff, B_mem, eps_H, alpha_base, w_dist, pose_xy
        )

        if C_sum <= 1e-9:
            # 选不到任何组

            return {"agent": agent_id, "selected": [], "U_sum":0.0,"C_sum":0.0,"H_sum":0.0,
                    "lambda_star": lam, "ranklist": []}
        numer = U_sum - w_dist * D_sum
        phi = numer - lam_eff * C_sum

        if abs(phi) < tol:
            # 收敛
            # 生成 ranklist（用于后续意向-竞价)
            ranklist = _score_ranklist(groups, agent_id, lam_eff, alpha_base, w_dist, pose_xy)
            return {"agent": agent_id, "selected": S, "U_sum":U_sum, "C_sum":C_sum, "H_sum":H_sum,
                    "lambda_star": lam, "ranklist": ranklist}
        lam = numer / C_sum - price_p
        last = (S, U_sum, C_sum, H_sum, D_sum, lam)

    # 达到迭代上限，返回最后一次
    if last is None:
        return {"agent": agent_id, "selected": [], "U_sum":0.0,"C_sum":0.0,"H_sum":0.0,
                "lambda_star": lam, "ranklist": []}
    S, U_sum, C_sum, H_sum, D_sum, lam = last
    lam_eff = lam + price_p
    ranklist = _score_ranklist(groups, agent_id, lam_eff, alpha_base, w_dist, pose_xy)
    return {"agent": agent_id, "selected": S, "U_sum":U_sum, "C_sum":C_sum, "H_sum":H_sum,
            "lambda_star": lam, "ranklist": ranklist}

def _score_ranklist(groups, agent_id, lam_eff, alpha_base, w_dist, pose_xy):
    """给“意向-竞价”用的每组净分：bid = U - lam_eff*C - w_dist*D"""
    rx, ry = pose_xy
    raw_entries = []
    max_dist = 0.0
    max_cost = 0.0
    for g in groups:
        U = float(g["V"]) - alpha_base * float(g["H"])
        C = _effective_cost(g, agent_id)
        D = math.hypot(float(g.get("cx", 0)) - rx, float(g.get("cy", 0)) - ry)
        raw_entries.append((g["g_id"], U, C, D))
        if D > max_dist:
            max_dist = D
        if C > max_cost:
            max_cost = C

    dist_norm = max_dist + 1e-6
    cost_norm = max(max_cost, 1e-6)
    out = []
    for g_id, U, C, D in raw_entries:
        D_norm = min(D / dist_norm, 1.0 - 1e-9)
        C_norm = C / cost_norm
        bid = U - lam_eff * C_norm - w_dist * D_norm
        out.append((g_id, bid))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
