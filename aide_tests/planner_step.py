# ------------ planner_step.py ------------
from typing import Dict, List, Tuple, Optional, Set

from .selector import select_mipb_for_agent, _greedy_pack  # 复用前面的
from .price_coord import PriceCoordinator
from .intents import build_intent, resolve_conflicts, apply_winners

def one_step_assign(groups: List[Dict],
                    agents: Dict[str, Dict],   # {"r1":{"pose":(x,y,th),"B_mem":..., "eps_H":...}, ...}
                    alpha_base: float,
                    price_coordinator: PriceCoordinator,
                    price_broadcast: Tuple[float,float] = None,
                    w_dist: float = 0.6,
                    intent_topk: int = 2,
                    agent_forbidden: Optional[Dict[str, Set[int]]] = None,
                    score_bar: Optional[float] = None):
    """
    返回:
      - selections: {agent_id: [g_id...]}  # 去重后的证据组
      - stats: {"total_C":..., "total_H":..., "p":..., "tau":...}
      - intents/winners: （可用于可视化/调试）
    """
    # A) 价格（中心广播；如果传入则用，否则从协调器当前状态拿）
    if price_broadcast is None:
        p, tau = price_coordinator.p, price_coordinator.tau
    else:
        p, tau = price_broadcast

    # B) 各 agent 并行做 Dinkelbach+greedy
    results = {}

    forbidden = agent_forbidden or {}
    fallback_agents: Set[str] = set()
    for aid, ainfo in agents.items():
        pose = ainfo["pose"]; B_mem = ainfo["B_mem"]; eps_H = ainfo.get("eps_H", float("inf"))
        blocked = forbidden.get(aid, set())
        candidate_groups = [g for g in groups if g["g_id"] not in blocked] if blocked else groups
        res = select_mipb_for_agent(aid, candidate_groups, (pose[0],pose[1]),
                                    B_mem, eps_H,
                                    alpha_base=alpha_base,
                                    price_p=p, w_dist=w_dist, lam0=0.0)
        results[aid] = res

    # Apply score-based fallback before broadcasting intents
    if score_bar is not None:
        for aid, res in results.items():
            ranklist = res.get("ranklist") or []
            # Check V value of the top-ranked group instead of ranklist score
            if ranklist:
                top_gid = ranklist[0][0]  # Get the group ID of the top-ranked group
                # Find the V value of this group
                top_group = next((g for g in groups if g["g_id"] == top_gid), None)
                top_v_score = top_group["V"] if top_group else None
                if top_v_score is not None and top_v_score < score_bar:
                    res["selected"] = []
                    res["ranklist"] = []
                    fallback_agents.add(aid)

    # C) 并行广播 top-K 意向（bid 已内含 lam* C 与 p）
    intents = [build_intent(aid, results[aid]["ranklist"], K=intent_topk) for aid in agents.keys()]
    winners = resolve_conflicts(intents, capacity={})  # 默认每组容量=1；你也可设某些大组=2

    # D) 本地应用赢家集：输家补齐
    def make_filler(aid):
        blocked = forbidden.get(aid, set())

        def _fill(S_partial, ranklist):
            # 在 ranklist 中自上而下尝试加入，直到预算满或无正收益
            pose = agents[aid]["pose"]; B_mem = agents[aid]["B_mem"]; eps_H = agents[aid].get("eps_H", float("inf"))
            lam_eff = results[aid]["lambda_star"] + p
            # 将 S_partial 固定后继续 greedy
            fixed = set(S_partial)
            # 禁止选择已被其他 agent 赢得的组，确保唯一分配
            remain = [
                g for g in groups
                if g["g_id"] not in fixed
                and g["g_id"] not in blocked
                and (aid in winners.get(g["g_id"], []) or g["g_id"] not in winners)
            ]
            # 复用 greedy：把 fixed 先放入，再往 remain 加
            # 先算 fixed 的已用预算
            usedC = sum(0.0 if ("resident_agents" in g and aid in g["resident_agents"]) else g["C"]
                        for g in groups if g["g_id"] in fixed)
            usedH = sum(g["H"] for g in groups if g["g_id"] in fixed)
            # 调整预算
            B_left = max(0.0, B_mem - usedC); H_left = max(0.0, eps_H - usedH)
            addS, *_ = _greedy_pack(remain, aid, lam_eff, B_left, H_left,
                                    alpha_base, w_dist, (pose[0],pose[1]))
            return list(fixed.union(addS))
        return _fill

    selections = {}
    for aid in agents.keys():
        S0 = results[aid]["selected"]
        S_final = apply_winners(S0, results[aid]["ranklist"], winners, aid, fill_budget_fn=make_filler(aid))
        selections[aid] = S_final

    # E) 汇总统计 & 更新价格
    total_C = 0.0; total_H = 0.0
    for aid, S in selections.items():
        for g in groups:
            if g["g_id"] in S:
                total_C += (0.0 if ("resident_agents" in g and aid in g["resident_agents"]) else g["C"])
                total_H += g["H"]
    p_new, tau_new = price_coordinator.update(total_C, total_H)

    stats = {"total_C": total_C, "total_H": total_H, "p": p_new, "tau": tau_new}
    debug = {
        "intents": intents,
        "winners": winners,
        "raw_results": results,
        "fallback_agents": list(fallback_agents),
    }
    return selections, stats, debug
