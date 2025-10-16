# ------------ intents.py ------------
from typing import Dict, List, Tuple

def build_intent(agent_id: str, ranklist: List[Tuple[int,float]], K: int = 2):
    # ranklist: [(g_id, bid_desc), ...]
    return {"agent": agent_id, "intent": ranklist[:K]}

def resolve_conflicts(intents: List[Dict], capacity: Dict[int,int] = None):
    """
    本地根据所有 intents 得到每个 g 的赢家集合；capacity[g]=同时允许的机器人数量（默认1）
    返回 winners: {g_id: [agent_ids...]}
    """
    from collections import defaultdict
    capacity = capacity or {}
    bids = defaultdict(list)  # g_id -> [(agent,bid), ...]
    for it in intents:
        a = it["agent"]
        for (g,b) in it["intent"]:
            bids[g].append((a, float(b)))
    winners = {}
    for g, arr in bids.items():
        arr.sort(key=lambda x: (-x[1], x[0]))  # bid大优先，平局按agent字母序
        cap = capacity.get(g, 1)
        winners[g] = [a for (a,_) in arr[:cap]]
    return winners

def apply_winners(local_selected: List[int], local_ranklist: List[Tuple[int,float]],
                  winners: Dict[int, List[str]], agent_id: str, fill_budget_fn):
    """
    若本机在某些 g 上输掉（不在 winners[g]），则把这些 g 删除，并按 ranklist 向下选择替补，
    由 fill_budget_fn(S_partial) 在预算内补齐，返回最终 S。
    """
    S = set(local_selected)
    # 删掉输的组
    for g in list(S):
        if g in winners and agent_id not in winners[g]:
            S.remove(g)
    # 用 ranklist 下一个补齐（不破坏预算）
    S_final = fill_budget_fn(list(S), local_ranklist)
    return S_final
