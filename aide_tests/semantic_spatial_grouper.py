# semantic_spatial_grouper.py
import logging
import math
import pdb
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Ellipse
from transformers import CLIPModel, CLIPTokenizerFast
from constants import LLM_SEMANTIC_GROUP_SYSTEM_PROMPT
# ---------- Config ----------
@dataclass
class GroupingCfg:
    clip_model: str = "openai/clip-vit-base-patch16"
    device: str = "cuda"
    # semantic "attention" (softmax over groups) threshold
    tau_assign: float = 0.25
    beta_softmax: float = 6.0
    # spatial split threshold (meters)
    spatial_radius_m: float = 15
    # per-subgroup proxy KV budget (tokens -> bytes proxy)
    max_subgroup_kv_bytes: int = 200_000
    # semantic assignment strategy: "clip" (default) or "llm"
    semantic_assignment_strategy: str = "clip"
    llm_model: str = "cogvlm2"
    llm_temperature: float = 0.2
    llm_top_p: float = 0.9
    llm_max_tokens: int = 16

# ---------- CLIP text encoder with cache ----------
class CLIPTextEncoder:
    def __init__(self, model_name: str, device="cuda"):
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.tok   = CLIPTokenizerFast.from_pretrained(model_name)
        self.device = device
        self.cache: Dict[str, np.ndarray] = {}
        torch.set_grad_enabled(False)

    def embed(self, text: str) -> np.ndarray:
        key = text.lower().strip()
        if key in self.cache:
            return self.cache[key]
        tokens = self.tok([key], return_tensors="pt").to(self.device)
        z = self.model.get_text_features(**tokens)  # (1, d)
        z = torch.nn.functional.normalize(z, dim=-1)  # L2 normalize
        vec = z.detach().cpu().numpy()[0]
        self.cache[key] = vec
        return vec

    def embed_texts(self, texts: Iterable[str]) -> List[np.ndarray]:
        """Vectorize a batch of texts, caching results to avoid duplicate encoding."""
        texts_list = list(texts)
        if not texts_list:
            return []

        keys = [t.lower().strip() for t in texts_list]
        missing = [(key, text) for key, text in zip(keys, texts_list) if key not in self.cache]

        if missing:
            to_tokenize = [key for key, _ in missing]
            tokens = self.tok(to_tokenize, padding=True, return_tensors="pt").to(self.device)
            z = self.model.get_text_features(**tokens)
            z = torch.nn.functional.normalize(z, dim=-1)
            vecs = z.detach().cpu().numpy()
            for (key, _), vec in zip(missing, vecs):
                self.cache[key] = vec

        return [self.cache[key] for key in keys]

# ---------- Data types ----------
@dataclass
class Det:
    det_id: int
    category: str
    xy_m: Tuple[float,float]
    status: str
    conf: float = 1.0

@dataclass
class SpaSubGroup:
    sid: int
    member_ids: List[int] = field(default_factory=list)
    centroid_xy: Tuple[float, float] = (0.0, 0.0)
    sem_centroid: Optional[np.ndarray] = None
    kv_bytes_est: int = 0
    tokens: List[str] = field(default_factory=list)
    parent_gid: Optional[int] = None
    v_score: float = 0.0
    h_score: float = 0.0
    c_score: float = 0.0
    selector_entry: Dict[str, float] = field(default_factory=dict)
    selector_lookup: Dict[str, object] = field(default_factory=dict)

@dataclass
class SemGroup:
    gid: int
    member_ids: List[int] = field(default_factory=list)
    sem_centroid: Optional[np.ndarray] = None
    subgroups: List[SpaSubGroup] = field(default_factory=list)
    kv_bytes_est: int = 0
    v_score: float = 0.0
    h_score: float = 0.0
    c_score: float = 0.0

# ---------- Utilities ----------
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

def kv_bytes_for_tokens(n_tokens:int, layers=24, heads=32, head_dim=128, dtype_bytes=2) -> int:
    return int(max(1, n_tokens) * layers * heads * head_dim * 2 * dtype_bytes)

def mean_xy(xys: List[Tuple[float,float]]) -> Tuple[float,float]:
    if not xys: return (0.0, 0.0)
    xs, ys = zip(*xys); return (float(np.mean(xs)), float(np.mean(ys)))

# ---------- The two-level grouper ----------
class SemanticSpatialGrouper:
    def __init__(self, cfg: GroupingCfg, llm_client: Optional[Any] = None):
        self.cfg = cfg
        self.llm_client = llm_client
        self.clip = CLIPTextEncoder(cfg.clip_model, cfg.device)
        self.sem_groups: List[SemGroup] = []
        self.next_gid = 0

        # store per-det info for fast stats
        self._det_emb: Dict[int, np.ndarray] = {}
        self._det_xy: Dict[int, Tuple[float,float]] = {}
        self._det_cat: Dict[int, str] = {}

        self.goal_text: Optional[str] = None
        self.goal_embedding: Optional[np.ndarray] = None

    # ---- public API ----
    def reset(self) -> None:
        """Reset grouping state for a new episode."""
        self.sem_groups.clear()
        self.next_gid = 0
        self._det_emb.clear()
        self._det_xy.clear()
        self._det_cat.clear()
        self.goal_text = None
        self.goal_embedding = None

    def set_llm_client(self, llm_client: Optional[Any]) -> None:
        """Attach or replace the LLM client used for semantic assignments."""
        self.llm_client = llm_client

    def add_detections(self, dets: List[Det]) -> List[SemGroup]:
        """
        Streaming update: for each new detection, (1) semantic assign -> group,
        (2) spatial assign -> subgroup, (3) update stats, (4) enforce budget.
        Returns current semantic groups (with subgroups) after update.
        """
        if not dets:
            return self.sem_groups

        # 1) build text embeddings (one per label)
        new_det_ids: List[int] = []
        new_texts: List[str] = []
        for d in dets:
            if d.status != "new":
                continue
            # pdb.set_trace()
            self._det_cat[d.det_id] = d.category
            self._det_xy[d.det_id]  = d.xy_m
            if d.det_id not in self._det_emb:
                new_det_ids.append(d.det_id)
                new_texts.append(d.category)

        if new_texts:
            embeddings = self.clip.embed_texts(new_texts)
            for det_id, emb in zip(new_det_ids, embeddings):
                self._det_emb[det_id] = emb

        # 2) assign each det
        strategy = getattr(self.cfg, "semantic_assignment_strategy", "clip").lower()
        current_gid: Optional[int] = None
        for d in dets:
            if d.status == "new":
                if strategy == "llm":
                    current_gid = self._assign_semantic_llm(d.det_id)
                else:
                    current_gid = self._assign_semantic(d.det_id)
                self._refresh_group_stats(current_gid)

            if current_gid is None or d.status != "new":
                inferred_gid = self._find_semantic_group_for_det(d.det_id)
                if inferred_gid is not None:
                    current_gid = inferred_gid

            if current_gid is None:
                logging.debug("[DEBUG] No semantic group found for det_id %s; skipping spatial assignment", d.det_id)
                continue

            self._assign_spatial(d.det_id, current_gid)

        # 3) merge nearby spatial subgroups within each semantic group
        self._merge_nearby_spatial_subgroups()
        
        # 4) (optional) merge highly similar semantic groups here if you like
        # pdb.set_trace()
        return self.sem_groups

    def set_goal_text(self, goal_text: Optional[str]) -> None:
        """Update the goal text and cache its CLIP embedding for scoring."""
        cleaned = goal_text.strip().lower() if goal_text else None
        if cleaned == self.goal_text:
            return

        if cleaned:
            self.goal_text = cleaned
            self.goal_embedding = self.clip.embed(cleaned)
        else:
            self.goal_text = None
            self.goal_embedding = None

        for G in list(self.sem_groups):
            self._refresh_group_stats(G.gid)

    def export_aird_units(self) -> List[Dict]:
        """
        Export *spatial subgroups* as the AIRD units (each with tokens + kv_bytes proxy).
        Return list of dict: {gid, sid, tokens, kv_bytes}
        """
        out=[]
        for G in self.sem_groups:
            for S in G.subgroups:
                if not S.tokens:
                    # create minimal tokens for AIRD (category@xy)
                    toks=[]
                    for det_id in S.member_ids:
                        cat = self._det_cat[det_id]
                        x,y = self._det_xy[det_id]
                        toks.append(f"obj:{cat}@({x:.2f},{y:.2f})")
                    S.tokens = toks
                if S.kv_bytes_est <= 0:
                    S.kv_bytes_est = kv_bytes_for_tokens(len(S.tokens))
                out.append({"gid": G.gid, "sid": S.sid, "tokens": S.tokens, "kv_bytes": S.kv_bytes_est})
        return out

    def get_selector_groups(self) -> Tuple[List[Dict], Dict[int, Dict]]:
        """Return planner-ready descriptors and lookup metadata."""
        groups: List[Dict] = []
        lookup: Dict[int, Dict] = {}
        for G in self.sem_groups:
            for S in G.subgroups:
                if not S.selector_entry:
                    continue
                groups.append(S.selector_entry)
                lookup[S.selector_entry["g_id"]] = S.selector_lookup
        return groups, lookup

    def prune_missing_detections(self, active_det_ids: Iterable[int]) -> None:
        """Drop cached detections that vanished from the map manager."""
        active_ids = set(active_det_ids)

        stale_ids = (set(self._det_emb) | set(self._det_xy) | set(self._det_cat)) - active_ids
        for det_id in stale_ids:
            self._det_emb.pop(det_id, None)
            self._det_xy.pop(det_id, None)
            self._det_cat.pop(det_id, None)

        for G in list(self.sem_groups):
            G.member_ids = [det_id for det_id in G.member_ids if det_id in active_ids]
            for S in list(G.subgroups):
                S.member_ids = [det_id for det_id in S.member_ids if det_id in active_ids]
                if not S.member_ids:
                    G.subgroups.remove(S)
                    continue
                self._refresh_spatial_stats(S)
            if not G.member_ids:
                self.sem_groups.remove(G)
                continue
            if not G.subgroups:
                first_id = G.member_ids[0]
                centroid = self._det_xy.get(first_id, (0.0, 0.0))
                sid = self._new_sid(G)
                S = SpaSubGroup(sid=sid, member_ids=list(G.member_ids), centroid_xy=centroid)
                self._refresh_spatial_stats(S)
                G.subgroups = [S]
            self._refresh_group_stats(G.gid)

    # ---- internals ----
    def _assign_semantic(self, det_id:int) -> int:
        """Choose best semantic group by cosine-similarity threshold; create new group if below."""
        import logging
        z = self._det_emb[det_id]
        logging.info(f"[DEBUG] _assign_semantic: Processing det_id {det_id} with embedding shape {z.shape}")
        
        if not self.sem_groups:
            logging.info(f"[DEBUG] _assign_semantic: No existing semantic groups, creating new group for det_id {det_id}")
            gid = self._create_sem_group([det_id])
            logging.info(f"[DEBUG] _assign_semantic: Created new semantic group {gid} for det_id {det_id}")
            return gid

        # avg cosine sim to each group's members (approx: to sem_centroid)
        sims = []
        logging.info(f"[DEBUG] _assign_semantic: Found {len(self.sem_groups)} existing semantic groups")
        for i, G in enumerate(self.sem_groups):
            if G.sem_centroid is None:
                sims.append(-1e9)
                logging.info(f"[DEBUG] _assign_semantic: Group {i} (gid={G.gid}) has no semantic centroid, similarity = -1e9")
                continue
            sim = cosine(z, G.sem_centroid)
            sims.append(sim)
            logging.info(f"[DEBUG] _assign_semantic: Group {i} (gid={G.gid}) similarity = {sim:.4f}")
        
        sims_np = np.array(sims, dtype=np.float32)
        best_idx = int(np.argmax(sims_np))
        best_sim = float(sims_np[best_idx])
        
        # Use raw cosine similarity directly (range [-1, 1])
        # Note: CLIP embeddings are L2-normalized, so cosine is in [0, 1] for typical cases
        # tau_assign threshold now directly corresponds to cosine similarity
        
        logging.info(f"[DEBUG] _assign_semantic: Best match is group {best_idx} with cosine similarity {best_sim:.4f}")
        logging.info(f"[DEBUG] _assign_semantic: Similarity threshold (tau_assign): {self.cfg.tau_assign}")
        logging.info(f"[DEBUG] _assign_semantic: Should assign to existing group: {best_sim >= self.cfg.tau_assign}")
        
        if best_sim >= self.cfg.tau_assign:
            logging.info(f"[DEBUG] _assign_semantic: ASSIGNING det_id {det_id} to existing group {self.sem_groups[best_idx].gid}")
            self.sem_groups[best_idx].member_ids.append(det_id)
            logging.info(f"[DEBUG] _assign_semantic: Group {self.sem_groups[best_idx].gid} now has members: {self.sem_groups[best_idx].member_ids}")
            return self.sem_groups[best_idx].gid
        else:
            logging.info(f"[DEBUG] _assign_semantic: CREATING NEW semantic group for det_id {det_id} (similarity {best_sim:.4f} < threshold {self.cfg.tau_assign})")
            gid = self._create_sem_group([det_id])
            logging.info(f"[DEBUG] _assign_semantic: Created new semantic group {gid} for det_id {det_id}")
            return gid

    def _assign_semantic_llm(self, det_id: int) -> int:
        """Use LLM reasoning to select a semantic group; fallback to CLIP when uncertain."""
        if not self.sem_groups:
            logging.info("[DEBUG] _assign_semantic_llm: No existing groups; creating new group for det_id %s", det_id)
            return self._create_sem_group([det_id])

        if self.llm_client is None:
            logging.warning("[DEBUG] _assign_semantic_llm: Missing LLM client; falling back to CLIP for det_id %s", det_id)
            return self._assign_semantic(det_id)

        prompt = self._build_llm_semantic_prompt(det_id)
        if not prompt:
            logging.warning("[DEBUG] _assign_semantic_llm: Prompt construction failed; using CLIP for det_id %s", det_id)
            return self._assign_semantic(det_id)

        response = self._query_llm_for_group_assignment(prompt)
        choice = self._parse_llm_group_choice(response, len(self.sem_groups))

        if choice == "new":
            logging.info("[DEBUG] _assign_semantic_llm: LLM requested new group for det_id %s", det_id)
            return self._create_sem_group([det_id])

        if isinstance(choice, int):
            selected_group = self.sem_groups[choice]
            selected_group.member_ids.append(det_id)
            logging.info("[DEBUG] _assign_semantic_llm: Assigned det_id %s to existing group index %s (gid=%s)", det_id, choice, selected_group.gid)
            return selected_group.gid

        logging.warning("[DEBUG] _assign_semantic_llm: Invalid LLM response '%s'; reverting to CLIP for det_id %s", response, det_id)
        return self._assign_semantic(det_id)

    def _build_llm_semantic_prompt(self, det_id: int) -> str:
        category = self._det_cat.get(det_id, "unknown")
        lines = []
        for idx, G in enumerate(self.sem_groups):
            cats = {self._det_cat.get(member_id) for member_id in G.member_ids}
            cats.discard(None)
            cats.discard("")
            if not cats:
                group_desc = "no known object categories"
            else:
                group_desc = ", ".join(sorted(cats))
            lines.append(f"Group {idx}: {group_desc}")

        groups_block = "\n".join(lines) if lines else "No existing groups."
        prompt = f"{groups_block}\nIncoming object: {category}"
        return prompt

    def _query_llm_for_group_assignment(self, prompt: str) -> str:
        if self.llm_client is None:
            return ""
        try:
            messages = [
                {
                    "role": "system",
                    "content": LLM_SEMANTIC_GROUP_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ]
            model_name = getattr(self.cfg, "llm_model", "llama")
            temperature = getattr(self.cfg, "llm_temperature", 0.2)
            top_p = getattr(self.cfg, "llm_top_p", 0.9)
            max_tokens = getattr(self.cfg, "llm_max_tokens", 16)
            _, content = self.llm_client.create_chat_completion(
                model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                use_stream=False,
            )
            # Extract integer or -1 from response
            if content:
                match = re.search(r"-?\d+", content)
                content = match.group(0) if match else "-1"

            return content or ""
        except Exception as exc:
            logging.warning("[DEBUG] _assign_semantic_llm: LLM query failed: %s", exc)
            return ""

    def _parse_llm_group_choice(self, raw_response: str, num_groups: int):
        if not raw_response:
            return None
        cleaned = raw_response.strip()
        if not cleaned:
            return None
        # Check for -1 (no suitable group) or "new" keyword
        if cleaned == "-1" or cleaned.upper() == "NEW":
            return "new"
        match = re.search(r"-?\d+", cleaned)
        if not match:
            return None
        idx = int(match.group(0))
        if 0 <= idx < num_groups:
            return idx
        if 1 <= idx <= num_groups:
            return idx - 1
        return None

    def _assign_spatial(self, det_id:int, gid:int) -> int:
        """within a semantic group, join nearest spatial subgroup within radius, else create."""
        import logging
        G = self._get_group_by_gid(gid)
        if G is None:
            logging.info(f"[DEBUG] _assign_spatial: Group {gid} not found for det_id {det_id}")
            return -1
        xy = self._det_xy[det_id]
        logging.info(f"[DEBUG] _assign_spatial: Processing det_id {det_id} at position {xy} in semantic group {gid}")
        
        if not G.subgroups:
            logging.info(f"[DEBUG] _assign_spatial: No existing subgroups in group {gid}, creating new subgroup")
            sid = self._create_spa_subgroup(G, [det_id], init_center=xy)
            self._refresh_group_stats(G.gid)
            logging.info(f"[DEBUG] _assign_spatial: Created new subgroup {sid} for det_id {det_id}")
            return sid
        
        # choose nearest subgroup
        logging.info(f"[DEBUG] _assign_spatial: Found {len(G.subgroups)} existing subgroups in group {gid}")
        
        # Original approach: distance to centroid
        centroid_dists = [math.hypot(xy[0]-S.centroid_xy[0], xy[1]-S.centroid_xy[1]) for S in G.subgroups]
        centroid_k = int(np.argmin(centroid_dists))
        centroid_distance = centroid_dists[centroid_k]
        
        # New approach: distance to nearest object in subgroup
        nearest_object_dists = []
        for S in G.subgroups:
            if not S.member_ids:
                nearest_object_dists.append(float('inf'))
                continue
            min_dist = min(math.hypot(xy[0]-self._det_xy[member_id][0], xy[1]-self._det_xy[member_id][1]) 
                          for member_id in S.member_ids)
            nearest_object_dists.append(min_dist)
        
        nearest_object_k = int(np.argmin(nearest_object_dists))
        nearest_object_distance = nearest_object_dists[nearest_object_k]
        
        # Use the new approach (nearest object) for decision
        k = nearest_object_k
        nearest_distance = nearest_object_distance
        nearest_subgroup = G.subgroups[k]
        
        logging.info(f"[DEBUG] _assign_spatial: Centroid-based approach:")
        logging.info(f"  - Nearest subgroup: {centroid_k}, distance: {centroid_distance:.2f} pixels")
        logging.info(f"  - Centroid: {G.subgroups[centroid_k].centroid_xy}")
        
        logging.info(f"[DEBUG] _assign_spatial: Nearest-object approach:")
        logging.info(f"  - Nearest subgroup: {nearest_object_k}, distance: {nearest_object_distance:.2f} pixels")
        logging.info(f"  - Subgroup members: {nearest_subgroup.member_ids}")
        for member_id in nearest_subgroup.member_ids:
            member_xy = self._det_xy[member_id]
            member_dist = math.hypot(xy[0]-member_xy[0], xy[1]-member_xy[1])
            logging.info(f"    - Member {member_id} at {member_xy}: distance {member_dist:.2f}")
        
        logging.info(f"[DEBUG] _assign_spatial: Using nearest-object approach for decision")
        logging.info(f"[DEBUG] _assign_spatial: Spatial radius threshold: {self.cfg.spatial_radius_m} pixels")
        logging.info(f"[DEBUG] _assign_spatial: Should merge: {nearest_distance <= self.cfg.spatial_radius_m}")
        
        if nearest_distance <= self.cfg.spatial_radius_m:
            logging.info(f"[DEBUG] _assign_spatial: MERGING det_id {det_id} into existing subgroup {nearest_subgroup.sid}")
            subgroup = G.subgroups[k]
            if det_id not in subgroup.member_ids:
                subgroup.member_ids.append(det_id)
                logging.info(f"[DEBUG] _assign_spatial: Added det_id {det_id} to subgroup {subgroup.sid}, members now: {subgroup.member_ids}")
            else:
                logging.info(f"[DEBUG] _assign_spatial: det_id {det_id} already in subgroup {subgroup.sid}")
            self._refresh_spatial_stats(subgroup)
            self._enforce_subgroup_budget(G, subgroup)

            assigned_sid = subgroup.sid
            for candidate in G.subgroups:
                if det_id in candidate.member_ids:
                    assigned_sid = candidate.sid
                    break
        else:
            logging.info(f"[DEBUG] _assign_spatial: CREATING NEW subgroup for det_id {det_id} (distance {nearest_distance:.2f} > threshold {self.cfg.spatial_radius_m})")
            logging.info(f"[DEBUG] _assign_spatial: Comparison - Centroid distance: {centroid_distance:.2f}, Nearest object distance: {nearest_distance:.2f}")
            assigned_sid = self._create_spa_subgroup(G, [det_id], init_center=xy)
            logging.info(f"[DEBUG] _assign_spatial: Created new subgroup {assigned_sid} for det_id {det_id}")

        self._refresh_group_stats(G.gid)
        logging.info(f"[DEBUG] _assign_spatial: Final assignment - det_id {det_id} assigned to subgroup {assigned_sid}")
        return assigned_sid

    # def _refresh_group_stats(self, gid:int):
    #     G = self._get_group_by_gid(gid)
    #     if G is None: return
    #     # semantic centroid
    #     Z = [self._det_emb[i] for i in G.member_ids]
    #     mu = np.mean(np.stack(Z, axis=0), axis=0)
    #     mu = mu / (np.linalg.norm(mu) + 1e-9)
    #     G.sem_centroid = mu
    #     # update group kv proxy (sum of subgroups)
    #     G.kv_bytes_est = int(sum(S.kv_bytes_est if S.kv_bytes_est>0 else kv_bytes_for_tokens(len(S.member_ids))
    #                              for S in G.subgroups))

    # def _refresh_spatial_stats(self, S: SpaSubGroup):
    #     xys = [self._det_xy[i] for i in S.member_ids]
    #     S.centroid_xy = mean_xy(xys)
    #     S.kv_bytes_est = kv_bytes_for_tokens(len(S.member_ids))
    #     # tokens lazily generated when exporting

    def _enforce_subgroup_budget(self, G: SemGroup, S: SpaSubGroup):
        """split subgroup if proxy KV exceeds per-subgroup budget"""
        if S.kv_bytes_est <= self.cfg.max_subgroup_kv_bytes:
            return
        # simple split: by farthest-first into two halves
        ids = S.member_ids
        if len(ids) < 4:  # too small, keep as is
            return
        # pick two farthest as seeds
        pts = np.array([self._det_xy[i] for i in ids])
        D = np.linalg.norm(pts[:,None,:] - pts[None,:,:], axis=-1)
        a,b = np.unravel_index(np.argmax(D), D.shape)
        left, right = [ids[a]], [ids[b]]
        for idx in range(len(ids)):
            if idx in (a,b): continue
            to_left = np.linalg.norm(pts[idx] - pts[a])
            to_right= np.linalg.norm(pts[idx] - pts[b])
            (left if to_left<=to_right else right).append(ids[idx])
        # replace S with two new subgroups
        G.subgroups.remove(S)
        Sa = SpaSubGroup(sid=self._new_sid(G), member_ids=left)
        Sb = SpaSubGroup(sid=self._new_sid(G), member_ids=right)
        self._refresh_spatial_stats(Sa); self._refresh_spatial_stats(Sb)
        G.subgroups += [Sa, Sb]

    def _merge_nearby_spatial_subgroups(self):
        """Merge spatial subgroups that are within spatial_radius_m of each other"""
        import logging
        logging.info(f"[DEBUG] Starting spatial subgroup merging for {len(self.sem_groups)} semantic groups")
        for G in self.sem_groups:
            logging.info(f"[DEBUG] Processing semantic group {G.gid} with {len(G.subgroups)} subgroups")
            if len(G.subgroups) <= 1:
                logging.info(f"[DEBUG] Group {G.gid} has only {len(G.subgroups)} subgroups, skipping merge")
                self._refresh_group_stats(G.gid)
                continue

            # Use Union-Find to efficiently merge nearby subgroups
            subgroups = G.subgroups.copy()
            n = len(subgroups)
            parent = list(range(n))
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py
                    return True
                return False
            
            # Find all pairs within radius and union them
            for i in range(n):
                for j in range(i + 1, n):
                    dist = math.hypot(
                        subgroups[i].centroid_xy[0] - subgroups[j].centroid_xy[0],
                        subgroups[i].centroid_xy[1] - subgroups[j].centroid_xy[1]
                    )
                    # Debug logging for spatial merging
                    import logging
                    logging.info(f"[DEBUG] Spatial merge check: subgroup {i} vs {j}")
                    logging.info(f"  - Subgroup {i} centroid: {subgroups[i].centroid_xy}")
                    logging.info(f"  - Subgroup {j} centroid: {subgroups[j].centroid_xy}")
                    logging.info(f"  - Distance: {dist:.2f} pixels")
                    logging.info(f"  - Spatial radius: {self.cfg.spatial_radius_m} pixels")
                    logging.info(f"  - Should merge: {dist <= self.cfg.spatial_radius_m}")
                    
                    if dist <= self.cfg.spatial_radius_m:
                        logging.info(f"  - MERGING subgroups {i} and {j}")
                        union(i, j)
                    else:
                        logging.info(f"  - NOT merging subgroups {i} and {j} (distance too large)")
            
            # Group subgroups by their root parent
            groups = {}
            for i in range(n):
                root = find(i)
                if root not in groups:
                    groups[root] = []
                groups[root].append(subgroups[i])
            
            import logging
            logging.info(f"[DEBUG] Union-Find results: {len(groups)} groups after merging")
            for root, group_subgroups in groups.items():
                logging.info(f"  - Group {root}: {len(group_subgroups)} subgroups")
                for sg in group_subgroups:
                    logging.info(f"    - Subgroup {sg.sid}: centroid={sg.centroid_xy}, members={sg.member_ids}")
            
            # Merge subgroups in each group
            new_subgroups = []
            for group_subgroups in groups.values():
                if len(group_subgroups) == 1:
                    logging.info(f"[DEBUG] Keeping single subgroup: {group_subgroups[0].sid}")
                    new_subgroups.append(group_subgroups[0])
                else:
                    # Merge all subgroups in this group
                    logging.info(f"[DEBUG] Merging {len(group_subgroups)} subgroups into one")
                    merged = group_subgroups[0]
                    for other in group_subgroups[1:]:
                        logging.info(f"  - Merging subgroup {other.sid} (members: {other.member_ids}) into {merged.sid}")
                        merged.member_ids.extend(other.member_ids)
                    self._refresh_spatial_stats(merged)
                    self._enforce_subgroup_budget(G, merged)
                    logging.info(f"  - Final merged subgroup: members={merged.member_ids}, centroid={merged.centroid_xy}")
                    new_subgroups.append(merged)

            logging.info(f"[DEBUG] Final result: {len(new_subgroups)} subgroups after merging")
            G.subgroups = new_subgroups
            self._refresh_group_stats(G.gid)

    def _create_spa_subgroup(self, G: SemGroup, member_ids: List[int],
                             init_center: Tuple[float, float]) -> int:
        """
        Create a new spatial subgroup under semantic group G and return its sid.
        - member_ids: detection ids to initialize the subgroup
        - init_center: (x,y) meters used as initial centroid; we recompute stats right after
        """
        sid = self._new_sid(G)
        S = SpaSubGroup(sid=sid, member_ids=list(member_ids), centroid_xy=init_center)
        self._refresh_spatial_stats(S)   # computes centroid & kv_bytes_est
        G.subgroups.append(S)
        return sid
        
    # ---- constructors & helpers ----
    def _create_sem_group(self, member_ids: List[int]) -> int:
        G = SemGroup(gid=self.next_gid, member_ids=list(member_ids), subgroups=[])
        self.next_gid += 1
        # start with one spatial subgroup initialized at first det
        first_xy = self._det_xy[member_ids[0]]
        S = SpaSubGroup(sid=0, member_ids=list(member_ids), centroid_xy=first_xy)
        self._refresh_spatial_stats(S)
        G.subgroups = [S]
        self.sem_groups.append(G)
        self._refresh_group_stats(G.gid)
        return G.gid

    def _get_group_by_gid(self, gid:int) -> Optional[SemGroup]:
        for G in self.sem_groups:
            if G.gid == gid: return G
        return None

    def _find_semantic_group_for_det(self, det_id: int) -> Optional[int]:
        for G in self.sem_groups:
            if det_id in G.member_ids:
                return G.gid
        return None

    def _new_sid(self, G: SemGroup) -> int:
        if not G.subgroups: return 0
        return 1 + max(s.sid for s in G.subgroups)

    def _build_subgroup_text(self, S: SpaSubGroup) -> Optional[str]:
        """Create a brief textual summary of objects contained in the subgroup."""
        labels = [self._det_cat.get(det_id) for det_id in S.member_ids]
        labels = [label for label in labels if label]
        if not labels:
            return None

        counts = Counter(labels)
        parts = []
        for label, count in counts.items():
            if count > 1:
                suffix = "" if label.endswith("s") else "s"
                parts.append(f"{count} {label}{suffix}")
            else:
                parts.append(label)

        centroid = S.centroid_xy
        centroid_text = f"near coordinates ({centroid[0]:.1f}, {centroid[1]:.1f})"
        return "Group containing " + ", ".join(parts) + f" {centroid_text}"

    def _estimate_subgroup_scores(self, G: SemGroup, S: SpaSubGroup) -> None:
        """
        Update V/H/C scores. Plug in real LLM/MI metrics where indicated.
        """
        if not S.member_ids:
            S.v_score = S.h_score = S.c_score = 0.0
            return

        # V-score: similarity between subgroup semantic centroid and goal text embedding
        if self.goal_embedding is not None:
            subgroup_vec = S.sem_centroid if S.sem_centroid is not None else self._compute_subgroup_semantic_centroid(S)
            S.sem_centroid = subgroup_vec
            if subgroup_vec is not None:
                cos_sim = float(cosine(subgroup_vec, self.goal_embedding))
                if math.isnan(cos_sim):
                    cos_sim = 0.0
                S.v_score = (cos_sim + 1.0) * 0.5  # map [-1,1] -> [0,1]
            else:
                S.v_score = 0.0
        else:
            S.v_score = 0.0

        # H-score: variance of distances between object embeddings and goal embedding
        if self.goal_embedding is not None:
            dists = []
            for det_id in S.member_ids:
                emb = self._det_emb.get(det_id)
                if emb is None:
                    continue
                cos_sim = float(cosine(emb, self.goal_embedding))
                if math.isnan(cos_sim):
                    continue
                # convert similarity to distance in [0, 2]
                dist = 1.0 - cos_sim
                dist = max(0.0, min(2.0, dist))
                dists.append(dist)
            if len(dists) >= 1:
                S.h_score = float(np.var(dists))
            else:
                S.h_score = 0.0
        else:
            S.h_score = 0.0

        # C-score: subgroup KV plus group’s aggregate KV
        subgroup_tokens = len(S.member_ids)
        subgroup_bytes = kv_bytes_for_tokens(subgroup_tokens)
        S.c_score = float(subgroup_bytes + G.kv_bytes_est)

    def _update_subgroup_selector_payload(self, G: SemGroup, S: SpaSubGroup) -> None:
        """Materialize planner-facing payload for a subgroup."""
        if not S.member_ids:
            S.selector_entry = {}
            S.selector_lookup = {}
            return

        cx, cy = S.centroid_xy if S.centroid_xy else (None, None)
        if cx is None or cy is None:
            S.selector_entry = {}
            S.selector_lookup = {}
            return

        if isinstance(cx, float) and (math.isnan(cx) or math.isnan(cy)):
            S.selector_entry = {}
            S.selector_lookup = {}
            return

        g_id = (int(G.gid) << 16) | int(S.sid)

        S.selector_entry = {
            "g_id": g_id,
            "V": float(S.v_score),
            "H": float(S.h_score),
            "C": float(S.kv_bytes_est),
            "cx": float(cx),
            "cy": float(cy),
        }

        S.selector_lookup = {
            "centroid": (float(cx), float(cy)),
            "sem_group": G.gid,
            "subgroup": S.sid,
            "members": list(S.member_ids),
        }

    def _compute_subgroup_semantic_centroid(self, S: SpaSubGroup) -> Optional[np.ndarray]:
        if not S.member_ids:
            return None
        embeds = [self._det_emb.get(det_id) for det_id in S.member_ids]
        embeds = [emb for emb in embeds if emb is not None]
        if not embeds:
            return None
        mu = np.mean(np.stack(embeds, axis=0), axis=0)
        norm = float(np.linalg.norm(mu))
        if not math.isfinite(norm) or norm < 1e-9:
            return None
        return mu / norm

    def _refresh_spatial_stats(self, S: SpaSubGroup):
        if S.member_ids:
            seen = set()
            unique_members = []
            for det_id in S.member_ids:
                if det_id in seen:
                    continue
                seen.add(det_id)
                unique_members.append(det_id)
            if len(unique_members) != len(S.member_ids):
                S.member_ids = unique_members
        xys = [self._det_xy[i] for i in S.member_ids]
        S.centroid_xy = mean_xy(xys)
        S.kv_bytes_est = kv_bytes_for_tokens(len(S.member_ids))
        S.sem_centroid = self._compute_subgroup_semantic_centroid(S)
        if S.parent_gid is not None:
            parent = self._get_group_by_gid(S.parent_gid)
            if parent:
                self._estimate_subgroup_scores(parent, S)

    def _refresh_group_stats(self, gid: int):
        G = self._get_group_by_gid(gid)
        if G is None:
            return
        if G.member_ids:
            G.member_ids = list(dict.fromkeys(G.member_ids))
        if G.member_ids:
            Z = [self._det_emb[i] for i in G.member_ids]
            mu = np.mean(np.stack(Z, axis=0), axis=0)
            G.sem_centroid = mu / (np.linalg.norm(mu) + 1e-9)
        else:
            G.sem_centroid = None

        G.kv_bytes_est = int(sum(
            S.kv_bytes_est if S.kv_bytes_est > 0 else kv_bytes_for_tokens(len(S.member_ids))
            for S in G.subgroups
        ))

        for S in G.subgroups:
            S.parent_gid = G.gid
            S.sem_centroid = self._compute_subgroup_semantic_centroid(S)
            self._estimate_subgroup_scores(G, S)
            self._update_subgroup_selector_payload(G, S)

        if G.subgroups:
            G.v_score = float(np.mean([S.v_score for S in G.subgroups]))
            G.h_score = float(np.mean([S.h_score for S in G.subgroups]))
            G.c_score = float(np.mean([S.c_score for S in G.subgroups]))
        else:
            G.v_score = G.h_score = G.c_score = 0.0

            
    # ---- helper: convert all_object -> DataFrame ----
    def df_from_all_object(self, all_object: List[Dict],
                        res_m_per_px: float = 0.05,
                        origin_xy_m: Tuple[float, float] = (0.0, 0.0)) -> pd.DataFrame:
        """
        all_object: list of dicts like
            {'category': 'chair', 'category_id': 0,
            'map_position': {'x': 319, 'y': 283},
            'bounding_box': {'min_x': 317, 'min_y': 279, 'max_x': 324, 'max_y': 289},
            'area': 43, 'confidence': 0.26, 'step': 1, 'object_id': 0, 'object_state': 'new'}
        """
        rows = []
        for o in all_object.values():
            mx = o.get("map_position", {}).get("x", None)
            my = o.get("map_position", {}).get("y", None)
            if mx is None or my is None:
                # skip if no map_position
                continue
            x_m =  float(mx) 
            y_m =  float(my)
            rows.append({
                "object_id": int(o.get("object_id", -1)),
                "category": str(o.get("category", "unknown")),
                "category_id": int(o.get("category_id", -1)),
                "x_m": x_m, "y_m": y_m,
                "map_x": int(mx), "map_y": int(my),
                "confidence": float(o.get("confidence", 0.0)),
                "step": int(o.get("step", -1)),
                "object_state": str(o.get("object_state", ""))
            })
        df = pd.DataFrame(rows).dropna(subset=["x_m","y_m"])
        return df

    # ---- helper: naive spatial clustering (within each semantic group) ----
    def spatial_subgroups(self, df: pd.DataFrame,
                        spatial_radius_m: float = 1.2) -> pd.DataFrame:
        """
        Create a simple spatial clustering per semantic group:
        - semantic group = category code (df['category'].astype('category').cat.codes)
        - within each semantic group, build subgroups by greedy radius clustering
        Returns df with added 'sem_gid' and 'spa_sid' columns.
        """
        df = df.copy()
        df["sem_gid"] = df["category"].astype("category").cat.codes
        df["spa_sid"] = -1
        for gid, block in df.groupby("sem_gid"):
            idxs = list(block.index)
            centers: List[Tuple[float,float]] = []
            sid = 0
            for i in idxs:
                xi, yi = float(df.at[i,"x_m"]), float(df.at[i,"y_m"])
                assigned = False
                for k, (cx,cy) in enumerate(centers):
                    if math.hypot(xi - cx, yi - cy) <= spatial_radius_m:
                        df.at[i, "spa_sid"] = k
                        # update center (incremental mean): for simplicity, recompute later
                        assigned = True
                        break
                if not assigned:
                    centers.append((xi, yi))
                    df.at[i, "spa_sid"] = sid
                    sid += 1
            # Recompute centers (not strictly necessary for plotting)
        return df

    # ---- helper: draw covariance ellipse for a set of (x,y) points ----
    def draw_cov_ellipse(self, ax, xs: np.ndarray, ys: np.ndarray, n_std: float = 2.0,
                        edge_style: str = "dashed", lw: float = 1.5, label: Optional[str] = None):
        if len(xs) < 2:
            # draw a small circle
            e = Ellipse((float(xs[0]), float(ys[0])), width=0.2, height=0.2,
                        fill=False, linestyle=edge_style, linewidth=lw)
            ax.add_patch(e)
            if label:
                ax.text(float(xs[0]), float(ys[0]), label)
            return
        x = np.array(xs); y = np.array(ys)
        cov = np.cov(x, y)
        vals, vecs = np.linalg.eig(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = math.degrees(math.atan2(vecs[1,0], vecs[0,0]))
        width, height = 2 * n_std * np.sqrt(vals + 1e-9)
        e = Ellipse((np.mean(x), np.mean(y)), width=width, height=height,
                    angle=theta, fill=False, linestyle=edge_style, linewidth=lw)
        ax.add_patch(e)
        if label:
            ax.text(float(np.mean(x)), float(np.mean(y)), label)

    # ---- main plotting ----
    def visualize_grouping(self, step,all_object: List[Dict],
                        assignments: Optional[Dict[int, Tuple[int,int]]] = None,
                        res_m_per_px: float = 0.05,
                        origin_xy_m: Tuple[float,float] = (0.0, 0.0),
                        spatial_radius_m: float = 1.2,
                        out_prefix: str = "group_vis"):
        """
        - If `assignments` is provided as {object_id: (sem_gid, spa_sid)}, we use it.
        - Else we build a simple two-level grouping (sem=category, spa=radius clusters).
        Saves two PNGs to /mnt/data and returns their paths.
        """
        df = self.df_from_all_object(all_object, res_m_per_px, origin_xy_m)
        if assignments is not None:
            # Convert sem_groups to assignments dictionary if needed
            if isinstance(assignments, list):  # assignments is self.sem_groups
                assignments_dict = {}
                for sem_group in assignments:
                    for subgroup in sem_group.subgroups:
                        for member_id in subgroup.member_ids:
                            assignments_dict[member_id] = (sem_group.gid, subgroup.sid)
                assignments = assignments_dict
            
            df["sem_gid"] = df["object_id"].map(lambda oid: assignments.get(oid, (-1,-1))[0])
            df["spa_sid"] = df["object_id"].map(lambda oid: assignments.get(oid, (-1,-1))[1])
        else:
            df = self.spatial_subgroups(df, spatial_radius_m=spatial_radius_m)

        # ---- Plot 1: scatter with semantics ----
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        # group by category for auto-colors
        for cat, block in df.groupby("category"):
            ax1.scatter(block["x_m"], block["y_m"], label=f"{cat}", s=18)
            # annotate a few points per category
            for _, r in block.sample(min(10, len(block)), random_state=0).iterrows():
                ax1.annotate(f'{r["category"][:3]}#{int(r["object_id"])}', (r["x_m"], r["y_m"]), fontsize=7)
        ax1.set_title("All objects: XY scatter by semantic category")
        ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
        ax1.legend(loc="best", fontsize=8)
        fig1.tight_layout()
        p1 = "./aide_tests/grouping_results/{}_scatter_{}.png".format(out_prefix, step)
        fig1.savefig(p1, dpi=150)

        # ---- Plot 2: hierarchical: semantic groups + spatial subgroups ----
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        ax2.scatter(df["x_m"], df["y_m"], s=12)  # all points

        # draw semantic-group ellipses (dashed)
        for gid, block in df.groupby("sem_gid"):
            xs, ys = block["x_m"].values, block["y_m"].values
            self.draw_cov_ellipse(ax2, xs, ys, n_std=2.0, edge_style="dashed", lw=1.5, label=f"G{int(gid)}")
            # label group
            ax2.annotate(f"G{int(gid)}", (float(np.mean(xs)), float(np.mean(ys))), fontsize=9)

            # draw spatial subgroups (solid ellipse) inside each sem group
            for sid, b2 in block.groupby("spa_sid"):
                xs2, ys2 = b2["x_m"].values, b2["y_m"].values
                self.draw_cov_ellipse(ax2, xs2, ys2, n_std=1.2, edge_style="solid", lw=1.0, label=None)
                # mark subgroup centroid
                cx, cy = float(np.mean(xs2)), float(np.mean(ys2))
                ax2.plot([cx], [cy], marker="x")
                ax2.annotate(f"G{int(gid)}/S{int(sid)}", (cx, cy), fontsize=7)

        ax2.set_title("Hierarchical grouping: semantic (dashed) → spatial (solid)")
        ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
        fig2.tight_layout()
        p2 = "./aide_tests/grouping_results/{}_hier_{}.png".format(out_prefix, step)
        fig2.savefig(p2, dpi=150)
        
        # Close figures to prevent memory accumulation
        plt.close(fig1)
        plt.close(fig2)

        return {"scatter_path": p1, "hier_path": p2}

        # # ---- Demo (synthetic) so you can see what the outputs look like now ----
        # # If you want to run on your real data, call visualize_grouping(all_object, ...) instead.
        # if True:
        #     rng = np.random.default_rng(0)
        #     # create a tiny synthetic "all_object" with 3 semantics and two spatial clusters each
        #     cats = ["chair","sink","plant"]
        #     all_object_demo = []
        #     oid = 0
        #     for ci, cat in enumerate(cats):
        #         for k in range(2):  # two clusters per category
        #             cx, cy = rng.uniform(0, 8), rng.uniform(0, 8)
        #             for _ in range(12):
        #                 x = int((cx + rng.normal(scale=0.3)) / 0.05)
        #                 y = int((cy + rng.normal(scale=0.3)) / 0.05)
        #                 all_object_demo.append({
        #                     "category": cat, "category_id": ci,
        #                     "map_position": {"x": x, "y": y},
        #                     "bounding_box": {"min_x": x-2, "min_y": y-2, "max_x": x+2, "max_y": y+2},
        #                     "area": int(rng.integers(30, 80)),
        #                     "confidence": float(rng.uniform(0.6, 0.95)),
        #                     "step": int(rng.integers(1, 10)),
        #                     "object_id": oid, "object_state": "new"
        #                 })
        #                 oid += 1
        #     paths = visualize_grouping(all_object_demo, assignments=None,
        #                             res_m_per_px=0.05, origin_xy_m=(0.0, 0.0),
        #                             spatial_radius_m=1.2, out_prefix="demo_group")
        #     print("Saved demo figures:", paths)
        
