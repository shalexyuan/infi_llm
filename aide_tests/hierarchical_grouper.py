
import logging
import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any, Set
import numpy as np
import pdb
try:
    from aide_tests.semantic_spatial_grouper import CLIPTextEncoder
except Exception:  # pragma: no cover - optional dependency
    CLIPTextEncoder = None  # type: ignore


def _text_to_counter(text: str) -> Counter:
    tokens = [tok for tok in str(text).lower().split() if tok]
    if not tokens:
        tokens = [str(text).lower()]
    return Counter(tokens)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


@dataclass
class SpatialCluster:
    cid: int
    member_ids: set = field(default_factory=set)
    member_positions: Dict[int, np.ndarray] = field(default_factory=dict)
    centroid: Optional[np.ndarray] = None
    v_score: float = 0.0
    h_score: float = 0.0

    def add_member(self, det_id: int, pos: np.ndarray) -> None:
        self.member_ids.add(det_id)
        self.member_positions[det_id] = pos
        self._refresh_centroid()

    def update_member(self, det_id: int, pos: np.ndarray) -> None:
        if det_id in self.member_positions:
            self.member_positions[det_id] = pos
            self._refresh_centroid()

    def remove_member(self, det_id: int) -> None:
        if det_id in self.member_ids:
            self.member_ids.remove(det_id)
            self.member_positions.pop(det_id, None)
            self._refresh_centroid()

    def _refresh_centroid(self) -> None:
        if self.member_positions:
            stacked = np.stack(list(self.member_positions.values()), axis=0)
            self.centroid = stacked.mean(axis=0)
        else:
            self.centroid = None

    def closest_distance(self, pos: np.ndarray) -> float:
        if not self.member_positions:
            return float("inf")
        distances = [np.linalg.norm(pos - mp) for mp in self.member_positions.values()]
        return min(distances)


@dataclass
class SemanticGroup:
    gid: int
    semantic_vector: np.ndarray
    members: set = field(default_factory=set)
    clusters: Dict[int, SpatialCluster] = field(default_factory=dict)
    vector_sum: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    next_cluster_id: int = 0
    v_score: float = 0.0
    h_score: float = 0.0

    def __post_init__(self):
        self.vector_sum = self.semantic_vector.copy()

    def add_member(self, det_id: int, vector: np.ndarray) -> None:
        self.members.add(det_id)
        self.vector_sum = self.vector_sum + vector
        norm = np.linalg.norm(self.vector_sum)
        if norm > 1e-9:
            self.semantic_vector = self.vector_sum / norm

    def remove_member(self, det_id: int, vector_lookup: Dict[int, np.ndarray]) -> None:
        if det_id in self.members:
            self.members.remove(det_id)
            vec = vector_lookup.get(det_id)
            if vec is not None:
                self.vector_sum = self.vector_sum - vec
                norm = np.linalg.norm(self.vector_sum)
                if norm > 1e-9:
                    self.semantic_vector = self.vector_sum / norm


class HierarchicalGrouper:
    """
    Lightweight hierarchical grouper with two levels:
      1. Semantic grouping (clip similarity or LLM query).
      2. Spatial grouping within each semantic group.
    """

    def __init__(
        self,
        semantic_policy: str = "clip",
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        semantic_threshold: float = 0.6,
        spatial_threshold: float = 3.0,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.semantic_policy = semantic_policy.lower()
        self.semantic_threshold = semantic_threshold
        self.spatial_threshold = spatial_threshold
        self.llm_client = llm_client
        self._clip_model_name = clip_model
        self._clip_device = device

        self.encoder: Optional[Any] = None
        if self.semantic_policy == "clip":
            if CLIPTextEncoder is None:
                raise RuntimeError(
                    "CLIPTextEncoder import failed. Install transformers to use clip-based grouping."
                )
            self.encoder = CLIPTextEncoder(clip_model, device=device)

        self.groups: Dict[int, SemanticGroup] = {}
        self.det_to_group: Dict[int, int] = {}
        self.det_to_cluster: Dict[int, Tuple[int, int]] = {}
        self.det_vectors: Dict[int, np.ndarray] = {}
        self.det_positions: Dict[int, np.ndarray] = {}
        self.det_labels: Dict[int, str] = {}
        self.next_gid = 0
        self.goal_text: Optional[str] = None
        self.goal_vector: Optional[np.ndarray] = None
        self._unique_category_set: Set[str] = set()
        self.unique_categories: List[str] = []

    # --- public API -----------------------------------------------------
    def reset(self) -> None:
        self.groups.clear()
        self.det_to_group.clear()
        self.det_to_cluster.clear()
        self.det_vectors.clear()
        self.det_positions.clear()
        self.det_labels.clear()
        self.next_gid = 0
        self.goal_vector = None
        self._unique_category_set.clear()
        self.unique_categories.clear()

    def set_goal_text(self, text: Optional[str]) -> None:
        self.goal_text = text
        self.goal_vector = None
        if text:
            encoder = self._ensure_encoder()
            if encoder is not None:
                vec = encoder.embed(text)
                self.goal_vector = vec / (np.linalg.norm(vec) + 1e-9)
        if self.goal_vector is None:
            for group in self.groups.values():
                group.v_score = 0.0
                group.h_score = 0.0
                for cluster in group.clusters.values():
                    cluster.v_score = 0.0
                    cluster.h_score = 0.0
        else:
            self._update_all_group_scores()

    def add_detections(
        self,
        detections: Sequence[Dict[str, Any]],
        active_ids: Optional[Iterable[int]] = None,
    ) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for det in detections:
            det_id = det.get("det_id")
            label = det.get("label") or det.get("category")
            pos = det.get("xy")
            status = det.get("status", "new")
            if det_id is None or label is None or pos is None:
                continue
            position = np.asarray(pos, dtype=np.float32)
            self.det_positions[det_id] = position
            self.det_labels[det_id] = label
            self._register_category(label)

            if status == "new":
                gid = self._assign_semantic_group(det_id, label)
                cid = self._assign_spatial_group(det_id, gid, position)
                summaries.append(
                    {
                        "det_id": det_id,
                        "semantic_group": gid,
                        "spatial_cluster": cid,
                        "category": label,
                    }
                )
            elif status in {"updated", "merged"}:
                self._update_existing(det_id, position)

        if active_ids is not None:
            self.prune_missing_detections(active_ids)

        return summaries

    def prune_missing_detections(self, active_ids: Iterable[int]) -> None:
        active = set(active_ids)
        stale = [det_id for det_id in self.det_to_group if det_id not in active]
        for det_id in stale:
            self._remove_detection(det_id)

    def get_groups_summary(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for gid, group in self.groups.items():
            group_info = {
                "semantic_group": gid,
                "size": len(group.members),
                "v_score": float(group.v_score),
                "h_score": float(group.h_score),
                "semantic_vector": group.semantic_vector.tolist(),
                "clusters": [],
            }
            for cluster in group.clusters.values():
                cluster_info = {
                    "cluster_id": cluster.cid,
                    "size": len(cluster.member_ids),
                    "centroid": cluster.centroid.tolist() if cluster.centroid is not None else None,
                    "v_score": float(cluster.v_score),
                    "h_score": float(cluster.h_score),
                    "members": list(cluster.member_ids),
                }
                group_info["clusters"].append(cluster_info)
            summary.append(group_info)
        return summary

    def get_unique_categories(self) -> List[str]:
        """Return a copy of the detected unique semantic categories."""
        return list(self.unique_categories)

    def select_goal_object(
        self,
        agent_position: Sequence[float],
        goal_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Use the stored detections (and LLM when available) to select an object goal.

        Args:
            agent_position: Agent location in [row, col] grid coordinates.
            goal_hint: Optional override for the desired goal text; defaults to self.goal_text.

        Returns:
            Dictionary with the selected object information or None when selection fails.
        """
        if not self.det_positions:
            logging.debug("[HierGrouper] No detections available for goal selection.")
            return None

        categories = self.get_unique_categories()
        if not categories:
            logging.debug("[HierGrouper] No semantic categories recorded for goal selection.")
            return None

        hint = (goal_hint or self.goal_text or "").strip()
        chosen_category = self._choose_category_with_llm(hint, categories)
        if chosen_category is None:
            chosen_category = self._fallback_category_selection(hint, categories)

        if chosen_category is None:
            logging.debug("[HierGrouper] Unable to determine a category for goal selection.")
            return None

        candidate_objects: List[Tuple[int, np.ndarray]] = []
        for det_id, label in self.det_labels.items():
            if label != chosen_category:
                continue
            det_pos = self.det_positions.get(det_id)
            if det_pos is None or det_pos.size < 2 or not np.all(np.isfinite(det_pos[:2])):
                continue
            candidate_objects.append((det_id, det_pos))

        if not candidate_objects:
            logging.debug(
                "[HierGrouper] No objects available for selected category '%s'.",
                chosen_category,
            )
            return None

        agent_xy = self._agent_position_to_xy(agent_position)
        # If agent position is unavailable, fall back to the first candidate
        if agent_xy is None or agent_xy.size < 2:
            chosen_det_id, chosen_pos = candidate_objects[0]
            distance = None
        elif len(candidate_objects) == 1:
            chosen_det_id, chosen_pos = candidate_objects[0]
            distance = float(np.linalg.norm(chosen_pos[:2] - agent_xy))
        else:
            # Choose the nearest instance of the selected category
            distances = [
                (float(np.linalg.norm(det_pos[:2] - agent_xy)), det_id, det_pos)
                for det_id, det_pos in candidate_objects
            ]
            distances.sort(key=lambda item: item[0])
            distance, chosen_det_id, chosen_pos = distances[0]

        map_y = int(round(float(chosen_pos[1])))
        map_x = int(round(float(chosen_pos[0])))
        goal_info = {
            "det_id": chosen_det_id,
            "category": chosen_category,
            "position": {"x": map_x, "y": map_y},
            "map_point": [map_y, map_x],
        }
        if distance is not None:
            goal_info["distance"] = distance

        logging.info(
            "[HierGrouper] Selected goal object det_id=%s category='%s' at map (%d, %d).",
            chosen_det_id,
            chosen_category,
            map_y,
            map_x,
        )
        return goal_info

    # --- internal helpers -----------------------------------------------
    def _register_category(self, label: Optional[str]) -> None:
        if not label:
            return
        if label not in self._unique_category_set:
            self._unique_category_set.add(label)
            self.unique_categories.append(label)

    def _refresh_unique_categories(self) -> None:
        current = set(self.det_labels.values())
        if current == self._unique_category_set:
            return
        self._unique_category_set = current
        retained = [label for label in self.unique_categories if label in current]
        if len(retained) != len(self.unique_categories):
            self.unique_categories = retained
        for label in current:
            if label not in self.unique_categories:
                self.unique_categories.append(label)

    def _choose_category_with_llm(
        self,
        goal_hint: Optional[str],
        categories: Sequence[str],
    ) -> Optional[str]:
        if self.llm_client is None or not categories:
            return None
        target_desc = goal_hint or self.goal_text or "target object"
        options_text = "\n".join(f"- {cat}" for cat in categories)
        prompt = (
            "You are assisting a navigation robot searching for a goal item inside a building.\n"
            f"Target description: {target_desc or 'unknown'}\n"
            "Detected semantic categories that may contain relevant objects:\n"
            f"{options_text}\n"
            "Select the single category that is most likely to contain the target object. "
            "Respond with exactly one category name from the list above."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You help select the best semantic category to inspect for a target object. "
                    "Always answer with exactly one category string from the provided list."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        try:
            completion = self.llm_client.create_chat_completion(
                "cogvlm2",
                messages,
                temperature=0.0,
                max_tokens=32,
                top_p=1.0,
                use_stream=False,
            )
        except Exception as exc:
            logging.warning("[HierGrouper] LLM category selection failed: %s", exc)
            return None

        if not completion:
            return None

        _, raw_response = completion
        response = (raw_response or "").strip()
        if not response:
            return None

        return self._parse_category_from_response(response, categories)

    def _parse_category_from_response(
        self,
        response: str,
        categories: Sequence[str],
    ) -> Optional[str]:
        normalized = response.strip().lower()
        if not normalized:
            return None

        for category in categories:
            if normalized == category.lower():
                return category

        for category in categories:
            if category.lower() in normalized:
                return category

        tokens = [tok.strip() for tok in re.split(r"[\n,;:\\|]", normalized) if tok.strip()]
        for token in tokens:
            for category in categories:
                if token == category.lower():
                    return category

        return None

    def _fallback_category_selection(
        self,
        goal_hint: Optional[str],
        categories: Sequence[str],
    ) -> Optional[str]:
        if not categories:
            return None
        if goal_hint:
            hint_lower = goal_hint.lower()
            for category in categories:
                if category.lower() == hint_lower:
                    return category
            for category in categories:
                if hint_lower in category.lower() or category.lower() in hint_lower:
                    return category
        return categories[0]

    @staticmethod
    def _agent_position_to_xy(agent_position: Sequence[float]) -> Optional[np.ndarray]:
        if agent_position is None:
            return None
        try:
            row = float(agent_position[0])
            col = float(agent_position[1])
        except (TypeError, ValueError, IndexError):
            return None
        return np.asarray([col, row], dtype=np.float32)

    def _ensure_encoder(self) -> Optional[Any]:
        if self.encoder is not None:
            return self.encoder
        if CLIPTextEncoder is None:
            return None
        try:
            self.encoder = CLIPTextEncoder(self._clip_model_name, device=self._clip_device)
        except Exception as exc:
            logging.warning("[HierGrouper] CLIP encoder initialisation failed: %s", exc)
            self.encoder = None
        return self.encoder

    def _maybe_embed_label(self, label: str) -> Optional[np.ndarray]:
        encoder = self._ensure_encoder()
        if encoder is None:
            return None
        try:
            vec = encoder.embed(label)
        except Exception as exc:
            logging.warning("[HierGrouper] CLIP embedding failed for '%s': %s", label, exc)
            return None
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.astype(np.float32)

    def _get_det_vector(self, det_id: int, label: str) -> Optional[np.ndarray]:
        vec = self.det_vectors.get(det_id)
        if vec is not None and np.isfinite(np.linalg.norm(vec)) and vec.ndim > 0:
            return vec
        vec = self._maybe_embed_label(label)
        if vec is not None:
            self.det_vectors[det_id] = vec
        return vec

    def _compute_cluster_centroid(self, cluster: SpatialCluster) -> Optional[np.ndarray]:
        embeds = []
        for det_id in cluster.member_ids:
            label = self.det_labels.get(det_id, "")
            emb = self._get_det_vector(det_id, label)
            if emb is not None:
                embeds.append(emb)
        if not embeds:
            return None
        mu = np.mean(np.stack(embeds, axis=0), axis=0)
        norm = float(np.linalg.norm(mu))
        if not math.isfinite(norm) or norm < 1e-9:
            return None
        return mu / norm

    def _update_group_scores(self, gid: int) -> None:
        group = self.groups.get(gid)
        if group is None:
            return
        if self.goal_vector is None or self.goal_vector.size == 0:
            group.v_score = 0.0
            group.h_score = 0.0
            return

        goal_vec = self.goal_vector
        sem_vec = group.semantic_vector
        if sem_vec is not None and sem_vec.size == goal_vec.size:
            cos_sim = float(np.dot(sem_vec, goal_vec) / (np.linalg.norm(sem_vec) * (np.linalg.norm(goal_vec) + 1e-9)))
            if math.isnan(cos_sim):
                cos_sim = 0.0
            group.v_score = (cos_sim + 1.0) * 0.5
        else:
            group.v_score = 0.0

        dists = []
        goal_norm = np.linalg.norm(goal_vec) + 1e-9
        for det_id in group.members:
            label = self.det_labels.get(det_id, "")
            emb = self._get_det_vector(det_id, label)
            if emb is None or emb.size != goal_vec.size:
                continue
            emb_norm = np.linalg.norm(emb) + 1e-9
            cos_sim = float(np.dot(emb, goal_vec) / (emb_norm * goal_norm))
            if math.isnan(cos_sim):
                continue
            dist = 1.0 - cos_sim
            dist = max(0.0, min(2.0, dist))
            dists.append(dist)
        group.h_score = float(np.var(dists)) if dists else 0.0

    def _update_cluster_scores(self, gid: int, cluster: SpatialCluster) -> None:
        if self.goal_vector is None or self.goal_vector.size == 0:
            cluster.v_score = 0.0
            cluster.h_score = 0.0
            return

        goal_vec = self.goal_vector
        centroid_vec = self._compute_cluster_centroid(cluster)
        if centroid_vec is not None and centroid_vec.size == goal_vec.size:
            cos_sim = float(np.dot(centroid_vec, goal_vec) / (np.linalg.norm(centroid_vec) * (np.linalg.norm(goal_vec) + 1e-9)))
            if math.isnan(cos_sim):
                cos_sim = 0.0
            cluster.v_score = (cos_sim + 1.0) * 0.5
        else:
            cluster.v_score = 0.0

        dists = []
        goal_norm = np.linalg.norm(goal_vec) + 1e-9
        for det_id in cluster.member_ids:
            label = self.det_labels.get(det_id, "")
            emb = self._get_det_vector(det_id, label)
            if emb is None or emb.size != goal_vec.size:
                continue
            emb_norm = np.linalg.norm(emb) + 1e-9
            cos_sim = float(np.dot(emb, goal_vec) / (emb_norm * goal_norm))
            if math.isnan(cos_sim):
                continue
            dist = 1.0 - cos_sim
            dist = max(0.0, min(2.0, dist))
            dists.append(dist)
        cluster.h_score = float(np.var(dists)) if dists else 0.0

    def _update_all_group_scores(self) -> None:
        for gid, group in self.groups.items():
            self._update_group_scores(gid)
            for cluster in group.clusters.values():
                self._update_cluster_scores(gid, cluster)

    def _embed(self, label: str) -> np.ndarray:
        encoder = self._ensure_encoder()
        if encoder is None:
            raise RuntimeError("CLIP encoder not initialised for clip-based grouping.")
        vec = encoder.embed(label)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.astype(np.float32)

    def _assign_semantic_group(self, det_id: int, label: str) -> int:
        if self.semantic_policy == "clip":
            vector = self._embed(label)
            self.det_vectors[det_id] = vector
            best_gid = None
            best_sim = -1.0
            similarity_rows = []
            for gid, group in self.groups.items():
                sim = _cosine(vector, group.semantic_vector)
                members = []
                for mid in group.members:
                    member_label = self.det_labels.get(mid, "?")
                    member_pos = self.det_positions.get(mid)
                    members.append((member_label, member_pos))
                similarity_rows.append((gid, sim, members))
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid
            if similarity_rows:
                header = f"Incoming det_id={det_id} label={label}"
                table_lines = [header, "gid | cosine_similarity | members(label@position)", "----|-------------------|--------------------------"]
                for gid, sim, members in similarity_rows:
                    member_strs = []
                    for m_label, pos in members:
                        if pos is None:
                            member_strs.append(f"{m_label}@(? , ?)")
                        else:
                            member_strs.append(f"{m_label}@({pos[0]:.1f},{pos[1]:.1f})")
                    member_col = ", ".join(member_strs) if member_strs else "--"
                    table_lines.append(f"{gid:>3} | {sim: .4f} | {member_col}")
                logging.debug("[HierGrouper] semantic similarity table\n%s", "\n".join(table_lines))
            goal_bonus = 0.0
            if self.goal_vector is not None:
                goal_bonus = _cosine(vector, self.goal_vector)
            if best_gid is not None and best_sim >= self.semantic_threshold:
                group = self.groups[best_gid]
                group.add_member(det_id, vector)
                self.det_to_group[det_id] = best_gid
                logging.debug(
                    "[HierGrouper] det_id=%s label=%s assigned to semantic group %d (sim=%.3f, goal_bonus=%.3f)",
                    det_id,
                    label,
                    best_gid,
                    best_sim,
                    goal_bonus,
                )
                self._update_group_scores(best_gid)
                return best_gid

            gid = self._create_semantic_group(vector)
            self.groups[gid].add_member(det_id, vector)
            self.det_to_group[det_id] = gid
            logging.debug(
                "[HierGrouper] det_id=%s label=%s created new semantic group %d",
                det_id,
                label,
                gid,
            )
            self._update_group_scores(gid)
            return gid

        if self.semantic_policy == "llm" and self.llm_client is not None:
            option_map: Dict[str, int] = {}

            def _letter_code(index: int) -> str:
                base = string.ascii_uppercase
                n = len(base)
                code: List[str] = []
                idx = index
                while True:
                    idx, rem = divmod(idx, n)
                    code.append(base[rem])
                    if idx == 0:
                        break
                    idx -= 1
                return "".join(reversed(code))

            option_lines: List[str] = []
            for opt_idx, (gid, group) in enumerate(self.groups.items()):
                option_label = _letter_code(opt_idx)
                option_map[option_label] = gid
                member_labels = [self._label_for_detection(mid) for mid in group.members]
                human_labels = ", ".join(lbl for lbl in member_labels if lbl)
                option_lines.append(f"{option_label}: {human_labels or 'empty'}")

            none_label = _letter_code(len(self.groups))
            option_map[none_label] = None
            option_lines.append(f"{none_label}: None of the above (create a new group)")

            prompt = (
                "Considering Indoor room layout and semantics, which of the following groups does the incoming object best belong to?\n"
                + "\n".join(option_lines)
                + f"\nIncoming object: {label}\nAnswer with a single letter and nothing else."
            )
            try:
                messages = [
                    {   
                            "role": "system",
                            "content": "You are a knowledgeable assistant to answer multiple choice questions by considering Indoor room layout. Always answer with a single letter and nothing else.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
                completion = self.llm_client.create_chat_completion(
                    "cogvlm2",
                    messages,
                    temperature=0.0,
                    max_tokens=16,
                    top_p=1.0,
                    use_stream=False,
                )
                if not completion:
                    raise ValueError("Empty LLM response")
                _, raw_response = completion
                response = (raw_response or "").strip()
                if not response:
                    raise ValueError("Blank LLM response content")
                token = response.split()[0].upper().rstrip(".,:;")

                if token in option_map:
                    mapped_gid = option_map[token]
                    if mapped_gid is None:
                        vec = self.det_vectors.get(det_id)
                        if vec is None and self.encoder is not None:
                            vec = self._embed(label)
                        if vec is None:
                            vec = np.zeros(1, dtype=np.float32)
                        gid = self._create_semantic_group(vec)
                        self.det_vectors[det_id] = vec
                        self.groups[gid].add_member(det_id, vec)
                        self.det_to_group[det_id] = gid
                        logging.debug(
                            "[HierGrouper] det_id=%s label=%s created semantic group %d via LLM choice %s",
                            det_id,
                            label,
                            gid,
                            token,
                        )
                        self._update_group_scores(gid)
                        return gid

                    gid = mapped_gid
                    vec = self.det_vectors.get(det_id)
                    if vec is None and self.encoder is not None:
                        vec = self._embed(label)
                        self.det_vectors[det_id] = vec
                    elif vec is None:
                        vec = np.zeros(1, dtype=np.float32)
                    self.groups[gid].add_member(det_id, vec)
                    self.det_to_group[det_id] = gid
                    logging.debug(
                        "[HierGrouper] det_id=%s label=%s assigned to semantic group %d via LLM choice %s",
                        det_id,
                        label,
                        gid,
                        token,
                    )
                    self._update_group_scores(gid)
                    return gid
                else:
                    logging.warning(
                        "[HierGrouper] Unexpected LLM response %r; defaulting to new group for det_id=%s label=%s",
                        response,
                        det_id,
                        label,
                    )
            except Exception as exc:
                logging.warning("LLM grouping fallback due to error: %s", exc)
            vector = np.zeros(1, dtype=np.float32)
            gid = self._create_semantic_group(vector)
            self.det_vectors[det_id] = vector
            self.groups[gid].add_member(det_id, vector)
            self.det_to_group[det_id] = gid
            logging.debug(
                "[HierGrouper] det_id=%s label=%s created semantic group %d (LLM fallback)",
                det_id,
                label,
                gid,
            )
            self._update_group_scores(gid)
            return gid

        # default fallback: create separate group
        vector = np.zeros(1, dtype=np.float32)
        self.det_vectors[det_id] = vector
        gid = self._create_semantic_group(vector)
        self.groups[gid].add_member(det_id, vector)
        self.det_to_group[det_id] = gid
        logging.debug(
            "[HierGrouper] det_id=%s label=%s created semantic group %d (no policy match)",
            det_id,
            label,
            gid,
        )
        return gid

    def _create_semantic_group(self, vector: np.ndarray) -> int:
        gid = self.next_gid
        self.next_gid += 1
        if vector.ndim == 1:
            norm = np.linalg.norm(vector)
            semantic_vec = vector / (norm + 1e-9)
        else:
            semantic_vec = vector
        self.groups[gid] = SemanticGroup(gid=gid, semantic_vector=semantic_vec)
        return gid

    def _assign_spatial_group(self, det_id: int, gid: int, position: np.ndarray) -> int:
        group = self.groups[gid]
        best_cluster: Optional[SpatialCluster] = None
        best_distance = float("inf")
        distance_rows = []
        for cluster in group.clusters.values():
            distance = cluster.closest_distance(position)
            member_strs = []
            for mid in cluster.member_ids:
                label = self.det_labels.get(mid, "?")
                pos = self.det_positions.get(mid)
                if pos is None:
                    member_strs.append(f"{label}@(? , ?)")
                else:
                    member_strs.append(f"{label}@({pos[0]:.1f},{pos[1]:.1f})")
            distance_rows.append(
                (
                    cluster.cid,
                    distance,
                    member_strs or ["--"],
                    cluster.centroid.copy() if cluster.centroid is not None else None,
                )
            )
            if distance < best_distance:
                best_distance = distance
                best_cluster = cluster

        if distance_rows:
            header = f"Spatial grouping candidate distances for det_id={det_id}"
            table_lines = [
                header,
                "cid | distance | centroid | members(label@position)",
                "----|----------|----------|-------------------------",
            ]
            for cid, dist, member_strs, centroid in distance_rows:
                centroid_str = (
                    f"({centroid[0]:.1f},{centroid[1]:.1f})" if centroid is not None else "--"
                )
                table_lines.append(
                    f"{cid:>3} | {dist: .3f} | {centroid_str} | {', '.join(member_strs)}"
                )
            logging.debug("[HierGrouper] spatial distance table\n%s", "\n".join(table_lines))

        if best_cluster is not None and best_distance <= self.spatial_threshold:
            best_cluster.add_member(det_id, position)
            self.det_to_cluster[det_id] = (gid, best_cluster.cid)
            self._update_cluster_scores(gid, best_cluster)
            self._update_group_scores(gid)
            logging.debug(
                "[HierGrouper] det_id=%s assigned to spatial cluster %d (gid=%d, dist=%.3f)",
                det_id,
                best_cluster.cid,
                gid,
                best_distance,
            )
            return best_cluster.cid

        cid = group.next_cluster_id
        group.next_cluster_id += 1
        new_cluster = SpatialCluster(cid=cid)
        new_cluster.add_member(det_id, position)
        group.clusters[cid] = new_cluster
        self.det_to_cluster[det_id] = (gid, cid)
        self._update_cluster_scores(gid, new_cluster)
        self._update_group_scores(gid)
        logging.debug(
            "[HierGrouper] det_id=%s created spatial cluster %d (gid=%d)",
            det_id,
            cid,
            gid,
        )
        return cid

    def _update_existing(self, det_id: int, position: np.ndarray) -> None:
        self.det_positions[det_id] = position
        mapping = self.det_to_cluster.get(det_id)
        if mapping is None:
            return
        gid, cid = mapping
        group = self.groups.get(gid)
        if group is None:
            return
        cluster = group.clusters.get(cid)
        if cluster is None:
            return
        cluster.update_member(det_id, position)
        self._update_cluster_scores(gid, cluster)
        logging.debug(
            "[HierGrouper] det_id=%s updated position in spatial cluster %d (gid=%d)",
            det_id,
            cid,
            gid,
        )

    def _remove_detection(self, det_id: int) -> None:
        mapping = self.det_to_cluster.pop(det_id, None)
        gid = self.det_to_group.pop(det_id, None)
        self.det_positions.pop(det_id, None)
        removed_label = self.det_labels.pop(det_id, None)
        if removed_label is not None:
            self._refresh_unique_categories()
        vector = self.det_vectors.pop(det_id, None)
        if gid is None:
            return
        group = self.groups.get(gid)
        if group is None:
            return
        group.remove_member(det_id, self.det_vectors if vector is None else {det_id: vector})
        if mapping:
            _, cid = mapping
            cluster = group.clusters.get(cid)
            if cluster is not None:
                cluster.remove_member(det_id)
                if not cluster.member_ids:
                    group.clusters.pop(cid, None)
                    logging.debug(
                        "[HierGrouper] removed empty spatial cluster %d (gid=%d)",
                        cid,
                        gid,
                    )
                else:
                    self._update_cluster_scores(gid, cluster)
        if not group.members:
            self.groups.pop(gid, None)
            logging.debug("[HierGrouper] removed empty semantic group %d", gid)
        else:
            self._update_group_scores(gid)

    def _label_for_detection(self, det_id: int) -> Optional[str]:
        # Returning None keeps prompt clean when label unknown
        return self.det_labels.get(det_id)
