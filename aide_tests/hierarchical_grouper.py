
import logging
import math
import string
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any
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

    def set_goal_text(self, text: Optional[str]) -> None:
        self.goal_text = text
        self.goal_vector = None
        if text and self.encoder is not None:
            vec = self.encoder.embed(text)
            self.goal_vector = vec / (np.linalg.norm(vec) + 1e-9)

    def add_detections(
        self,
        detections: Sequence[Dict[str, Any]],
        active_ids: Optional[Iterable[int]] = None,
    ) -> List[Dict[str, Any]]:
        phase_start = time.perf_counter()
        semantic_acc = 0.0
        spatial_acc = 0.0
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

            if status == "new":
                semantic_start = time.perf_counter()
                gid = self._assign_semantic_group(det_id, label)
                semantic_acc += time.perf_counter() - semantic_start
                spatial_start = time.perf_counter()
                cid = self._assign_spatial_group(det_id, gid, position)
                spatial_acc += time.perf_counter() - spatial_start
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
        total_duration = time.perf_counter() - phase_start
        logging.debug(
            "[HierGrouper] add_detections timing: total=%.3f ms (semantic=%.3f ms, spatial=%.3f ms, detections=%d)",
            total_duration * 1000.0,
            semantic_acc * 1000.0,
            spatial_acc * 1000.0,
            len(detections),
        )
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
                "semantic_vector": group.semantic_vector.tolist(),
                "clusters": [],
            }
            for cluster in group.clusters.values():
                cluster_info = {
                    "cluster_id": cluster.cid,
                    "size": len(cluster.member_ids),
                    "centroid": cluster.centroid.tolist() if cluster.centroid is not None else None,
                    "members": list(cluster.member_ids),
                }
                group_info["clusters"].append(cluster_info)
            summary.append(group_info)
        return summary

    # --- internal helpers -----------------------------------------------
    def _embed(self, label: str) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError("CLIP encoder not initialised for clip-based grouping.")
        vec = self.encoder.embed(label)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec.astype(np.float32)

    def _assign_semantic_group(self, det_id: int, label: str) -> int:
        if self.semantic_policy == "clip":
            total_start = time.perf_counter()
            embed_start = time.perf_counter()
            vector = self._embed(label)
            embed_time = time.perf_counter() - embed_start
            self.det_vectors[det_id] = vector
            best_gid = None
            best_sim = -1.0
            similarity_rows = []
            sim_start = time.perf_counter()
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
            sim_time = time.perf_counter() - sim_start
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

            def _log_clip_timing(context: str) -> None:
                total_elapsed = time.perf_counter() - total_start
                logging.debug(
                    "[HierGrouper] timing semantic clip (%s): total=%.3f ms (embed=%.3f ms, similarity=%.3f ms)",
                    context,
                    total_elapsed * 1000.0,
                    embed_time * 1000.0,
                    sim_time * 1000.0,
                )
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
                _log_clip_timing("reuse")
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
            _log_clip_timing("new_group")
            return gid

        if self.semantic_policy == "llm" and self.llm_client is not None:
            total_start = time.perf_counter()
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
                f"Considering Indoor room layout and semantics, which of the following groups does the {label} best belong to?\n"
                + "\n".join(option_lines)
                + "\nAnswer with a single letter and nothing else."
            )
            logging.debug("[HierGrouper] LLM prompt:\n%s", prompt)
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
            prepare_time = time.perf_counter() - total_start
            llm_call_time = 0.0
            parse_time = 0.0
            fallback_reason: Optional[str] = None
            try:
                llm_call_start = time.perf_counter()
                completion = self.llm_client.create_chat_completion(
                    "cogvlm2",
                    messages,
                    temperature=0.0,
                    max_tokens=16,
                    top_p=1.0,
                    use_stream=False,
                )
                llm_call_time = time.perf_counter() - llm_call_start
                if not completion:
                    raise ValueError("Empty LLM response")
                _, raw_response = completion
                parse_start = time.perf_counter()
                response = (raw_response or "").strip()
                logging.debug("[HierGrouper] LLM raw response: %s", raw_response)
                if not response:
                    raise ValueError("Blank LLM response content")
                token = response.split()[0].upper().rstrip(".,:;")
                parse_time = time.perf_counter() - parse_start

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
                        total_elapsed = time.perf_counter() - total_start
                        logging.debug(
                            "[HierGrouper] timing semantic llm (new_group): total=%.3f ms (prepare=%.3f ms, llm=%.3f ms, parse=%.3f ms)",
                            total_elapsed * 1000.0,
                            prepare_time * 1000.0,
                            llm_call_time * 1000.0,
                            parse_time * 1000.0,
                        )
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
                    total_elapsed = time.perf_counter() - total_start
                    logging.debug(
                        "[HierGrouper] timing semantic llm (reuse): total=%.3f ms (prepare=%.3f ms, llm=%.3f ms, parse=%.3f ms)",
                        total_elapsed * 1000.0,
                        prepare_time * 1000.0,
                        llm_call_time * 1000.0,
                        parse_time * 1000.0,
                    )
                    fallback_reason = "unexpected_token"
                    total_elapsed = time.perf_counter() - total_start
                    logging.debug(
                        "[HierGrouper] timing semantic llm (unexpected token): total=%.3f ms (prepare=%.3f ms, llm=%.3f ms, parse=%.3f ms)",
                        total_elapsed * 1000.0,
                        prepare_time * 1000.0,
                        llm_call_time * 1000.0,
                        parse_time * 1000.0,
                    )
                    return gid
                else:
                    logging.warning(
                        "[HierGrouper] Unexpected LLM response %r; defaulting to new group for det_id=%s label=%s",
                        response,
                        det_id,
                        label,
                    )
            except Exception as exc:
                fallback_reason = "exception"
                total_elapsed = time.perf_counter() - total_start
                logging.debug(
                    "[HierGrouper] timing semantic llm (error): total=%.3f ms (prepare=%.3f ms, llm=%.3f ms, parse=%.3f ms)",
                    total_elapsed * 1000.0,
                    prepare_time * 1000.0,
                    llm_call_time * 1000.0,
                    parse_time * 1000.0,
                )
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
            total_elapsed = time.perf_counter() - total_start
            logging.debug(
                "[HierGrouper] timing semantic llm (fallback_new_group%s): total=%.3f ms (prepare=%.3f ms, llm=%.3f ms, parse=%.3f ms)",
                "" if fallback_reason is None else f"_{fallback_reason}",
                total_elapsed * 1000.0,
                prepare_time * 1000.0,
                llm_call_time * 1000.0,
                parse_time * 1000.0,
            )
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
        total_start = time.perf_counter()
        group = self.groups[gid]
        best_cluster: Optional[SpatialCluster] = None
        best_distance = float("inf")
        distance_rows = []
        search_start = time.perf_counter()
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

        distance_time = time.perf_counter() - search_start
        assign_start = time.perf_counter()
        if best_cluster is not None and best_distance <= self.spatial_threshold:
            best_cluster.add_member(det_id, position)
            self.det_to_cluster[det_id] = (gid, best_cluster.cid)
            assign_time = time.perf_counter() - assign_start
            total_elapsed = time.perf_counter() - total_start
            logging.debug(
                "[HierGrouper] det_id=%s assigned to spatial cluster %d (gid=%d, dist=%.3f)",
                det_id,
                best_cluster.cid,
                gid,
                best_distance,
            )
            logging.debug(
                "[HierGrouper] timing spatial (reuse): total=%.3f ms (search=%.3f ms, assign=%.3f ms)",
                total_elapsed * 1000.0,
                distance_time * 1000.0,
                assign_time * 1000.0,
            )
            return best_cluster.cid

        cid = group.next_cluster_id
        group.next_cluster_id += 1
        new_cluster = SpatialCluster(cid=cid)
        new_cluster.add_member(det_id, position)
        group.clusters[cid] = new_cluster
        self.det_to_cluster[det_id] = (gid, cid)
        assign_time = time.perf_counter() - assign_start
        total_elapsed = time.perf_counter() - total_start
        logging.debug(
            "[HierGrouper] det_id=%s created spatial cluster %d (gid=%d)",
            det_id,
            cid,
            gid,
        )
        logging.debug(
            "[HierGrouper] timing spatial (new_cluster): total=%.3f ms (search=%.3f ms, assign=%.3f ms)",
            total_elapsed * 1000.0,
            distance_time * 1000.0,
            assign_time * 1000.0,
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
        self.det_labels.pop(det_id, None)
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
        if not group.members:
            self.groups.pop(gid, None)
            logging.debug("[HierGrouper] removed empty semantic group %d", gid)

    def _label_for_detection(self, det_id: int) -> Optional[str]:
        # Returning None keeps prompt clean when label unknown
        return self.det_labels.get(det_id)
