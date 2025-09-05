"""
AIDE Agent State Adapter for LLM_Agent_GT

This module provides the interface methods required for AIDE integration
specifically for the LLM_Agent_GT class. It handles the GT agent's specific
data structures including local_map, target_point_map, and coordinate systems.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch


class AgentStateAdapterGT:
    """
    Adapter class that provides AIDE interface methods for LLM_Agent_GT.
    This isolates AIDE-specific functionality from the GT agent implementation.
    """
    
    def __init__(self, agent: Any):
        """
        Initialize the adapter with an LLM_Agent_GT instance.
        
        Args:
            agent: LLM_Agent_GT instance containing agent state information
        """
        self.agent = agent
        self.local_map = getattr(agent, 'local_map', None)
        self.target_point_map = getattr(agent, 'target_point_map', None)
        self.target_edge_map = getattr(agent, 'target_edge_map', None)
        self.goal_id = getattr(agent, 'goal_id', None)
        self.curr_loc = getattr(agent, 'curr_loc', [0, 0])
        self.planner_pose_inputs = getattr(agent, 'planner_pose_inputs', np.zeros(7))
        self.visited = getattr(agent, 'visited', None)
        self.visited_vis = getattr(agent, 'visited_vis', None)
        self.l_step = getattr(agent, 'l_step', 0)
        self.episode_n = getattr(agent, 'episode_n', 0)
        
        # GT-specific attributes
        self.hm3d_semantic_mapping = getattr(agent, 'hm3d_semantic_mapping', {})
        self.object_category = getattr(agent, 'object_category', [])
        self.local_w = getattr(agent, 'local_w', 480)
        self.local_h = getattr(agent, 'local_h', 480)
        
        # Initialize group management
        self.groups = []
        self.group_directory = set()
        self.global_graph = None
        self.group_index = {'counter': 0}
    
    def encode_tokens_for_aide(self, goal_text: str) -> List[str]:
        """
        Convert GT agent's perception/mapping/history/frontier data into token list.
        Uses local_map semantic channels and target_point_map for frontier information.
        
        Args:
            goal_text: The goal text to encode
            
        Returns:
            List of tokens representing the GT agent's state
        """
        tokens = []
        
        # Extract semantic information from local_map
        if self.local_map is not None and hasattr(self.local_map, 'shape'):
            # local_map format: [channels, height, width]
            # Channel 0: obstacle map, Channel 1: explored map, Channel 2-3: current location
            # Channel 4+: semantic categories
            
            # Add map dimensions and explored area
            explored_area = torch.sum(self.local_map[1, :, :] > 0).item()
            total_area = self.local_map.shape[1] * self.local_map.shape[2]
            exploration_ratio = explored_area / total_area if total_area > 0 else 0
            tokens.append(f"explored:{exploration_ratio:.3f}")
            
            # Extract semantic categories (channels 4+)
            if self.local_map.shape[0] > 4:
                semantic_channels = self.local_map[4:, :, :]
                for ch in range(min(semantic_channels.shape[0], 20)):  # Limit channels
                    channel_data = semantic_channels[ch]
                    if torch.sum(channel_data > 0).item() > 0:
                        max_val = torch.max(channel_data).item()
                        tokens.append(f"sem_ch{ch}:{max_val:.2f}")
            
            # Add current location information
            if len(self.curr_loc) >= 2:
                tokens.append(f"curr_loc:{self.curr_loc[0]:.1f}:{self.curr_loc[1]:.1f}")
        
        # Add target point map information (frontier-like)
        if self.target_point_map is not None:
            # target_point_map is typically [local_w, local_h] with frontier points
            frontier_points = np.where(self.target_point_map > 0)
            if len(frontier_points[0]) > 0:
                tokens.append(f"frontier_count:{len(frontier_points[0])}")
                
                # Add spatial distribution of frontier points
                if len(frontier_points[0]) > 0:
                    x_coords = frontier_points[1]  # Note: x and y might be swapped
                    y_coords = frontier_points[0]
                    x_center = np.mean(x_coords)
                    y_center = np.mean(y_coords)
                    tokens.append(f"frontier_center:{x_center:.1f}:{y_center:.1f}")
        
        # Add visited area information (history-like)
        if self.visited is not None:
            visited_points = np.where(self.visited > 0)
            if len(visited_points[0]) > 0:
                tokens.append(f"visited_count:{len(visited_points[0])}")
                
                # Add recent visit information
                if hasattr(self, 'visited_vis') and self.visited_vis is not None:
                    recent_visits = np.where(self.visited_vis > 0)
                    if len(recent_visits[0]) > 0:
                        tokens.append(f"recent_visits:{len(recent_visits[0])}")
        
        # Add goal information
        if goal_text:
            tokens.append(f"goal:{goal_text}")
        
        # Add goal_id if available
        if self.goal_id is not None:
            tokens.append(f"goal_id:{self.goal_id}")
            
            # Add semantic mapping if available
            if hasattr(self, 'hm3d_semantic_mapping') and self.hm3d_semantic_mapping:
                goal_category = self.hm3d_semantic_mapping.get(str(self.goal_id), 'unknown')
                tokens.append(f"goal_category:{goal_category}")
        
        # Add step and episode information
        tokens.append(f"step:{self.l_step}")
        tokens.append(f"episode:{self.episode_n}")
        
        return tokens
    
    def candidate_features(self, topk: int, selected_groups: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert GT agent's candidate nodes to structured features.
        Uses target_point_map and goal_points for candidate generation.
        
        Args:
            topk: Number of top candidates to return
            selected_groups: List of selected groups for aggregation
            
        Returns:
            List of feature dictionaries for each candidate
        """
        features = []
        
        # Get candidate list from target_point_map
        candidate_list = self.top_candidates_raw_gt(topk)
        
        def hit(group, candidate):
            """
            Check if group is related to candidate based on semantic proximity.
            For GT agent, we use coordinate-based matching.
            """
            if not hasattr(group, 'tokens'):
                return False
            
            # Check if candidate coordinates match any group tokens
            candidate_coords = f"{candidate['x']:.1f}:{candidate['y']:.1f}"
            return any(candidate_coords in token for token in group.tokens)
        
        for i, candidate in enumerate(candidate_list):
            # Calculate aggregated values from selected groups
            v_sum = sum(getattr(g, 'v_hat', 0.0) for g in selected_groups if hit(g, candidate))
            h_sum = sum(getattr(g, 'h_hat', 0.0) for g in selected_groups if hit(g, candidate))
            c_sum = sum(getattr(g, 'c_hat', 0.0) for g in selected_groups if hit(g, candidate))
            
            # Calculate distance from current location
            curr_x, curr_y = self.curr_loc[0], self.curr_loc[1]
            distance = np.sqrt((candidate['x'] - curr_x)**2 + (candidate['y'] - curr_y)**2)
            
            # Determine if candidate is reachable (not in obstacle)
            is_reachable = self.is_reachable_gt(candidate['x'], candidate['y'])
            
            # Calculate information gain (based on unexplored area around candidate)
            info_gain = self.calculate_info_gain_gt(candidate['x'], candidate['y'])
            
            features.append({
                "id": int(candidate['id']),
                "room": candidate.get('room_type', 'frontier'),
                "dist": round(distance, 2),
                "free": 1 if is_reachable else 0,
                "ig": round(info_gain, 3),
                "v": round(v_sum, 3),
                "h": round(h_sum, 3),
                "cost": int(c_sum),
                "seen": candidate.get('seen_objs', [])[:3]
            })
        
        return features
    
    def top_candidates_raw_gt(self, topk: int) -> List[Dict[str, Any]]:
        """
        Get raw list of top candidates from GT agent's target_point_map.
        
        Args:
            topk: Number of top candidates to return
            
        Returns:
            List of candidate dictionaries with coordinates and metadata
        """
        candidates = []
        
        if self.target_point_map is not None:
            # Find all frontier points in target_point_map
            frontier_points = np.where(self.target_point_map > 0)
            
            if len(frontier_points[0]) > 0:
                # Convert map coordinates to world coordinates
                for i in range(min(len(frontier_points[0]), topk)):
                    map_y, map_x = frontier_points[0][i], frontier_points[1][i]
                    
                    # Convert to world coordinates (adjust based on your coordinate system)
                    world_x = (map_x - self.local_w // 2) * 0.05  # Assuming 5cm resolution
                    world_y = (map_y - self.local_h // 2) * 0.05
                    
                    candidate = {
                        'id': i,
                        'x': world_x,
                        'y': world_y,
                        'map_x': map_x,
                        'map_y': map_y,
                        'room_type': 'frontier',
                        'seen_objs': []
                    }
                    candidates.append(candidate)
        
        # If not enough candidates from target_point_map, add some from visited areas
        if len(candidates) < topk and self.visited is not None:
            visited_points = np.where(self.visited > 0)
            if len(visited_points[0]) > 0:
                # Add some visited points as history candidates
                for i in range(min(len(visited_points[0]), topk - len(candidates))):
                    map_y, map_x = visited_points[0][i], visited_points[1][i]
                    
                    world_x = (map_x - self.local_w // 2) * 0.05
                    world_y = (map_y - self.local_h // 2) * 0.05
                    
                    candidate = {
                        'id': len(candidates) + i,
                        'x': world_x,
                        'y': world_y,
                        'map_x': map_x,
                        'map_y': map_y,
                        'room_type': 'history',
                        'seen_objs': []
                    }
                    candidates.append(candidate)
        
        return candidates[:topk]
    
    def is_reachable_gt(self, x: float, y: float) -> bool:
        """
        Check if a coordinate is reachable (not in obstacle) for GT agent.
        
        Args:
            x, y: World coordinates
            
        Returns:
            True if reachable, False otherwise
        """
        if self.local_map is None:
            return True
        
        # Convert world coordinates to map coordinates
        map_x = int(x / 0.05 + self.local_w // 2)
        map_y = int(y / 0.05 + self.local_h // 2)
        
        # Check bounds
        if map_x < 0 or map_x >= self.local_w or map_y < 0 or map_y >= self.local_h:
            return False
        
        # Check if location is in obstacle (channel 0)
        if self.local_map[0, map_y, map_x] > 0:
            return False
        
        return True
    
    def calculate_info_gain_gt(self, x: float, y: float, radius: int = 10) -> float:
        """
        Calculate information gain for a coordinate based on unexplored area.
        
        Args:
            x, y: World coordinates
            radius: Radius around point to check for unexplored area
            
        Returns:
            Information gain value
        """
        if self.local_map is None:
            return 0.0
        
        # Convert world coordinates to map coordinates
        map_x = int(x / 0.05 + self.local_w // 2)
        map_y = int(y / 0.05 + self.local_h // 2)
        
        # Check bounds
        if map_x < 0 or map_x >= self.local_w or map_y < 0 or map_y >= self.local_h:
            return 0.0
        
        # Calculate unexplored area in radius around point
        y_start = max(0, map_y - radius)
        y_end = min(self.local_h, map_y + radius + 1)
        x_start = max(0, map_x - radius)
        x_end = min(self.local_w, map_x + radius + 1)
        
        # Get explored area (channel 1)
        explored_area = self.local_map[1, y_start:y_end, x_start:x_end]
        total_area = (y_end - y_start) * (x_end - x_start)
        
        if total_area == 0:
            return 0.0
        
        # Information gain is proportional to unexplored area
        unexplored_ratio = 1.0 - (torch.sum(explored_area > 0).item() / total_area)
        return unexplored_ratio
    
    def node_from_choice(self, cid: int) -> Dict[str, Any]:
        """
        Convert returned id to map node/subgoal object for GT agent.
        
        Args:
            cid: Choice ID returned by AIDE
            
        Returns:
            Map node or subgoal object
        """
        return self.lookup_node_gt(cid)
    
    def lookup_node_gt(self, node_id: int) -> Dict[str, Any]:
        """
        Look up a node by ID from the available candidates for GT agent.
        
        Args:
            node_id: Node ID to look up
            
        Returns:
            Node dictionary with coordinates
        """
        candidates = self.top_candidates_raw_gt(100)  # Get enough candidates
        
        if node_id < len(candidates):
            return candidates[node_id]
        
        # Fallback to current location
        return {
            'x': self.curr_loc[0],
            'y': self.curr_loc[1],
            'id': node_id,
            'room_type': 'current'
        }
    
    def has_group_signature(self, sig: bytes) -> bool:
        """
        Check if group signature exists in directory.
        Optional method for cross-robot summary deduplication.
        
        Args:
            sig: Group signature to check
            
        Returns:
            True if signature exists, False otherwise
        """
        return sig in self.group_directory
    
    def add_group_from_summary(self, group_summary: Any):
        """
        Add group from summary to directory.
        Optional method for cross-robot summary management.
        
        Args:
            group_summary: Group summary object with signature
        """
        if hasattr(group_summary, 'sig'):
            self.group_directory.add(group_summary.sig)


def create_agent_state_adapter_gt(agent: Any) -> AgentStateAdapterGT:
    """
    Factory function to create an AgentStateAdapterGT from LLM_Agent_GT instance.
    
    Args:
        agent: LLM_Agent_GT instance
        
    Returns:
        AgentStateAdapterGT instance
    """
    return AgentStateAdapterGT(agent)


# Example usage and integration with existing GT agent code
def integrate_with_gt_agent(agent: Any, goal_text: str) -> Dict[str, Any]:
    """
    Example integration function showing how to use the GT adapter.
    
    Args:
        agent: LLM_Agent_GT instance
        goal_text: Goal text for encoding
        
    Returns:
        Enhanced agent state with AIDE methods
    """
    adapter = create_agent_state_adapter_gt(agent)
    
    # Create agent state dictionary with AIDE methods
    agent_state = {
        'agent_id': getattr(agent, 'agent_id', 0),
        'map_state': getattr(agent, 'local_map', None),
        'frontier_nodes': adapter.top_candidates_raw_gt(32),
        'history_nodes': [],  # Can be populated from visited areas
        'pose': {
            'x': getattr(agent, 'curr_loc', [0, 0])[0],
            'y': getattr(agent, 'curr_loc', [0, 0])[1],
            'theta': 0.0  # GT agent doesn't track orientation explicitly
        },
        'last_goal': None,
        'top_candidates': adapter.top_candidates_raw_gt(6),
        'groups': adapter.groups,
        'group_directory': adapter.group_directory,
        'global_graph': adapter.global_graph,
        'group_index': adapter.group_index,
        
        # Add AIDE methods
        'encode_tokens_for_aide': adapter.encode_tokens_for_aide,
        'candidate_features': adapter.candidate_features,
        'node_from_choice': adapter.node_from_choice,
        'has_group_signature': adapter.has_group_signature,
        'add_group_from_summary': adapter.add_group_from_summary
    }
    
    return agent_state
