"""
AIDE Agent State Adapter

This module provides the interface methods required for AIDE integration.
These methods serve as the only integration points and won't break existing pipelines.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np


class AgentStateAdapter:
    """
    Adapter class that provides AIDE interface methods for AgentState.
    This isolates AIDE-specific functionality from the main agent state.
    """
    
    def __init__(self, agent_state: Dict[str, Any]):
        """
        Initialize the adapter with an agent state dictionary.
        
        Args:
            agent_state: Dictionary containing agent state information including
                        map_state, frontier_nodes, history_nodes, etc.
        """
        self.agent_state = agent_state
        self.map_state = agent_state.get('map_state')
        self.frontier_nodes = agent_state.get('frontier_nodes', [])
        self.history_nodes = agent_state.get('history_nodes', [])
        self.pose = agent_state.get('pose', {})
        self.last_goal = agent_state.get('last_goal')
        self.top_candidates = agent_state.get('top_candidates', [])
        self.groups = agent_state.get('groups', [])
        self.group_directory = agent_state.get('group_directory', set())
        self.global_graph = agent_state.get('global_graph')
        self.group_index = agent_state.get('group_index', {'counter': 0})
    
    def encode_tokens_for_aide(self, goal_text: str) -> List[str]:
        """
        Convert perception/mapping/history/frontier data into token list.
        Can reuse existing semantic_mapping extraction.
        
        Args:
            goal_text: The goal text to encode
            
        Returns:
            List of tokens representing the agent's state
        """
        tokens = []
        
        # Extract object information from map_state
        if self.map_state is not None:
            # Handle different map_state formats
            if hasattr(self.map_state, 'objects'):
                # If map_state has objects attribute
                for obj in self.map_state.objects:
                    room = getattr(obj, 'room', 'unknown')
                    label = getattr(obj, 'label', 'unknown')
                    tokens.append(f"obj:{label}@{room}")
            elif isinstance(self.map_state, np.ndarray):
                # If map_state is a numpy array (semantic map)
                # Extract semantic information from the map
                if len(self.map_state.shape) >= 3:
                    # Multi-channel semantic map
                    for ch in range(min(self.map_state.shape[0], 10)):  # Limit channels
                        channel_data = self.map_state[ch]
                        if hasattr(channel_data, 'max'):
                            max_val = channel_data.max()
                            if max_val > 0.1:  # Threshold filter
                                tokens.append(f"semantic_ch{ch}:{max_val:.2f}")
        
        # Add frontier node information
        for i, fr in enumerate(self.frontier_nodes[:32]):  # Limit to 32 nodes
            if isinstance(fr, (list, tuple)) and len(fr) >= 2:
                # Simple coordinate format [x, y]
                x, y = fr[0], fr[1]
                tokens.append(f"frontier:{x:.1f}:{y:.1f}")
            elif hasattr(fr, 'room') and hasattr(fr, 'area'):
                # Object with room and area attributes
                tokens.append(f"frontier:{fr.room}:{int(fr.area)}")
            else:
                # Fallback for unknown format
                tokens.append(f"frontier:unknown:{i}")
        
        # Add history node information
        for i, hi in enumerate(self.history_nodes[:32]):  # Limit to 32 nodes
            if isinstance(hi, (list, tuple)) and len(hi) >= 2:
                # Simple coordinate format [x, y]
                x, y = hi[0], hi[1]
                tokens.append(f"history:{x:.1f}:{y:.1f}")
            elif hasattr(hi, 'room') and hasattr(hi, 'visits'):
                # Object with room and visits attributes
                tokens.append(f"history:{hi.room}:{int(hi.visits)}")
            else:
                # Fallback for unknown format
                tokens.append(f"history:unknown:{i}")
        
        # Add goal information
        if goal_text:
            tokens.append(f"goal:{goal_text}")
        
        return tokens
    
    def candidate_features(self, topk: int, selected_groups: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert candidate nodes to structured features.
        Must include at least id, room, dist, free, ig, and aggregated v, h, cost from selected groups.
        
        Args:
            topk: Number of top candidates to return
            selected_groups: List of selected groups for aggregation
            
        Returns:
            List of feature dictionaries for each candidate
        """
        features = []
        
        # Get candidate list (combine frontier and history nodes)
        candidate_list = self.top_candidates_raw(topk)
        
        def hit(group, candidate):
            """
            Check if group is related to candidate based on room/semantic proximity.
            Can be customized based on specific requirements.
            """
            if not hasattr(group, 'tokens'):
                return False
            
            # Check if candidate room type matches any group tokens
            candidate_room = getattr(candidate, 'room_type', 'unknown').lower()
            return any(candidate_room in token.lower() for token in group.tokens)
        
        for candidate in candidate_list:
            # Calculate aggregated values from selected groups
            v_sum = sum(getattr(g, 'v_hat', 0.0) for g in selected_groups if hit(g, candidate))
            h_sum = sum(getattr(g, 'h_hat', 0.0) for g in selected_groups if hit(g, candidate))
            c_sum = sum(getattr(g, 'c_hat', 0.0) for g in selected_groups if hit(g, candidate))
            
            # Extract candidate features
            candidate_id = getattr(candidate, 'id', id(candidate))
            room_type = getattr(candidate, 'room_type', 'unknown')
            distance = getattr(candidate, 'distance', 0.0)
            is_reachable = getattr(candidate, 'is_reachable', True)
            info_gain = getattr(candidate, 'info_gain', 0.0)
            seen_objects = getattr(candidate, 'seen_objs', [])
            
            # Handle coordinate-based candidates
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                candidate_id = len(features)  # Use index as ID
                room_type = 'coordinate'
                distance = np.sqrt(candidate[0]**2 + candidate[1]**2)  # Distance from origin
                is_reachable = True
                info_gain = 0.0
                seen_objects = []
            
            features.append({
                "id": int(candidate_id),
                "room": room_type,
                "dist": round(distance, 2),
                "free": 1 if is_reachable else 0,
                "ig": round(info_gain, 3),
                "v": round(v_sum, 3),
                "h": round(h_sum, 3),
                "cost": int(c_sum),
                "seen": list(seen_objects)[:3]  # Limit to 3 objects
            })
        
        return features
    
    def top_candidates_raw(self, topk: int) -> List[Any]:
        """
        Get raw list of top candidates (frontier/history node objects).
        This method should be implemented based on existing candidate selection logic.
        
        Args:
            topk: Number of top candidates to return
            
        Returns:
            List of candidate node objects
        """
        # Combine frontier and history nodes
        all_candidates = []
        
        # Add frontier nodes
        for i, node in enumerate(self.frontier_nodes[:topk//2]):
            # Create a simple object-like structure for coordinate-based nodes
            if isinstance(node, (list, tuple)) and len(node) >= 2:
                candidate = type('Candidate', (), {
                    'id': i,
                    'room_type': 'frontier',
                    'distance': np.sqrt(node[0]**2 + node[1]**2),
                    'is_reachable': True,
                    'info_gain': 1.0,  # Default value
                    'seen_objs': []
                })()
                all_candidates.append(candidate)
            else:
                all_candidates.append(node)
        
        # Add history nodes
        for i, node in enumerate(self.history_nodes[:topk//2]):
            if isinstance(node, (list, tuple)) and len(node) >= 2:
                candidate = type('Candidate', (), {
                    'id': i + len(self.frontier_nodes),
                    'room_type': 'history',
                    'distance': np.sqrt(node[0]**2 + node[1]**2),
                    'is_reachable': True,
                    'info_gain': 0.5,  # Lower value for history nodes
                    'seen_objs': []
                })()
                all_candidates.append(candidate)
            else:
                all_candidates.append(node)
        
        return all_candidates[:topk]
    
    def node_from_choice(self, cid: int) -> Any:
        """
        Convert returned id to map node/subgoal object.
        
        Args:
            cid: Choice ID returned by AIDE
            
        Returns:
            Map node or subgoal object
        """
        return self.lookup_node(cid)
    
    def lookup_node(self, node_id: int) -> Any:
        """
        Look up a node by ID from the available candidates.
        
        Args:
            node_id: Node ID to look up
            
        Returns:
            Node object or coordinate tuple
        """
        # Try to find in frontier nodes first
        if node_id < len(self.frontier_nodes):
            return self.frontier_nodes[node_id]
        
        # Then try history nodes
        history_id = node_id - len(self.frontier_nodes)
        if history_id < len(self.history_nodes):
            return self.history_nodes[history_id]
        
        # Fallback to default
        return [0, 0]
    
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
        # If needed, map summary back to map directory layer (no need to load specific KV)


def create_agent_state_adapter(agent_state: Dict[str, Any]) -> AgentStateAdapter:
    """
    Factory function to create an AgentStateAdapter from agent state dictionary.
    
    Args:
        agent_state: Dictionary containing agent state information
        
    Returns:
        AgentStateAdapter instance
    """
    return AgentStateAdapter(agent_state)


# Example usage and integration with existing code
def integrate_with_main_loop(agent_state_dict: Dict[str, Any], goal_text: str) -> Dict[str, Any]:
    """
    Example integration function showing how to use the adapter with existing main loop.
    
    Args:
        agent_state_dict: Agent state dictionary from main loop
        goal_text: Goal text for encoding
        
    Returns:
        Enhanced agent state with AIDE methods
    """
    adapter = create_agent_state_adapter(agent_state_dict)
    
    # Add AIDE methods to the agent state
    agent_state_dict['encode_tokens_for_aide'] = adapter.encode_tokens_for_aide
    agent_state_dict['candidate_features'] = adapter.candidate_features
    agent_state_dict['node_from_choice'] = adapter.node_from_choice
    agent_state_dict['has_group_signature'] = adapter.has_group_signature
    agent_state_dict['add_group_from_summary'] = adapter.add_group_from_summary
    
    return agent_state_dict
