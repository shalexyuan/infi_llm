import logging
import numpy as np
import torch
from skimage import measure
from skimage.morphology import disk, binary_dilation


class MapManager:
    """
    Manages object detection and tracking from semantic maps.
    Handles object detection, tracking, merging, and cleanup operations.
    """
    
    def __init__(self, args=None, device='cuda'):
        """
        Initialize the MapManager.
        
        Args:
            args: Configuration arguments object
            device: Device to use for tensor operations
        """
        self.args = args
        self.device = device
        self.full_map = None
        self.prev_full_map = None
        self.tracked_objects = {}
        self.next_object_id = 0
        self.l_step = 0

    @staticmethod
    def _ensure_position(info):
        """Ensure legacy 'position' key mirrors 'map_position'."""
        if not isinstance(info, dict):
            return info
        map_pos = info.get('map_position')
        if isinstance(map_pos, dict):
            x = map_pos.get('x')
            y = map_pos.get('y')
            if x is not None and y is not None:
                info['position'] = {'x': int(x), 'y': int(y)}
        return info
    
    def get_object_positions(self, full_map, object_category, min_area=5):
        """
        Extract positions of newly detected objects from the full semantic map.
        Only processes areas that have changed since the last step.
        
        Args:
            full_map: torch.Tensor of shape [channels, height, width]
            object_category: List of semantic category names
            min_area: Minimum area threshold for object detection
        
        Returns:
            object_positions: List of dictionaries containing object information
        """
        # Convert to numpy and initialize tracking
        full_map = full_map.cpu().numpy() if isinstance(full_map, torch.Tensor) else full_map
        self.full_map = full_map
        self._init_tracking(full_map)
        
        # Update step counter
        self.l_step += 1
        
        # Determine the correct category list based on task configuration
        # IMPORTANT: For HM3D, we use MP3D naming convention (via coco_categories_hm3d2mp3d mapping)
        # to maintain consistency with agent goal naming
        if self.args and hasattr(self.args, 'task_config'):
            if 'objectnav_hm3d' in self.args.task_config:
                # For HM3D, the semantic map channels are scrambled by coco_categories
                # coco_categories = [0, 3, 2, 4, 5, 1] means:
                #   HM3D index i → semantic channel (coco_categories[i] + 4)
                # Map manager does: category_id = channel - 4
                # So we need category_list[channel_offset] to give the right name
                
                from constants import hm3d_category, coco_categories_hm3d2mp3d, coco_categories
                
                # Create inverse mapping: channel_offset → HM3D index
                channel_to_hm3d = {}
                for hm3d_idx in range(len(coco_categories)):
                    channel_offset = coco_categories[hm3d_idx]
                    channel_to_hm3d[channel_offset] = hm3d_idx
                
                # Build category_list arranged by channel offset
                category_list = []
                for channel_offset in range(len(hm3d_category)):
                    if channel_offset in channel_to_hm3d:
                        # This is one of the 6 goal categories
                        hm3d_idx = channel_to_hm3d[channel_offset]
                        mp3d_idx = coco_categories_hm3d2mp3d[hm3d_idx]
                        category_list.append(object_category[mp3d_idx])
                    else:
                        # Non-goal category: channels 6+ map sequentially to HM3D indices 6+
                        hm3d_idx = channel_offset
                        category_list.append(hm3d_category[hm3d_idx])
                
            elif 'objectnav_mp3d' in self.args.task_config:
                # For MP3D, use object_category (20 categories)
                category_list = object_category
            else:
                # Fallback to the passed object_category
                category_list = object_category
        else:
            # Fallback to the passed object_category
            category_list = object_category
        
        # Find changed regions
        changed_regions = self.find_changed_regions(full_map, category_list)
        if not changed_regions:
            self.prev_full_map = full_map.copy()
            return []
        
        # Process each changed region
        object_positions = []
        for ch, category_name, new_detections in changed_regions:
            roi_bounds = self.get_roi_bounds(new_detections, full_map.shape[1:])
            if not roi_bounds:
                continue
                
            current_components, prev_components, roi_semantic = self.extract_components(
                full_map[ch], self.prev_full_map[ch], roi_bounds
            )
            
            for component in current_components:
                if component.area < min_area:
                    continue
                    
                category_id = ch - 4  # Ensure category_id is non-negative
                object_info = self.create_object_info(component, category_name, category_id, roi_bounds, roi_semantic)
                connected_objects = self.find_connected_objects(component, prev_components, category_id, roi_bounds)
                
                if connected_objects:
                    # Handle multiple connected objects
                    if len(connected_objects) == 1:
                        # Single connection - just update and keep the same ID
                        connected_id = connected_objects[0]
                        self.update_tracked_object(connected_id, object_info)
                        # Mark as existing/updated object
                        self.tracked_objects[connected_id]['object_state'] = 'updated'
                        object_positions.append(self._ensure_position(self.tracked_objects[connected_id]))
                    else:
                        # Multiple connections - merge all connected objects
                        # Keep the ID of the first connected object
                        primary_id = connected_objects[0]
                        merged_info = self.merge_multiple_connected_objects(connected_objects, object_info)
                        merged_info['object_id'] = primary_id
                        merged_info['object_state'] = 'merged'  # Mark as merged
                        
                        # Remove the other connected objects (keep the primary one)
                        for old_id in connected_objects[1:]:  # Skip the first one (primary)
                            if old_id in self.tracked_objects:
                                del self.tracked_objects[old_id]
                        
                        # Update the primary object with merged info
                        self.tracked_objects[primary_id] = self._ensure_position(merged_info)
                        object_positions.append(self.tracked_objects[primary_id])
                else:
                    # No connections - create new object
                    connected_id = self.create_new_object(object_info)
                    # Mark as new object
                    self.tracked_objects[connected_id]['object_state'] = 'new'
                    object_positions.append(self._ensure_position(self.tracked_objects[connected_id]))
        
        self.prev_full_map = full_map.copy()
        return object_positions

    def _init_tracking(self, full_map):
        """Initialize tracking variables if not exists."""
        if not hasattr(self, 'prev_full_map') or self.prev_full_map is None:
            self.prev_full_map = np.zeros_like(full_map)
        if not hasattr(self, 'tracked_objects'):
            self.tracked_objects = {}
            self.next_object_id = 0

    def find_changed_regions(self, full_map, category_list):
        """Find regions with new detections for each semantic category."""
        changed_regions = []
        # Ensure we don't exceed the number of available channels or categories
        max_channels = min(full_map.shape[0], 4 + len(category_list))
        for ch in range(4, max_channels):
            category_idx = ch - 4
            if category_idx < len(category_list):
                category_name = category_list[category_idx]
                new_detections = (self.prev_full_map[ch] <= 0.5) & (full_map[ch] > 0.5)
                if np.any(new_detections):
                    changed_regions.append((ch, category_name, new_detections))
        return changed_regions

    def get_roi_bounds(self, new_detections, map_shape):
        """Get region of interest bounds with adaptive padding."""
        coords = np.where(new_detections)
        if len(coords[0]) == 0:
            return None
            
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        
        region_size = max(max_y - min_y, max_x - min_x)
        padding = max(20, int(region_size * 0.2))
        
        height, width = map_shape
        return {
            'min_y': max(0, min_y - padding),
            'max_y': min(height, max_y + padding),
            'min_x': max(0, min_x - padding),
            'max_x': min(width, max_x + padding)
        }

    def extract_components(self, current_channel, prev_channel, roi_bounds):
        """Extract connected components from ROI."""
        # Extract ROI
        roi_current = current_channel[roi_bounds['min_y']:roi_bounds['max_y'], 
                                    roi_bounds['min_x']:roi_bounds['max_x']]
        roi_prev = prev_channel[roi_bounds['min_y']:roi_bounds['max_y'], 
                                roi_bounds['min_x']:roi_bounds['max_x']]
        
        # Create binary masks and apply morphology
        selem = disk(2)
        current_mask = binary_dilation((roi_current > 0.5).astype(np.uint8), selem)
        prev_mask = binary_dilation((roi_prev > 0.5).astype(np.uint8), selem)
        
        # Find components
        current_components = measure.regionprops(measure.label(current_mask, connectivity=2))
        prev_components = measure.regionprops(measure.label(prev_mask, connectivity=2))
        
        return current_components, prev_components, roi_current

    def create_object_info(self, component, category_name, category_id, roi_bounds, roi_semantic):
        """Create object information from component."""
        # Adjust coordinates for ROI offset
        centroid_y = component.centroid[0] + roi_bounds['min_y']
        centroid_x = component.centroid[1] + roi_bounds['min_x']
        
        bbox = component.bbox
        
        # Calculate confidence from the semantic channel values
        component_coords = component.coords
        confidence = roi_semantic[component_coords[:, 0], component_coords[:, 1]].mean()
        
        info = {
            'category': category_name,
            'category_id': category_id,
            'map_position': {'x': int(centroid_x), 'y': int(centroid_y)},
            'bounding_box': {
                'min_x': int(bbox[1] + roi_bounds['min_x']),
                'min_y': int(bbox[0] + roi_bounds['min_y']),
                'max_x': int(bbox[3] + roi_bounds['min_x']),
                'max_y': int(bbox[2] + roi_bounds['min_y'])
            },
            'area': int(component.area),
            'confidence': float(confidence),
            'step': self.l_step
        }
        return self._ensure_position(info)

    def find_connected_objects(self, component, prev_components, category_id, roi_bounds):
        """Find all tracked objects that this component connects with."""
        connected_objects = []
        if not prev_components:
            return connected_objects
            
        # Check overlap with all previous components
        for prev_component in prev_components:
            if self.components_overlap(component, prev_component):
                # Find matching tracked object
                prev_centroid = (prev_component.centroid[0] + roi_bounds['min_y'],
                                prev_component.centroid[1] + roi_bounds['min_x'])
                
                for obj_id, tracked_obj in self.tracked_objects.items():
                    if (tracked_obj['category_id'] == category_id and
                        self._positions_close(tracked_obj['map_position'], prev_centroid)):
                        if obj_id not in connected_objects:  # Avoid duplicates
                            connected_objects.append(obj_id)
        
        return connected_objects

    def merge_multiple_connected_objects(self, connected_objects, new_object_info):
        """Merge multiple connected objects with a new component."""
        if not connected_objects:
            return new_object_info

        # Get all objects to merge (including the new one)
        all_objects = [self.tracked_objects[obj_id] for obj_id in connected_objects if obj_id in self.tracked_objects]
        all_objects.append(new_object_info)
        
        if not all_objects:
            return new_object_info
        
        # Calculate merged properties
        total_area = sum(obj['area'] for obj in all_objects)
        
        # Calculate weighted centroid
        weighted_x = sum(obj['map_position']['x'] * obj['area'] for obj in all_objects) / total_area
        weighted_y = sum(obj['map_position']['y'] * obj['area'] for obj in all_objects) / total_area
        
        # Calculate merged bounding box
        all_bboxes = [obj['bounding_box'] for obj in all_objects]
        min_x = min(bbox['min_x'] for bbox in all_bboxes)
        min_y = min(bbox['min_y'] for bbox in all_bboxes)
        max_x = max(bbox['max_x'] for bbox in all_bboxes)
        max_y = max(bbox['max_y'] for bbox in all_bboxes)
        
        # Calculate weighted confidence
        weighted_confidence = sum(obj['confidence'] * obj['area'] for obj in all_objects) / total_area
        
        # Use the most recent step
        latest_step = max(obj['step'] for obj in all_objects)
        
        info = {
            'category': new_object_info['category'],
            'category_id': new_object_info['category_id'],
            'map_position': {'x': int(weighted_x), 'y': int(weighted_y)},
            'bounding_box': {
                'min_x': int(min_x),
                'min_y': int(min_y),
                'max_x': int(max_x),
                'max_y': int(max_y)
            },
            'area': int(total_area),
            'confidence': float(weighted_confidence),
            'step': latest_step,
            'merged_from': connected_objects + ['new_component']  # Track which objects were merged
        }
        logging.info(
            "MapManager: merged objects %s with new component -> category=%s position=%s",
            connected_objects,
            info.get('category'),
            info.get('map_position'),
        )
        return self._ensure_position(info)

    def components_overlap(self, comp1, comp2):
        """Check if two components overlap."""
        # Simple overlap check based on bounding boxes
        bbox1, bbox2 = comp1.bbox, comp2.bbox
        return not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or 
                    bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])

    def _positions_close(self, pos1, pos2, threshold=20):
        """Check if two positions are close enough."""
        return (abs(pos1['x'] - pos2[1]) < threshold and 
                abs(pos1['y'] - pos2[0]) < threshold)

    def update_tracked_object(self, object_id, object_info):
        """Update existing tracked object."""
        self.tracked_objects[object_id].update(object_info)
        self._ensure_position(self.tracked_objects[object_id])

    def cleanup_semantic_map(self):
        """Clean up semantic map to ensure only one semantic category per pixel."""
        # Convert to numpy if needed
        full_map_np = self.full_map.cpu().numpy() if isinstance(self.full_map, torch.Tensor) else self.full_map.numpy()
        
        # Get semantic channels (channels 4 onwards)
        args_num = getattr(self.args, "num_sem_categories", None) if self.args else None
        total_sem_channels = max(0, full_map_np.shape[0] - 4)
        if args_num is not None:
            num_sem = min(args_num, total_sem_channels)
        else:
            num_sem = total_sem_channels
        semantic_channels = full_map_np[4:4 + num_sem, :, :]
        
        # Create a mask for pixels with multiple semantic detections
        semantic_sum = np.sum(semantic_channels, axis=0)
        multi_semantic_mask = semantic_sum > 1.0
        if np.any(multi_semantic_mask):
            print(f"  Found {np.sum(multi_semantic_mask)} pixels with multiple semantic detections")
            
            # Set all semantic values to 0 for pixels with multiple semantics (vectorized)
            semantic_channels[:, multi_semantic_mask] = 0.0
            
            # Update the full map with cleaned semantic channels
            self.full_map[4:4 + num_sem, :, :] = torch.from_numpy(semantic_channels).to(self.device)
            
            print(f"  Cleaned up semantic map - removed {np.sum(multi_semantic_mask)} ambiguous pixels")

    def cleanup_tracked_objects(self, full_map, object_category):
        """Clean up tracked objects every 10 steps by checking if they still exist and merging nearby objects."""
        if not hasattr(self, 'tracked_objects') or not self.tracked_objects:
            return
        
        print(f"\n=== Cleaning up tracked objects (Step {self.l_step}) ===")
        print(f"Before cleanup: {len(self.tracked_objects)} tracked objects")
        
        # Convert to numpy if needed
        full_map = full_map.cpu().numpy() if isinstance(full_map, torch.Tensor) else full_map
        
        # Determine the correct category list based on task configuration
        if self.args and hasattr(self.args, 'task_config'):
            if 'objectnav_hm3d' in self.args.task_config:
                from constants import hm3d_category
                category_list = hm3d_category
            elif 'objectnav_mp3d' in self.args.task_config:
                category_list = object_category
            else:
                category_list = object_category
        else:
            category_list = object_category
        
        # Check each tracked object to see if it still exists
        objects_to_remove = []
        objects_to_merge = {}
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            category_id = tracked_obj['category_id']
            map_pos = tracked_obj['map_position']
            
            # Check if the object still exists in the current map
            if not self._object_still_exists(full_map, category_id, map_pos, category_list):
                objects_to_remove.append(obj_id)
                print(f"  Removing object {obj_id} ({tracked_obj['category']}) - no longer detected")
            else:
                # Check if this object should be merged with nearby objects
                merge_candidates = self._find_merge_candidates(obj_id, tracked_obj, self.tracked_objects)
                if merge_candidates:
                    objects_to_merge[obj_id] = merge_candidates
        
        # Remove objects that no longer exist
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
        
        # Merge nearby objects
        merged_objects = set()
        for obj_id, candidates in objects_to_merge.items():
            if obj_id in merged_objects:
                continue
                
            # Find all objects that should be merged together
            merge_group = [obj_id] + [c for c in candidates if c not in merged_objects]
            
            if len(merge_group) > 1:
                # Merge the group - keep the ID of the first object
                primary_id = merge_group[0]
                merged_obj = self._merge_tracked_objects(merge_group)
                merged_obj['object_id'] = primary_id
                
                # Remove the other objects (keep the primary one)
                for old_obj_id in merge_group[1:]:  # Skip the first one (primary)
                    if old_obj_id in self.tracked_objects:
                        del self.tracked_objects[old_obj_id]
                    merged_objects.add(old_obj_id)
                
                # Update the primary object with merged info
                self.tracked_objects[primary_id] = self._ensure_position(merged_obj)
                merged_objects.add(primary_id)
                
                print(f"  Merged {len(merge_group)} objects into object {primary_id}")
        
        print(f"After cleanup: {len(self.tracked_objects)} tracked objects")

    def _object_still_exists(self, full_map, category_id, map_pos, category_list):
        """Check if a tracked object still exists in the current map."""
        if category_id >= len(category_list):
            return False
        
        # Get the semantic channel for this category
        channel_idx = 4 + category_id
        if channel_idx >= full_map.shape[0]:
            return False
        
        semantic_channel = full_map[channel_idx]
        
        # Check a small region around the object's position
        x, y = map_pos['x'], map_pos['y']
        check_radius = 10
        
        # Get bounds for checking
        min_x = max(0, x - check_radius)
        max_x = min(semantic_channel.shape[1], x + check_radius)
        min_y = max(0, y - check_radius)
        max_y = min(semantic_channel.shape[0], y + check_radius)
        
        # Check if there's still semantic content in this region
        region = semantic_channel[min_y:max_y, min_x:max_x]
        return np.any(region > 0.5)

    def _find_merge_candidates(self, obj_id, tracked_obj, all_tracked_objects):
        """Find other tracked objects that should be merged with this one."""
        candidates = []
        obj_pos = tracked_obj['map_position']
        obj_category = tracked_obj['category_id']
        
        for other_id, other_obj in all_tracked_objects.items():
            if other_id == obj_id or other_obj['category_id'] != obj_category:
                continue
            
            other_pos = other_obj['map_position']
            
            # Calculate distance between objects
            distance = np.sqrt((obj_pos['x'] - other_pos['x'])**2 + (obj_pos['y'] - other_pos['y'])**2)
            
            # If objects are very close, they should be merged
            if distance < 15:  # Threshold for merging
                candidates.append(other_id)
        
        return candidates

    def _merge_tracked_objects(self, object_ids):
        """Merge multiple tracked objects into one."""
        if not object_ids:
            return None
        
        # Get all objects to merge
        objects_to_merge = [self.tracked_objects[obj_id] for obj_id in object_ids if obj_id in self.tracked_objects]
        
        if not objects_to_merge:
            return None
        
        # Calculate merged properties
        total_area = sum(obj['area'] for obj in objects_to_merge)
        
        # Calculate weighted centroid
        weighted_x = sum(obj['map_position']['x'] * obj['area'] for obj in objects_to_merge) / total_area
        weighted_y = sum(obj['map_position']['y'] * obj['area'] for obj in objects_to_merge) / total_area
        
        # Calculate merged bounding box
        all_bboxes = [obj['bounding_box'] for obj in objects_to_merge]
        min_x = min(bbox['min_x'] for bbox in all_bboxes)
        min_y = min(bbox['min_y'] for bbox in all_bboxes)
        max_x = max(bbox['max_x'] for bbox in all_bboxes)
        max_y = max(bbox['max_y'] for bbox in all_bboxes)
        
        # Calculate weighted confidence
        weighted_confidence = sum(obj['confidence'] * obj['area'] for obj in objects_to_merge) / total_area
        
        # Use the most recent step
        latest_step = max(obj['step'] for obj in objects_to_merge)
        
        info = {
            'category': objects_to_merge[0]['category'],
            'category_id': objects_to_merge[0]['category_id'],
            'map_position': {'x': int(weighted_x), 'y': int(weighted_y)},
            'bounding_box': {
                'min_x': int(min_x),
                'min_y': int(min_y),
                'max_x': int(max_x),
                'max_y': int(max_y)
            },
            'area': int(total_area),
            'confidence': float(weighted_confidence),
            'step': latest_step,
            'merged_from': object_ids  # Track which objects were merged
        }
        return self._ensure_position(info)

    def create_new_object(self, object_info):
        """Create new tracked object."""
        object_id = self.next_object_id
        self.next_object_id += 1
        object_info['object_id'] = object_id
        self.tracked_objects[object_id] = self._ensure_position(object_info)
        logging.info(
            "MapManager: activated new object_id=%d category=%s position=%s step=%s",
            object_id,
            object_info.get('category'),
            object_info.get('map_position'),
            object_info.get('step'),
        )
        return object_id

    def _create_new_object(self, object_info):
        """Create new tracked object (internal method)."""
        return self.create_new_object(object_info)

    def print_object_summary(self, object_positions):
        """
        Print a summary of detected objects.
        
        Args:
            object_positions: List of object position dictionaries
        """
        if not object_positions:
            print("No objects detected in this step.")
            return
        
        print(f"\n=== Object Detection Summary (Step {self.l_step}) ===")
        print(f"Total objects detected: {len(object_positions)}")
        
        # Group by category
        category_counts = {}
        for obj in object_positions:
            category = obj['category']
            if category not in category_counts:
                category_counts[category] = []
            category_counts[category].append(obj)
        
        for category, objects in category_counts.items():
            print(f"\n{category}: {len(objects)} objects")
            for i, obj in enumerate(objects):
                print(f"  {i+1}. Map Position: ({obj['map_position']['x']}, {obj['map_position']['y']}), "
                        f"Confidence: {obj['confidence']:.3f}, Area: {obj['area']} pixels")

    def update_step(self, step):
        """Update the current step number."""
        self.l_step = step

    def reset_tracking(self):
        """Reset all tracking data."""
        self.full_map = None
        self.prev_full_map = None
        self.tracked_objects = {}
        self.next_object_id = 0
        self.l_step = 0

    def clear_object_states(self):
        """Clear object state information after visualization."""
        for obj_id, obj in self.tracked_objects.items():
            if 'object_state' in obj:
                del obj['object_state']

    def get_newly_added_objects(self, object_positions):
        """Get only newly added objects from object_positions."""
        return [obj for obj in object_positions if obj.get('object_state') == 'new']

    def get_updated_objects(self, object_positions):
        """Get only updated objects from object_positions."""
        return [obj for obj in object_positions if obj.get('object_state') == 'updated']

    def get_merged_objects(self, object_positions):
        """Get only merged objects from object_positions."""
        return [obj for obj in object_positions if obj.get('object_state') == 'merged']
