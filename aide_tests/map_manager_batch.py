import numpy as np
import torch
from skimage import measure
from skimage.morphology import disk, binary_dilation
from collections import defaultdict


class MapManagerBatch:
    """
    Batch-optimized MapManager for faster object detection and tracking.
    Processes multiple objects simultaneously for better performance.
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
    
    def get_object_positions_batch(self, full_map, object_category, min_area=5):
        """
        Batch-optimized object detection and tracking.
        
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
        
        # Find changed regions
        changed_regions = self.find_changed_regions(full_map, category_list)
        if not changed_regions:
            self.prev_full_map = full_map.copy()
            return []
        
        # Batch process all changed regions
        object_positions = self._batch_process_regions(changed_regions, full_map, min_area)
        
        self.prev_full_map = full_map.copy()
        return object_positions

    def _batch_process_regions(self, changed_regions, full_map, min_area):
        """Batch process all changed regions for better performance."""
        object_positions = []
        
        # Group regions by category for batch processing
        regions_by_category = defaultdict(list)
        for ch, category_name, new_detections in changed_regions:
            regions_by_category[ch].append((category_name, new_detections))
        
        # Process each category in batch
        for ch, regions in regions_by_category.items():
            category_results = self._batch_process_category(ch, regions, full_map, min_area)
            object_positions.extend(category_results)
        
        return object_positions

    def _batch_process_category(self, ch, regions, full_map, min_area):
        """Batch process all regions for a specific category."""
        category_id = ch - 4
        category_name = regions[0][0]  # Get category name from first region
        
        # Collect all components from all regions for this category
        all_components = []
        all_prev_components = []
        all_roi_bounds = []
        all_roi_semantic = []
        
        for category_name, new_detections in regions:
            roi_bounds = self.get_roi_bounds(new_detections, full_map.shape[1:])
            if not roi_bounds:
                continue
                
            current_components, prev_components, roi_semantic = self.extract_components(
                full_map[ch], self.prev_full_map[ch], roi_bounds
            )
            
            # Filter by area and collect
            valid_components = [comp for comp in current_components if comp.area >= min_area]
            all_components.extend(valid_components)
            all_prev_components.extend(prev_components)
            all_roi_bounds.append(roi_bounds)
            all_roi_semantic.append(roi_semantic)
        
        if not all_components:
            return []
        
        # Batch create object info for all components
        object_infos = self._batch_create_object_info(
            all_components, category_name, category_id, all_roi_bounds, all_roi_semantic
        )
        
        # Batch find connected objects
        connected_objects_map = self._batch_find_connected_objects(
            all_components, all_prev_components, category_id, all_roi_bounds
        )
        
        # Batch process object connections and merging
        return self._batch_process_connections(object_infos, connected_objects_map)

    def _batch_create_object_info(self, components, category_name, category_id, roi_bounds_list, roi_semantic_list):
        """Batch create object information for multiple components."""
        object_infos = []
        
        for i, component in enumerate(components):
            # Use the ROI bounds for this component
            roi_bounds = roi_bounds_list[i] if i < len(roi_bounds_list) else roi_bounds_list[0]
            roi_semantic = roi_semantic_list[i] if i < len(roi_semantic_list) else roi_semantic_list[0]
            
            object_info = self.create_object_info(component, category_name, category_id, roi_bounds, roi_semantic)
            object_infos.append(object_info)
        
        return object_infos

    def _batch_find_connected_objects(self, components, prev_components, category_id, roi_bounds_list):
        """Batch find connected objects for multiple components."""
        connected_objects_map = {}
        
        for i, component in enumerate(components):
            roi_bounds = roi_bounds_list[i] if i < len(roi_bounds_list) else roi_bounds_list[0]
            connected_objects = self.find_connected_objects(component, prev_components, category_id, roi_bounds)
            connected_objects_map[i] = connected_objects
        
        return connected_objects_map

    def _batch_process_connections(self, object_infos, connected_objects_map):
        """Batch process object connections and merging."""
        object_positions = []
        processed_objects = set()
        
        # Group objects by their connections for batch processing
        connection_groups = self._group_connected_objects(connected_objects_map)
        
        for group in connection_groups:
            if len(group) == 1:
                # Single object - process normally
                idx = group[0]
                obj_info = object_infos[idx]
                connected_objects = connected_objects_map[idx]
                
                if connected_objects:
                    # Update existing object
                    connected_id = connected_objects[0]
                    self.update_tracked_object(connected_id, obj_info)
                    self.tracked_objects[connected_id]['object_state'] = 'updated'
                    object_positions.append(self.tracked_objects[connected_id])
                else:
                    # Create new object
                    new_id = self.create_new_object(obj_info)
                    self.tracked_objects[new_id]['object_state'] = 'new'
                    object_positions.append(self.tracked_objects[new_id])
                
                processed_objects.add(idx)
            else:
                # Multiple connected objects - batch merge
                merged_result = self._batch_merge_objects(group, object_infos, connected_objects_map)
                object_positions.append(merged_result)
                
                for idx in group:
                    processed_objects.add(idx)
        
        return object_positions

    def _group_connected_objects(self, connected_objects_map):
        """Group objects that are connected to each other."""
        groups = []
        processed = set()
        
        for idx, connected_objects in connected_objects_map.items():
            if idx in processed:
                continue
                
            if not connected_objects:
                # No connections - single object group
                groups.append([idx])
                processed.add(idx)
            else:
                # Find all objects connected to this one
                group = [idx]
                processed.add(idx)
                
                # Add all connected objects
                for connected_id in connected_objects:
                    # Find which object index corresponds to this connected_id
                    for other_idx, other_connected in connected_objects_map.items():
                        if other_idx not in processed and connected_id in other_connected:
                            group.append(other_idx)
                            processed.add(other_idx)
                
                groups.append(group)
        
        return groups

    def _batch_merge_objects(self, group, object_infos, connected_objects_map):
        """Batch merge multiple connected objects."""
        # Get all connected object IDs
        all_connected_ids = set()
        for idx in group:
            all_connected_ids.update(connected_objects_map[idx])
        
        if not all_connected_ids:
            # No existing connections - create new object
            primary_obj = object_infos[group[0]]
            new_id = self.create_new_object(primary_obj)
            self.tracked_objects[new_id]['object_state'] = 'new'
            return self.tracked_objects[new_id]
        
        # Merge with existing objects
        primary_id = list(all_connected_ids)[0]
        merged_info = self.merge_multiple_connected_objects(list(all_connected_ids), object_infos[group[0]])
        merged_info['object_id'] = primary_id
        merged_info['object_state'] = 'merged'
        
        # Remove other connected objects
        for old_id in list(all_connected_ids)[1:]:
            if old_id in self.tracked_objects:
                del self.tracked_objects[old_id]
        
        # Update primary object
        self.tracked_objects[primary_id] = merged_info
        return self.tracked_objects[primary_id]

    # Include all the original methods from MapManager
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
        roi_current = current_channel[roi_bounds['min_y']:roi_bounds['max_y'], 
                                    roi_bounds['min_x']:roi_bounds['max_x']]
        roi_prev = prev_channel[roi_bounds['min_y']:roi_bounds['max_y'], 
                                roi_bounds['min_x']:roi_bounds['max_x']]
        
        selem = disk(2)
        current_mask = binary_dilation((roi_current > 0.5).astype(np.uint8), selem)
        prev_mask = binary_dilation((roi_prev > 0.5).astype(np.uint8), selem)
        
        current_components = measure.regionprops(measure.label(current_mask, connectivity=2))
        prev_components = measure.regionprops(measure.label(prev_mask, connectivity=2))
        
        return current_components, prev_components, roi_current

    def create_object_info(self, component, category_name, category_id, roi_bounds, roi_semantic):
        """Create object information from component."""
        centroid_y = component.centroid[0] + roi_bounds['min_y']
        centroid_x = component.centroid[1] + roi_bounds['min_x']
        
        bbox = component.bbox
        component_coords = component.coords
        confidence = roi_semantic[component_coords[:, 0], component_coords[:, 1]].mean()
        
        return {
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

    def find_connected_objects(self, component, prev_components, category_id, roi_bounds):
        """Find all tracked objects that this component connects with."""
        connected_objects = []
        if not prev_components:
            return connected_objects
            
        for prev_component in prev_components:
            if self.components_overlap(component, prev_component):
                prev_centroid = (prev_component.centroid[0] + roi_bounds['min_y'],
                                prev_component.centroid[1] + roi_bounds['min_x'])
                
                for obj_id, tracked_obj in self.tracked_objects.items():
                    if (tracked_obj['category_id'] == category_id and
                        self._positions_close(tracked_obj['map_position'], prev_centroid)):
                        if obj_id not in connected_objects:
                            connected_objects.append(obj_id)
        
        return connected_objects

    def merge_multiple_connected_objects(self, connected_objects, new_object_info):
        """Merge multiple connected objects with a new component."""
        if not connected_objects:
            return new_object_info
        
        all_objects = [self.tracked_objects[obj_id] for obj_id in connected_objects if obj_id in self.tracked_objects]
        all_objects.append(new_object_info)
        
        if not all_objects:
            return new_object_info
        
        total_area = sum(obj['area'] for obj in all_objects)
        weighted_x = sum(obj['map_position']['x'] * obj['area'] for obj in all_objects) / total_area
        weighted_y = sum(obj['map_position']['y'] * obj['area'] for obj in all_objects) / total_area
        
        all_bboxes = [obj['bounding_box'] for obj in all_objects]
        min_x = min(bbox['min_x'] for bbox in all_bboxes)
        min_y = min(bbox['min_y'] for bbox in all_bboxes)
        max_x = max(bbox['max_x'] for bbox in all_bboxes)
        max_y = max(bbox['max_y'] for bbox in all_bboxes)
        
        weighted_confidence = sum(obj['confidence'] * obj['area'] for obj in all_objects) / total_area
        latest_step = max(obj['step'] for obj in all_objects)
        
        return {
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
            'merged_from': connected_objects + ['new_component']
        }

    def components_overlap(self, comp1, comp2):
        """Check if two components overlap."""
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

    def create_new_object(self, object_info):
        """Create new tracked object."""
        object_id = self.next_object_id
        self.next_object_id += 1
        object_info['object_id'] = object_id
        self.tracked_objects[object_id] = object_info
        return object_id

    def get_newly_added_objects(self, object_positions):
        """Get only newly added objects from object_positions."""
        return [obj for obj in object_positions if obj.get('object_state') == 'new']

    def get_updated_objects(self, object_positions):
        """Get only updated objects from object_positions."""
        return [obj for obj in object_positions if obj.get('object_state') == 'updated']

    def get_merged_objects(self, object_positions):
        """Get only merged objects from object_positions."""
        return [obj for obj in object_positions if obj.get('object_state') == 'merged']

    def clear_object_states(self):
        """Clear object state information after visualization."""
        for obj_id, obj in self.tracked_objects.items():
            if 'object_state' in obj:
                del obj['object_state']







