import warnings
# Suppress PyTorch grid_sample/affine_grid warnings
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed")

from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
import os
import logging
import time
import json
import sys
import gym
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import quaternion
import pickle
import io
import re
import pdb
import random
from skimage import measure
import skimage.morphology
from PIL import Image

import math
import cv2
import habitat  # pyright: ignore[reportMissingImports]
import habitat_sim
from habitat.sims.habitat_simulator.actions import (  # pyright: ignore[reportMissingImports]
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)

from agents.vlm_agents import LLM_Agent
from agents.vlm_agents_gt import LLM_Agent_GT
# from agents.llm_agents import LLM_Agent
from constants import (
    color_palette, coco_categories, coco_categories_hm3d2mp3d,
    hm3d_category, category_to_id, object_category
)
from envs.habitat.multi_agent_env_vlm import Multi_Agent_Env
# from envs.habitat.multi_agent_env import Multi_Agent_Env

# from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import CogVLM2
from src.SystemPrompt import (
    form_prompt_for_PerceptionVLM, 
    form_prompt_for_FN,
    form_prompt_for_DecisionVLM_Frontier,
    Perception_weight_decision,
    Perception_weight_decision4,
)

import utils.pose as pu
import utils.depth_utils as du
import utils.visualization as vu

from arguments import get_args

# from detect_yolov9 import Detect
from ultralytics import YOLO
from aide_tests.map_manager import MapManager
from aide_tests.semantic_spatial_grouper import SemanticSpatialGrouper, GroupingCfg, Det
from aide_tests.price_coord import PriceCoordinator
from aide_tests.planner_step import one_step_assign

def alpha_label(index: int, lowercase: bool = False) -> str:
    """Return a base-26 alphabetic label for any non-negative index.
    0->A, 25->Z, 26->AA, 27->AB. Use lowercase=True for a..z.
    Prevents IndexError when more than 26 labels are needed.
    """
    if index < 0:
        return ""
    base = 26
    label = []
    while True:
        index, rem = divmod(index, base)
        base_char = ord('a') if lowercase else ord('A')
        label.append(chr(base_char + rem))
        if index == 0:
            break
        index -= 1
    return "".join(reversed(label))

def agent_near_gt_target(agent, args, radius_m: Optional[float] = None) -> bool:
    """Check if an agent-reported success aligns with its GT semantic map."""
    if not getattr(agent, "Find_Goal", False):
        return False
    goal_id = getattr(agent, "goal_id", None)
    if goal_id is None:
        return False
    if not getattr(agent, "use_gtsem", False):
        return False
    gt_offset = getattr(agent, "gt_channel_offset", None)
    num_classes = getattr(agent, "num_semantic_classes", None)
    if gt_offset is None or num_classes is None:
        return False

    full_map = getattr(agent, "full_map", None)
    if full_map is None or not isinstance(full_map, torch.Tensor):
        return False
    if full_map.numel() == 0:
        return False

    cn=coco_categories[goal_id] + 4
    semantic_idx = cn

    if semantic_idx < 0 or semantic_idx >= num_classes:
        return False
    target_channel = gt_offset + semantic_idx
    pdb.set_trace()
    if target_channel >= full_map.shape[0]:
        return False

    gt_channel = full_map[target_channel].detach().cpu().numpy()
    if np.max(gt_channel) <= 0:
        return False

    curr_loc = getattr(agent, "curr_loc", None)
    if not curr_loc or len(curr_loc) < 2:
        return False

    map_scale = 100.0 / args.map_resolution
    row_idx = int(curr_loc[1] * map_scale)
    col_idx = int(curr_loc[0] * map_scale)
    h, w = gt_channel.shape
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)

    radius_m = radius_m if radius_m is not None else getattr(args, "success_dist", 1.0)
    radius_cells = max(0, int(np.ceil(radius_m * map_scale)))
    r0 = max(0, row_idx - radius_cells)
    r1 = min(h, row_idx + radius_cells + 1)
    c0 = max(0, col_idx - radius_cells)
    c1 = min(w, col_idx + radius_cells + 1)
    neighborhood = gt_channel[r0:r1, c0:c1]
    return bool(np.any(neighborhood > 0.1))

@habitat.registry.register_action_space_configuration
class PreciseTurn(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.TURN_LEFT_S] = habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )
        config[HabitatSimActions.TURN_RIGHT_S] = habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S),
        )

        return config


def Objects_Extract(args, full_map_pred, use_sam):

    semantic_map = full_map_pred[4:4 + args.num_sem_categories]

    dst = np.zeros(semantic_map[0, :, :].shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    Object_list = {}
    for i in range(len(semantic_map)):
        if semantic_map[i, :, :].sum() != 0:
            Single_object_list = []
            se_object_map = semantic_map[i, :, :].cpu().numpy()
            se_object_map[se_object_map>0.1] = 1
            se_object_map = cv2.morphologyEx(se_object_map, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(cv2.inRange(se_object_map,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if len(cnt) > 30:
                    epsilon = 0.05 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    Single_object_list.append(approx)
                    cv2.polylines(dst, [approx], True, 1)
            if len(Single_object_list) > 0:
                if use_sam:
                    Object_list[object_category[i]] = Single_object_list
                else:
                    if 'objectnav_mp3d' in args.task_config:
                        Object_list[object_category[i]] = Single_object_list
                    elif 'objectnav_hm3d' in args.task_config:
                        Object_list[hm3d_category[i]] = Single_object_list
    return Object_list

def all_agents_exit_false(agents):
    for agent in agents:
        if agent.EXIT:
            return False
    return True

def all_agents_exit_true(agents):
    for agent in agents:
        if not agent.EXIT:
            return False
    return True

def ExtractExplorableAreas(full_map_pred, explo_area_map, VLM_PR, VLM_PR_last, color_map, count):
    PR = VLM_PR[0]

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    full_w = full_map_pred.shape[1]

    # local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)
    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(), 0.1, 1)

    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(free_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)
    explo_area_map_cur = np.zeros_like(local_ob_map)

    if len(contours) > 0:
        for contour in contours:
            if cv2.contourArea(contour) > 4: # Exclude very small areas from the example
                cv2.drawContours(explo_area_map_cur, [contour], -1, PR, -1) # Fill the interior with PR

    # Clear the border section
    explo_area_map_cur[0:2, 0:full_w] = 0
    explo_area_map_cur[full_w-2:full_w, 0:full_w] = 0
    explo_area_map_cur[0:full_w, 0:2] = 0
    explo_area_map_cur[0:full_w, full_w-2:full_w] = 0

    if VLM_PR_last:
        # mask = np.logical_and(explo_area_map_cur != PR, explo_area_map == VLM_PR_last[0])
        coords = np.where(explo_area_map != 0)
        # PR_coords = list(zip(coords[0], coords[1]))
        explo_area_map_cur[coords] = explo_area_map[coords]

    
    # Mark explorable areas as current colour
    intensity = int(PR * 100 * 2.55)
    intensity = max(0, min(intensity, 100))
    color_map[np.where(explo_area_map_cur == PR)] = [intensity, intensity, intensity]  #  RGB 值

    lipped_map = cv2.flip(color_map, 0)
    color_map__ = Image.fromarray(lipped_map)
    color_map__ = color_map__.convert("RGB")

    
    return explo_area_map_cur, color_map

def Frontiers(full_map_pred):
    # ------------------------------------------------------------------
    ##### Get the frontier map and filter
    # ------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    full_w = full_map_pred.shape[1]
    local_ex_map = np.zeros((full_w, full_w))
    local_ob_map = np.zeros((full_w, full_w))

    local_ob_map = cv2.dilate(full_map_pred[0].cpu().numpy(), kernel)

    show_ex = cv2.inRange(full_map_pred[1].cpu().numpy(),0.1,1)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

    contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(contours)>0:
        contour = max(contours, key = cv2.contourArea)
        cv2.drawContours(local_ex_map,contour,-1,1,1)

    # clear the boundary
    local_ex_map[0:2, 0:full_w]=0.0
    local_ex_map[full_w-2:full_w, 0:full_w-1]=0.0
    local_ex_map[0:full_w, 0:2]=0.0
    local_ex_map[0:full_w, full_w-2:full_w]=0.0

    target_edge = local_ex_map-local_ob_map
    # print("local_ob_map ", self.local_ob_map[200])
    # print("full_map ", self.full_map[0].cpu().numpy()[200])

    target_edge[target_edge>0.8]=1.0
    target_edge[target_edge!=1.0]=0.0

    wall_edge = local_ex_map - target_edge

    # contours, hierarchy = cv2.findContours(cv2.inRange(wall_edge,0.1,1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours)>0:
    #     dst = np.zeros(wall_edge.shape)
    #     cv2.drawContours(dst, contours, -1, 1, 1)

    # edges = cv2.Canny(cv2.inRange(wall_edge,0.1,1), 30, 90)
    Wall_lines = cv2.HoughLinesP(cv2.inRange(wall_edge,0.1,1), 1, np.pi / 180, threshold=30, minLineLength=10, maxLineGap=10)

    # original_image_color = cv2.cvtColor(cv2.inRange(wall_edge,0.1,1), cv2.COLOR_GRAY2BGR)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    img_label, num = measure.label(target_edge, connectivity=2, return_num=True) # Output all connected fields in the binary image
    props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.

    Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
    Goal_point = []
    Goal_area_list = []
    dict_cost = {}
    for i in range(1, len(props)):
        if props[i].area > 4:
            dict_cost[i] = props[i].area

    if dict_cost:
        dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)

        for i, (key, value) in enumerate(dict_cost):
            Goal_edge[img_label == key + 1] = 1
            Goal_point.append([int(props[key].centroid[0]), int(props[key].centroid[1])])
            Goal_area_list.append(value)
        # frontiers = cv2.HoughLinesP(cv2.inRange(Goal_edge,0.1,1), 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=10)

        # original_image_color = cv2.cvtColor(cv2.inRange(Goal_edge,0.1,1), cv2.COLOR_GRAY2BGR)
        # if frontiers is not None:
        #     for frontier in frontiers:
        #         x1, y1, x2, y2 = frontier[0]
        #         cv2.line(original_image_color, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return Wall_lines, Goal_area_list, Goal_edge, Goal_point

# 画出所有的Frontier
def Visualize(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, Frontiers_dict, goal_points):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, episode_n)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:4 + args.num_sem_categories, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    # no_cat_mask = sem_map == 20
    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True) # Output all connected fields in the binary image
        props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents and goal_points[i][0] != 9999:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i
    

    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    # vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)
    if 'objectnav_mp3d' in args.task_config:
        vis_image = vu.init_multi_vis_image(object_category[goal_name], color)
    elif 'objectnav_hm3d' in args.task_config:
        vis_image = vu.init_multi_vis_image(object_category[coco_categories_hm3d2mp3d[goal_name]], color)

    vis_image[50:530, 15:495] = sem_map_vis

    color_black = (0,0,0)
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    alpha = [chr(ord("A") + i) for i in range(26)]
    alpha0 = 0
    
    def d240(x):
        if x < 240:
            x = x + 2*(240-x)
        elif x >= 240:
            x = x - 2*(x-240)
        return x
    
    frontier_points = frontier_points or []

    frontier_markers: List[Tuple[str, int, int]] = []

    if Frontiers_dict:
        for keys, value in Frontiers_dict.items():
            match = re.match(pattern, value)
            if match:
                centroid_x = int(match.group(1)[1:])
                centroid_y = int(match.group(2)[:-1])
                number = float(match.group(3))
                # print(f"Centroid: ({centroid_x}, {centroid_y})")
                # print(f"Number: {number}")

                cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_black, -1)
                label = alpha_label(alpha0)
                alpha0 += 1
                cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 1)
                frontier_markers.append((keys, centroid_x, centroid_y))

    seen_frontiers: Set[Tuple[int, int]] = set((fy, fx) for _, fy, fx in frontier_markers)

    if frontier_points:
        for idx, pt in enumerate(frontier_points):
            if not pt or len(pt) < 2:
                continue
            fy, fx = int(pt[0]), int(pt[1])
            if (fy, fx) in seen_frontiers:
                continue
            frontier_markers.append((f"frontier_{idx}", fy, fx))
            seen_frontiers.add((fy, fx))

    if frontier_markers:
        color_frontier = (255, 165, 0)
        for idx, (f_key, fy, fx) in enumerate(frontier_markers):
            marker_pos = (int(fx), d240(int(fy)))
            key_suffix = f_key.split("_")[-1]
            label_id = key_suffix if key_suffix.isdigit() else str(idx)
            cv2.drawMarker(sem_map_vis, marker_pos, color_frontier, markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1)
            cv2.putText(sem_map_vis, f"F{label_id}", (marker_pos[0] + 4, marker_pos[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_frontier, 1)
    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)
    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_n {}".format(episode_n), vis_image)
        cv2.waitKey(1)

    if args.print_images:
        fn = '{}/episodes/eps_{}/Step-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        # print(fn)
        cv2.imwrite(fn, vis_image)   

def Decision_Generation_Vis(args, agents_seg_list, agent_j, episode_n, l_step, pose_pred, full_map_pred, goal_name,
                             visited_vis, map_edge, history_nodes, Frontiers_dict, goal_points, pre_goal_point):
    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:4 + args.num_sem_categories, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    # no_cat_mask = sem_map == 20
    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True) # Output all connected fields in the binary image
        props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents and goal_points[i][0] != 9999:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    if Frontiers_dict:
        for keys, value in Frontiers_dict.items():
            match = re.match(pattern, value)
            if match:
                centroid_x = int(match.group(1)[1:])
                centroid_y = int(match.group(2)[:-1])
                number = float(match.group(3))
            fgoal = np.zeros((full_w, full_w)) 
            fgoal[centroid_x, centroid_y] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                fgoal, selem) != True
            goal_mask = goal_mat == 1
            sem_map[goal_mask] = 2

    
    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))
    
    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color_black = (0,0,0)
    color_green = (0,255,0)
    color_red = (0,0,255)
    color_blue = (255,0,0)
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    alpha = [chr(ord("A") + i) for i in range(26)]
    alpha0 = 0
    
    def d240(x):
        if x < 240:
            x = x + 2*(240-x)
        elif x >= 240:
            x = x - 2*(x-240)
        return x

    


    # for i in range(args.num_agents):
    #     agent_arrow = vu.get_contour_points(pose_pred[i], origin=(0, 0), size=10)

    #     cv2.drawContours(sem_map_vis, [agent_arrow], 0, color[i], -1)
    # agent_arrow = vu.get_contour_points(pose_pred[agent_j], origin=(0, 0), size=10)

    # cv2.drawContours(sem_map_vis, [agent_arrow], 0, color[agent_j], -1)
    if Frontiers_dict:
        for keys, value in Frontiers_dict.items():
            match = re.match(pattern, value)
            if match:
                centroid_x = int(match.group(1)[1:])
                centroid_y = int(match.group(2)[:-1])
                number = float(match.group(3))
                # print(f"Centroid: ({centroid_x}, {centroid_y})")
                # print(f"Number: {number}")
                
                cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_black, -1)
                label = alpha_label(alpha0)
                alpha0 += 1
                cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 1)

    sem_map_vis2 = sem_map_vis.copy()
    beta = [chr(ord("a") + i) for i in range(26)]
    alpha0 = 0
    if len(history_nodes) > 0:
        for hs in history_nodes[:26]:
            centroid_x = int(hs[0])
            centroid_y = int(hs[1])
            cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_green, -1)
            label = alpha_label(alpha0, lowercase=True)
            alpha0 += 1
            cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)
        alpha0 = 0
        for hs in history_nodes[26:]:
            centroid_x = int(hs[0])
            centroid_y = int(hs[1])
            cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_green, -1)
            label = alpha_label(alpha0)
            alpha0 += 1
            cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_green, 1)
    # Iterate through the dictionary and draw polygons
    for key, value in agents_seg_list.items():
        # Convert each value into a format suitable for use with cv2.polylines (a numpy array).
        for array in value:
            pts = array.reshape((-1, 1, 2))
            if agent_j == 0:
                for i in pts:
                    for j in i:
                        j[1] = d240(j[1])
            
            # Draw polygons
            # cv2.polylines(sem_map_vis, [pts], isClosed=True, color=color_bule, thickness=2)
            
            # Label the key values with the text at the first coordinate of the polygon.
            text_position = (pts[0][0][0], pts[0][0][1])
            # moments = cv2.moments(pts)
            # cX = int(moments["m10"] / moments["m00"])
            # cY = int(moments["m01"] / moments["m00"])
            cv2.putText(sem_map_vis, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(sem_map_vis2, key, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    

    # Draw the arrows
    # cv2.circle(sem_map_vis, (int(pose_pred[agent_j][0]), int(pose_pred[agent_j][1])), 8, color_red, -1)
    # cv2.circle(sem_map_vis2, (int(pose_pred[agent_j][0]), int(pose_pred[agent_j][1])), 8, color_red, -1)
    
    agent_arrow = vu.get_contour_points(pose_pred[agent_j], origin=(0, 0), size=15)
    cv2.drawContours(sem_map_vis, [agent_arrow], 0, color_red, -1)
    cv2.drawContours(sem_map_vis2, [agent_arrow], 0, color_red, -1)
    if pre_goal_point:
        cv2.circle(sem_map_vis, (int(pre_goal_point[1]), int(d240(pre_goal_point[0]))), 8, color_blue, -1)
        cv2.circle(sem_map_vis2, (int(pre_goal_point[1]), int(d240(pre_goal_point[0]))), 8, color_blue, -1)

    
    
    
    ### TEST
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    vis_ep_dir = '{}/episodes/eps_{}/Agents_vis'.format(
                dump_dir, episode_n)
    if not os.path.exists(vis_ep_dir):
        os.makedirs(vis_ep_dir)
    
    fn = '{}/episodes/eps_{}/Agents_vis/VisStep-{}.png'.format(
                        dump_dir, episode_n,
                        l_step)
    fn2 = '{}/episodes/eps_{}/Agents_vis/VisStep2-{}.png'.format(
                        dump_dir, episode_n,
                        l_step)
    cv2.imwrite(fn, sem_map_vis)  
    cv2.imwrite(fn2, sem_map_vis2) 

    return sem_map_vis, sem_map_vis2



def Visualize0(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, goal_points):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, episode_n)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:4 + args.num_sem_categories, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True) # Output all connected fields in the binary image
        props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i


    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)

    vis_image[50:530, 15:495] = sem_map_vis

    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)

    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_n {}".format(episode_n), vis_image)
        cv2.waitKey(1)

    if args.print_images:
        fn = '{}/episodes/eps_{}/Vis-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        cv2.imwrite(fn, vis_image)

def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# 画出所有的Frontier和检测到的物体
def Visualize_obj(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge,
                  Frontiers_dict, goal_points, object_positions=None, tracked_objects=None,
                  assigned_centroids=None, frontier_points=None):
    dump_dir = "{}/dump/{}/".format(args.dump_location,
                                    args.exp_name)
    ep_dir = '{}/episodes/eps_{}/'.format(
        dump_dir, episode_n)
    if not os.path.exists(ep_dir):
        os.makedirs(ep_dir)

    full_w = full_map_pred.shape[1]

    map_pred = full_map_pred[0, :, :].cpu().numpy()
    exp_pred = full_map_pred[1, :, :].cpu().numpy()

    sem_map = full_map_pred[4:4 + args.num_sem_categories, :,:].argmax(0).cpu().numpy()

    sem_map += 5

    # no_cat_mask = sem_map == 20
    if 'objectnav_hm3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2
    elif 'objectnav_mp3d' in args.task_config:
        no_cat_mask = sem_map == len(object_category) - 2 + 5
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    for i in range(args.num_agents):
        sem_map[visited_vis[i] == 1] = 3+i
    sem_map[edge_mask] = 3


    def find_big_connect(image):
        img_label, num = measure.label(image, return_num=True) # Output all connected fields in the binary image
        props = measure.regionprops(img_label) # Output properties of connected fields, including area, etc.
        # print("img_label.shape: ", img_label.shape) # 480*480
        resMatrix = np.zeros(img_label.shape)
        tmp_area = 0
        for i in range(0, len(props)):
            if props[i].area > tmp_area:
                tmp = (img_label == i + 1).astype(np.uint8)
                resMatrix = tmp
                tmp_area = props[i].area 
        
        return resMatrix

    goal = np.zeros((full_w, full_w)) 
    if 'objectnav_mp3d' in args.task_config:
        cn = goal_name + 4
    elif 'objectnav_hm3d' in args.task_config:
        cn = coco_categories[goal_name] + 4
    if full_map_pred[cn, :, :].sum() != 0.:
        cat_semantic_map = full_map_pred[cn, :, :].cpu().numpy()
        cat_semantic_scores = cat_semantic_map
        cat_semantic_scores[cat_semantic_scores > 0] = 1.
        goal = find_big_connect(cat_semantic_scores)

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
    elif len(goal_points) == args.num_agents and goal_points[i][0] != 9999:
        for i in range(args.num_agents):
            goal = np.zeros((full_w, full_w)) 
            goal[goal_points[i][0], goal_points[i][1]] = 1
            selem = skimage.morphology.disk(4)
            goal_mat = 1 - skimage.morphology.binary_dilation(
                goal, selem) != True
            goal_mask = goal_mat == 1

            sem_map[goal_mask] = 3 + i
    

    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                    sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                interpolation=cv2.INTER_NEAREST)

    color = []
    for i in range(args.num_agents):
        color.append((int(color_palette[11+3*i] * 255),
                    int(color_palette[10+3*i] * 255),
                    int(color_palette[9+3*i] * 255)))

    # vis_image = vu.init_multi_vis_image(category_to_id[goal_name], color)
    if 'objectnav_mp3d' in args.task_config:
        vis_image = vu.init_multi_vis_image(object_category[goal_name], color)
    elif 'objectnav_hm3d' in args.task_config:
        vis_image = vu.init_multi_vis_image(object_category[coco_categories_hm3d2mp3d[goal_name]], color)

    vis_image[50:530, 15:495] = sem_map_vis

    color_black = (0,0,0)
    pattern = r'<centroid: (.*?), (.*?), number: (.*?)>'
    alpha = [chr(ord("A") + i) for i in range(26)]
    alpha0 = 0
    
    def d240(x):
        if x < 240:
            x = x + 2*(240-x)
        elif x >= 240:
            x = x - 2*(x-240)
        return x
    
    if Frontiers_dict:
        for keys, value in Frontiers_dict.items():
            match = re.match(pattern, value)
            if match:
                centroid_x = int(match.group(1)[1:])
                centroid_y = int(match.group(2)[:-1])
                number = float(match.group(3))
                # print(f"Centroid: ({centroid_x}, {centroid_y})")
                # print(f"Number: {number}")
                
                cv2.circle(sem_map_vis, (centroid_y, d240(centroid_x)), 5, color_black, -1)
                label = alpha_label(alpha0)
                alpha0 += 1
                cv2.putText(sem_map_vis, label, (centroid_y + 5, d240(centroid_x) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 1)
    
    # ===== NEW: Object Detection Visualization =====
    if object_positions is not None:
        # Colors for different object states
        color_new_object = (0, 255, 0)      # Green for newly detected objects
        color_existing_object = (255, 0, 0)  # Red for existing tracked objects
        color_updated_object = (0, 0, 255)   # Blue for updated objects
        
        # Draw newly detected objects
        for obj in object_positions:
            # Use the object_state field to determine visualization
            object_state = obj.get('object_state', 'unknown')
            
            if object_state == 'new':
                color = color_new_object
                label_prefix = "N"
            elif object_state == 'updated':
                color = color_existing_object
                label_prefix = "E"
            elif object_state == 'merged':
                color = color_updated_object
                label_prefix = "U"
            else:
                # Fallback for unknown states
                color = color_existing_object
                label_prefix = "?"
            
            # Get object position
            pos = obj['map_position']
            category = obj['category']
            confidence = obj['confidence']
            
            # Convert coordinates to visualization space
            center_x = int(pos['x'])
            center_y = d240(int(pos['y']))
            
            # Draw center point with different markers
            if label_prefix == "N":
                # Star for new objects
                cv2.drawMarker(sem_map_vis, (center_x, center_y), color, cv2.MARKER_STAR, 8, 2)
            elif label_prefix == "E":
                # Circle for existing objects
                cv2.circle(sem_map_vis, (center_x, center_y), 4, color, -1)
            elif label_prefix == "U":
                # Diamond for updated objects
                cv2.drawMarker(sem_map_vis, (center_x, center_y), color, cv2.MARKER_DIAMOND, 8, 2)
            
            # Draw label with category and confidence
            label = f"{label_prefix}:{category[:3]}({confidence:.2f})"
            cv2.putText(sem_map_vis, label, (center_x + 8, center_y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw all tracked objects (if provided separately)
    if tracked_objects is not None:
        color_tracked = (128, 128, 128)  # Gray for all tracked objects
        
        for obj_id, obj in tracked_objects.items():
            pos = obj['map_position']
            category = obj['category']
            
            # Convert coordinates
            center_x = int(pos['x'])
            center_y = d240(int(pos['y']))
            
            # Draw small square for tracked objects
            cv2.drawMarker(sem_map_vis, (center_x, center_y), color_tracked, cv2.MARKER_SQUARE, 4, 1)
            
            # Draw object ID
            cv2.putText(sem_map_vis, f"ID:{obj_id}", (center_x + 5, center_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_tracked, 1)
    
    # Debug: show AIRD assigned centroids (if provided)
    if assigned_centroids:
        palette = {
            'agent_0': (0, 255, 255),
            'agent_1': (255, 255, 0),
            'agent_2': (255, 0, 255),
            'agent_3': (0, 165, 255),
        }
        for agent_key, coord in assigned_centroids.items():
            try:
                cx, cy = float(coord[0]), float(coord[1])
            except Exception:
                continue
            px = int(cx)
            py = d240(int(cy))
            color = palette.get(agent_key, (255, 255, 255))
            cv2.drawMarker(sem_map_vis, (px, py), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(sem_map_vis, agent_key, (px + 6, py + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Update the main visualization image with object overlays
    vis_image[50:530, 15:495] = sem_map_vis
    
    for i in range(args.num_agents):
        agent_arrow = vu.get_contour_points(pose_pred[i], origin=(15, 50), size=10)

        cv2.drawContours(vis_image, [agent_arrow], 0, color[i], -1)
    if args.visualize:
        # Displaying the image
        cv2.imshow("episode_n {}".format(episode_n), vis_image)
        cv2.waitKey(1)

    if args.print_images:
        fn = '{}/episodes/eps_{}/Step-{}.png'.format(
            dump_dir, episode_n,
            l_step)
        # print(fn)
        cv2.imwrite(fn, vis_image)   



def _format_object_description(obj: Dict) -> str:
    # pos = obj.get("map_position", {})
    # x = pos.get("x", "?")
    # y = pos.get("y", "?")
    # conf = obj.get("confidence")
    # conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "unknown"
    # area = obj.get("area")
    # area_str = str(area) if area is not None else "unknown"
    # state = obj.get("object_state", "unknown")
    category = obj.get("category", "unknown")
    # return (
    #     f"object_id={obj.get('object_id', 'unknown')}, category={category}, position=({x}, {y}), "
    #     f"confidence={conf_str}, area={area_str}, state={state}"
    # )
    return (
        f"object_id={obj.get('object_id', 'unknown')}, category={category} "
        # f"confidence={conf_str}, area={area_str}, state={state}"
    )


def _build_subgroup_prompt(agent_label: str, goal_name: Optional[str], subgroup_id: int,
                           assignment_id: int, object_descriptions: List[str]) -> str:
    goal_fragment = goal_name or "unknown"

    header = (
        f"You are a helpful assistant planning its next move in a indoor environment. The high-level task is to find the target object '{goal_fragment}'.\n"
        # "You must find this target as quickly as possible. Below is the list of observed objects.\n"
        # "Each item is listed as object_id=..., category=...."
    )
    object_lines = "\n".join(object_descriptions)
    instructions = (
        "\nSelect the single object whose category suggests it is most likely close to the '{goal_fragment}' in a indoor house.\n"
        # "If none of the objects are likely to be near the target, output the single letter 'N'.\n"
        # "Reply with only the integer object_id, or 'N' (no JSON, no extra words)."
        "Reply with only the integer object_id (no JSON, no extra words)."
    )
    return header + "\n" + object_lines + instructions


def _query_llm_for_object(llm_client: CogVLM2, prompt: str) -> str:
    if llm_client is None:
        return ""
    messages = [
        {
            "role": "system",
            "content": "You help robots choose navigation targets. Always reply with a single integer object_id and nothing else.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    try:
        _, content = llm_client.create_chat_completion(
            "cogvlm2",
            messages=messages,
            temperature=0.2,
            top_p=0.9,
            max_tokens=256,
            use_stream=False,
        )
        return content or ""
    except Exception as exc:
        logging.warning("LLM query failed: %s", exc)
        return ""


def _parse_llm_object_id(raw_response: str, valid_ids: List[int]) -> int:
    if not raw_response:
        return -1
    match_int = re.search(r"(-?\d+)", raw_response)
    if match_int:
        obj_id = int(match_int.group(1))
        if not valid_ids or obj_id in valid_ids:
            return obj_id
    return -1


def _choose_objects_with_llm(llm_clients: List[CogVLM2],
                             subgroup_selections: Dict[str, List[int]],
                             assignment_ids: Dict[str, List[int]],
                             lookup: Dict[int, Dict],
                             map_manager: MapManager,
                             agents: List[LLM_Agent]) -> Dict[str, Dict]:
    if not subgroup_selections or not llm_clients:
        return {}

    def _worker(llm_client: CogVLM2, agent_key: str, subgroup_id: int, assignment_id: int) -> Tuple[str, Dict]:
        lookup_entry = lookup.get(assignment_id, {})
        member_ids = lookup_entry.get("members", [])
        if not member_ids:
            return agent_key, {}
        candidate_objects = []
        for member_id in member_ids:
            obj = map_manager.tracked_objects.get(member_id)
            if obj:
                candidate_objects.append(obj)
        if not candidate_objects:
            return agent_key, {}
        
        # Remove redundant categories - keep only the first object_id for each category
        seen_categories = set()
        deduplicated_objects = []
        for obj in candidate_objects:
            category = obj.get('category', 'unknown')
            if category not in seen_categories:
                seen_categories.add(category)
                deduplicated_objects.append(obj)
        
        candidate_objects = deduplicated_objects[:12]
        descriptions = [_format_object_description(obj) for obj in candidate_objects]
        agent_idx = int(agent_key.split("_")[1]) if "_" in agent_key else 0
        goal_name = agents[agent_idx].goal_name if agent_idx < len(agents) else "unknown"
        prompt = _build_subgroup_prompt(agent_key, goal_name, subgroup_id, assignment_id, descriptions)
        raw = _query_llm_for_object(llm_client, prompt)
        valid_ids = [int(obj.get("object_id", -1)) for obj in candidate_objects if obj.get("object_id") is not None]
        chosen_id = _parse_llm_object_id(raw, valid_ids)
        if chosen_id == -1 and valid_ids:
            chosen_id = valid_ids[0]
        selected = next((obj for obj in candidate_objects if int(obj.get("object_id", -1)) == chosen_id), None)
        if not selected and candidate_objects:
            selected = candidate_objects[0]
        payload = {
            "object": selected,
            "assignment_id": assignment_id,
            "subgroup_id": subgroup_id,
            "raw_response": raw,
        } if selected else {}
        return agent_key, payload

    outputs: Dict[str, Dict] = {}
    client_count = max(1, len(llm_clients))
    with ThreadPoolExecutor(max_workers=max(1, len(subgroup_selections))) as executor:
        futures = []
        for task_idx, (agent_key, subgroup_ids) in enumerate(subgroup_selections.items()):
            if not subgroup_ids:
                continue
            subgroup_id = subgroup_ids[0]
            assigned_list = assignment_ids.get(agent_key, [])
            assignment_id = None
            for cand in assigned_list:
                entry = lookup.get(cand)
                if entry and entry.get("subgroup") == subgroup_id:
                    assignment_id = cand
                    break
            if assignment_id is None and assigned_list:
                assignment_id = assigned_list[0]
            if assignment_id is None:
                continue
            llm_client = llm_clients[task_idx % client_count]
            futures.append(executor.submit(_worker, llm_client, agent_key, subgroup_id, assignment_id))

        for future in as_completed(futures):
            try:
                agent_key, payload = future.result()
            except Exception as exc:
                logging.warning("LLM selection task failed: %s", exc)
                continue
            if payload:
                outputs[agent_key] = payload

    return outputs

def _nearest_frontier_goal(agent_idx: int,
                           agent_target_point_map: List[List[List[int]]],
                           cur_goal_points: List[List[int]],
                           used_frontiers: Optional[Set[Tuple[int, int]]] = None,
                           global_frontiers: Optional[List[List[int]]] = None) -> Optional[List[int]]:
    frontier_points: List[List[int]] = []
    if global_frontiers:
        frontier_points = list(global_frontiers)
    elif agent_idx < len(agent_target_point_map):
        frontier_points = agent_target_point_map[agent_idx]
    if not frontier_points:
        return None
    current = cur_goal_points[agent_idx] if agent_idx < len(cur_goal_points) else None
    # Filter out already claimed frontier goals
    filtered: List[Tuple[int, int]] = []
    for pt in frontier_points:
        goal = (int(pt[0]), int(pt[1]))
        if used_frontiers is not None and goal in used_frontiers:
            continue
        filtered.append(goal)

    if not filtered:
        return None

    # When no other agent has claimed a frontier yet, keep the original
    # behavior of staying near the previous goal to reduce large redirects.
    chosen: Tuple[int, int]
    if not used_frontiers or len(used_frontiers) == 0:
        if current:
            cy, cx = current[0], current[1]
            chosen = min(filtered, key=lambda goal: (goal[0] - cy) ** 2 + (goal[1] - cx) ** 2)
        else:
            chosen = filtered[0]
    else:
        # Spread agents out: pick the frontier that maximizes its minimum distance
        # to the already claimed frontier assignments.
        def min_sq_dist_to_used(goal: Tuple[int, int]) -> float:
            return min((goal[0] - uy) ** 2 + (goal[1] - ux) ** 2 for uy, ux in used_frontiers)

        chosen = max(filtered, key=min_sq_dist_to_used)

    if used_frontiers is not None:
        used_frontiers.add(chosen)
    return [chosen[0], chosen[1]]



def _assign_groups_with_aird(grouper, pose_pred, args, price_coordinator, map_manager, agents, llm_clients,
                             deprecated_assignments: Optional[Set[int]] = None,
                             agent_cooldowns: Optional[Dict[str, Set[int]]] = None,
                             score_bar: Optional[float] = None,
                             episode_idx=None, step_idx=None):
    groups, lookup = grouper.get_selector_groups()

    if not groups:
        return None

    if deprecated_assignments:
        groups = [g for g in groups if g["g_id"] not in deprecated_assignments]
        if not groups:
            return None
        active_ids = {g["g_id"] for g in groups}
        lookup = {gid: info for gid, info in lookup.items() if gid in active_ids}

    agents_payload = {}
    for idx, pose in enumerate(pose_pred):
        if pose is None:
                continue
        agents_payload[f"agent_{idx}"] = {
            "pose": (float(pose[0]), float(pose[1]), float(pose[2])),
            "B_mem": float(args.aide_mem_bytes),
            "eps_H": float(args.aide_epsilon),
        }

    if not agents_payload:
        return None

    assignment_ids, stats, debug = one_step_assign(
        groups,
        agents_payload,
        alpha_base=float(args.aide_alpha),
        price_coordinator=price_coordinator,
        w_dist=0.6,
        intent_topk=max(1, min(len(groups), args.aide_topk_cands)),
        agent_forbidden=agent_cooldowns,
        score_bar=score_bar,
    )

    subgroup_selections: Dict[str, List[int]] = {}
    for agent_key, assigned in assignment_ids.items():
        for gid in assigned:
            info = lookup.get(gid)
            if not info:
                continue
            subgroup_id = int(info.get("subgroup", -1))
            if subgroup_id < 0:
                continue
            subgroup_selections.setdefault(agent_key, []).append(subgroup_id)

    centroids = {}
    for agent_key, assigned in assignment_ids.items():
        if assigned:
            gid = assigned[0]
            centroid = lookup.get(gid, {}).get("centroid")
            if centroid is not None:
                centroids[agent_key] = centroid

    llm_objects = _choose_objects_with_llm(llm_clients, subgroup_selections, assignment_ids, lookup, map_manager, agents)

    result = {
        "selections": subgroup_selections,
        "raw_assignments": assignment_ids,
        "lookup": lookup,
        "stats": stats,
        "debug": debug,
        "centroids": centroids,
        "llm_objects": llm_objects,
        "fallback_agents": debug.get("fallback_agents", []) if debug else [],
    }

    if getattr(args, "aide_debug", False):
        result["groups_payload"] = groups
        result["agents_payload"] = agents_payload
        result["raw_debug"] = debug
        print("[AIRD] Episode %s Step %s | price=%.4f comm=%.1f H=%.3f" %
                     (str(episode_idx), str(step_idx), stats.get("p", 0.0), stats.get("total_C", 0.0), stats.get("total_H", 0.0)))
        for agent_key, selected in assignment_ids.items():
            centroid = centroids.get(agent_key)
            print("    %s -> %s centroid=%s" % (agent_key, selected, centroid))
            ranklist = debug.get("raw_results", {}).get(agent_key, {}).get("ranklist") if debug else None
            if ranklist:
                print("      top bids: %s" % ranklist[:5])
        if llm_objects:
            for agent_key, payload in llm_objects.items():
                obj = payload.get("object") if payload else None
                if obj:
                    print("    %s -> LLM object %s (%s)" % (agent_key, obj.get('object_id'), obj.get('category')))

        if args.dump_location:
            debug_dir = os.path.join(args.dump_location, args.exp_name, "aird_debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(
                debug_dir,
                f"episode_{str(episode_idx).zfill(4)}_step_{str(step_idx).zfill(4)}.json",
            )
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "groups": groups,
                        "agents": agents_payload,
                        "selections": subgroup_selections,
                        "centroids": centroids,
                        "stats": stats,
                        "debug": debug,
                        "llm_objects": llm_objects,
                    }, f, indent=2)
            except Exception as ex:  # pragma: no cover
                print("Failed to write AIRD debug dump: %s" % ex)
    # pdb.set_trace()
    return result


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if args.cuda else "cpu")

###########################################################===Load Models and Datasets===########################################################
    # Initialise habitat
    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")
    print("+"*10)
    print(args.task_config)

    config_env = habitat.get_config(config_paths=["envs/habitat/configs/"
                                         + args.task_config])
    config_env.defrost()

    # Optional: override dataset path to a specific val_<idx>
    if getattr(args, 'val_idx', None) is not None and 'objectnav_hm3d' in args.task_config:
        idx = int(args.val_idx)
        override_path = f"data/datasets/objectnav_hm3d_v2/val_{idx}/val.json.gz"
        print(f"Overriding DATASET.DATA_PATH -> {override_path}")
        config_env.DATASET.DATA_PATH = override_path
        # Keep SPLIT as 'val' for downstream logic, path override is explicit
    
    agent_sensors = []
    agent_sensors.append("RGB_SENSOR")
    agent_sensors.append("DEPTH_SENSOR")
    agent_sensors.append("SEMANTIC_SENSOR")

    config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
    config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = \
        [0, args.camera_height, 0]

    config_env.TASK.POSSIBLE_ACTIONS = config_env.TASK.POSSIBLE_ACTIONS + [
        "TURN_LEFT_S",
        "TURN_RIGHT_S",
    ]
    config_env.TASK.ACTIONS.TURN_LEFT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_LEFT_S.TYPE = "TurnLeftAction_S"
    config_env.TASK.ACTIONS.TURN_RIGHT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_RIGHT_S.TYPE = "TurnRightAction_S"
    config_env.SIMULATOR.ACTION_SPACE_CONFIG = "PreciseTurn"
    config_env.freeze()

    # ------------------------------------------------------------------
    # Load VLM
    # ------------------------------------------------------------------
    # vlm = VLM(args.vlm_model_id, args.hf_token, device)
    # base_url_raw = args.base_url or ""
    # base_urls = [url.strip() for url in str(base_url_raw).split(',') if url.strip()]
    base_urls = ['http://127.0.0.1:31511','http://127.0.0.1:31512']

    if not base_urls:
        base_urls = ["http://127.0.0.1:31511"]
    cogvlm_clients = [CogVLM2(url) for url in base_urls]
    cogvlm2 = cogvlm_clients[0]
    # ------------------------------------------------------------------
    # Load Yolo
    # ------------------------------------------------------------------
    # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
    if args.yolo == 'yolov9':
        # yolo = Detect(imgsz=(args.env_frame_height, args.env_frame_width), device=device)
        pass
    else:
        yolo = YOLO('./aide_tests/yolov10x.pt')
    print(config_env)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    
    env = Multi_Agent_Env(config_env=config_env)
    grouper = SemanticSpatialGrouper(GroupingCfg(
        clip_model="openai/clip-vit-large-patch14",
        # clip_model='openai/clip-vit-base-patch32',
        device="cuda",
        tau_assign=0.82,
        spatial_radius_m=27,
        max_subgroup_kv_bytes=200_000,
        # enable_spatial_subgrouping=False
    ))

    aird_price_coordinator = None
    if getattr(args, "aide", False):
        total_h_budget = float(args.aide_epsilon) * float(args.num_agents)
        aird_price_coordinator = PriceCoordinator(
            B_comm_bytes=float(args.aide_comm_bytes),
            B_H=total_h_budget,
        )

    num_episodes = env.number_of_episodes

    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config_env.SIMULATOR.NUM_AGENTS

    agent = []
    agent_GT = []
    for i in range(num_agents):
        agent.append(LLM_Agent(args, config_env, i, device))
        agent_GT.append(LLM_Agent_GT(args, config_env, i, device))

    map_manager = MapManager(args, device)
    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    log_file = os.path.join(log_dir, "output.log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(levelname)s - %(message)s")

    file_handler_exists = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == os.path.abspath(log_file)
        for handler in root_logger.handlers
    )
    if not file_handler_exists:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    if getattr(args, "log_to_stdout", False):
        stream_handler_exists = any(
            isinstance(handler, logging.StreamHandler)
            and getattr(handler, "stream", None) in (sys.stdout, sys.__stdout__)
            for handler in root_logger.handlers
        )
        if not stream_handler_exists:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            root_logger.addHandler(stream_handler)

    print("Dumping at {}".format(log_dir))
    # print(args)
    # logging.info(args)
    # ------------------------------------------------------------------

    # print("num_episodes:",num_episodes)# 1000

    agg_metrics: Dict = defaultdict(float)
    # obj_SR: Dict = defaultdict(float)
    # sys_metrics: Dict = defaultdict(float)
    agg_metrics['multi_Total_SR'] = 0
    agg_metrics['SPL'] = 0
    agg_metrics['SoftSPL'] = 0
    agg_metrics['SPL_valid'] = 0  # episodes with at least one valid SPL
    agg_metrics['Skipped_Episodes'] = 0  # episodes skipped due to invalid initial pose
    agg_metrics['multi_GTCategory_SR'] = 0
    agg_metrics['multi_SPL'] = {}
    agg_metrics['multi_SoftSPL'] = {}
    agg_metrics['multi_SPL_valid'] = {}  # per-agent valid SPL counts
    agg_metrics['multi_Navigation_SR'] = 0
    for i in range(num_agents):
        agg_metrics['multi_SPL'][f'Agent_{i}'] = 0
        agg_metrics['multi_SoftSPL'][f'Agent_{i}'] = 0
        agg_metrics['multi_SPL_valid'][f'Agent_{i}'] = 0

    count_episodes = 0
    count_step = 0
    goal_points = []
    
    log_start = time.time()
    last_decision = []
    total_usage = []

    history_nodes = []
    history_score = []
    history_count = []
    history_states = []

    cur_goal_points = []
    pre_goal_points = []

    # random
    log_start = time.time()
    last_decision = []
    total_usage = []

    pre_g_points = []

    target_point = []

    exhausted_assignments: Set[int] = set()
    agent_active_assignments: Dict[int, Optional[int]] = {i: None for i in range(num_agents)}

    deprecated_assignments: Set[int] = set()
    agent_assignment_cooldown: Dict[str, Set[int]] = defaultdict(set)
    aide_score_bar = getattr(args, "aide_score_bar", None)
    try:
        aide_score_bar = float(aide_score_bar) if aide_score_bar is not None else None
    except (TypeError, ValueError):
        aide_score_bar = None

    

    # logging.info(f"num agents: {num_agents}")

###########################################################===Main MCoCoNav===########################################################
    while count_episodes < num_episodes:
        observations = env.reset()
        for i in range(num_agents):
            agent[i].reset()
            if agent_GT:
                agent_GT[i].reset()

        map_manager.reset_tracking()
        if hasattr(grouper, 'reset'):
            grouper.reset()

        history_nodes.clear()
        history_score.clear()
        history_count.clear()
        history_states.clear()
        pre_g_points.clear()
        target_point.clear()
        exhausted_assignments.clear()
        for idx in range(num_agents):
            agent_active_assignments[idx] = None
        deprecated_assignments.clear()
        agent_assignment_cooldown.clear()

        goal_points.clear()
        try:
            map_dim = int(args.map_size_cm // args.map_resolution)
        except Exception:
            map_dim = 480
        map_dim = max(2, map_dim)
        initial_corners = [
            [0, 0],
            [0, map_dim - 1],
            [map_dim - 1, 0],
            [map_dim - 1, map_dim - 1],
        ]
        for j in range(num_agents):
            corner = initial_corners[j % len(initial_corners)]
            goal_points.append(list(corner))
        logging.info("Initial corner goals assigned: %s", goal_points)

        # Early-episode validity check: ensure each agent has a valid initial pose window.
        # If any agent lacks a valid planner pose input, skip this episode entirely.
        all_agents_valid = True
        for i in range(num_agents):
            ppi = getattr(agent[i], 'planner_pose_inputs', None)
            if ppi is None or (isinstance(ppi, np.ndarray) and not np.any(ppi)):
                all_agents_valid = False
                break
            # Initialize Start_Location from the first available pose window if missing
            try:
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = ppi
                r, c = start_y, start_x
                start = [int(r * 100.0 / args.map_resolution - int(gx1)),
                         int(c * 100.0 / args.map_resolution - int(gy1))]
                # Use the local map shape if available to clamp indices
                local_shape = agent[i].local_map[0, :, :].cpu().numpy().shape if hasattr(agent[i], 'local_map') else None
                if local_shape is not None:
                    start = pu.threshold_poses(start, local_shape)
                if getattr(agent[i], 'Start_Location', None) is None:
                    agent[i].Start_Location = start
            except Exception:
                # Any failure to parse a valid window counts as invalid for this episode
                all_agents_valid = False
                break

        if not all_agents_valid:
            logging.warning("Skipping episode due to invalid initial pose window(s).")
            agg_metrics['Skipped_Episodes'] += 1
            continue

        while not env.episode_over:
            
            # ===== EPISODE STEP TIMING ANALYSIS =====
            step_start_time = time.time()
            print(f"===== EPISODE STEP TIMING - Step {agent[0].l_step} =====")
            
            Local_Policy = 0 # local policy
            start = time.time()
            count_rotating = 0
            action = [0] * num_agents

            all_rgb = []
            
            full_map = []
            full_map1 = []
            visited_vis = []
            pose_pred = []
            agent_objs = {} # Record target detection information for each smart body in a single time step

            agent_FrontierList = [] # Record the robot Frontier
            agent_TargetEdgeMap = []
            agent_TargetPointMap = []
            agent_MapPred = []

            # Time agent mapping operations
            mapping_start = time.time()
            for i in range(num_agents):
                agent[i].mapping(observations[i])
                if agent_GT:
                    agent_GT[i].mapping(observations[i])
                    # cat=['chair', 'table', 'picture', 'cabinet', 'cushion', 'sofa', 'bed', 'chest_of_drawers', 'plant', 'sink', 'toilet', 'stool', 'towel', 'tv_monitor', 'shower', 'bathtub', 'counter', 'fireplace', 'gym_equipment', 'seating', 'clothes', 'background']
                    # # Plot the semantic segmentation over the RGB image using the category names in 'cat'
                    # import matplotlib.pyplot as plt
                    # import matplotlib.patches as mpatches


                    # rgb_img = observations[i]['rgb']  # shape: (H, W, 3)
                    # semantic = observations[i]['semantic']  # shape: (H, W)
                    # # Build a color map for each category
                    # num_classes = len(cat)
                    # # Use a fixed colormap for reproducibility
                    # cmap = plt.get_cmap('tab20', num_classes)
                    # semantic_rgb = cmap(semantic % num_classes)[..., :3]  # shape: (H, W, 3), values in [0,1]

                    # # Blend the RGB and semantic mask
                    # alpha = 0.5
                    # rgb_norm = rgb_img.astype(np.float32) / 255.0
                    # overlay = (1 - alpha) * rgb_norm + alpha * semantic_rgb
                    # overlay = np.clip(overlay, 0, 1)

                    # # Plot
                    # plt.figure(figsize=(8, 8))
                    # plt.imshow(overlay)
                    # # Build legend
                    # handles = []
                    # for idx, name in enumerate(cat):
                    #     color = cmap(idx)[:3]
                    #     handles.append(mpatches.Patch(color=color, label=name))
                    # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    # plt.axis('off')
                    # plt.title("Semantic Overlay")
                    # plt.tight_layout()
                    # plt.savefig('semantic_overlay.png', dpi=300, bbox_inches='tight')
                    # # INSERT_YOUR_CODE
                    # # Also save the RGB image for debugging/inspection
                    # plt.imsave('semantic_overlay_rgb.png', rgb_img)
                    # plt.close()

                local_map1, _ = torch.max(agent[i].local_map.unsqueeze(0), 0)
                full_map.append(agent[i].local_map)
                visited_vis.append(agent[i].visited_vis)
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs

                gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
                pos = (
                    (start_x * 100. / args.map_resolution - gy1)
                    * 480 / agent[i].visited_vis.shape[0],
                    (agent[i].visited_vis.shape[1] - start_y * 100. / args.map_resolution + gx1)
                    * 480 / agent[i].visited_vis.shape[1],
                    np.deg2rad(-start_o)
                )
                pose_pred.append(pos)

            mapping_time = time.time() - mapping_start
            print(f"Agent Mapping Time: {mapping_time:.4f}s")

            
            # Time map processing and frontier detection
            map_processing_start = time.time()
            # pdb.set_trace()
            full_map2 = torch.stack(full_map, dim=0)
            full_map_pred = torch.max(full_map2, dim=0).values
            Wall_list, full_Frontier_list, full_target_edge_map, full_target_point_map = Frontiers(full_map_pred)
            map_processing_time = time.time() - map_processing_start
            print(f"Map Processing & Frontier Detection Time: {map_processing_time:.4f}s")

            print(":"*20)
            # here implement merge info of tracked object from different agents

            is_local_planning_step = (
                agent[0].l_step % args.num_local_steps == args.num_local_steps - 1
                or agent[0].l_step == 0
            )
            logging.info(
                "===== STEP %d (local_planning=%s) =====",
                agent[0].l_step,
                is_local_planning_step,
            )
            logging.info(
                "Agent poses: %s",
                {
                    f"agent_{idx}": (
                        float(pose_pred[idx][0]) if pose_pred[idx] is not None else None,
                        float(pose_pred[idx][1]) if pose_pred[idx] is not None else None,
                        float(pose_pred[idx][2]) if pose_pred[idx] is not None else None,
                    )
                    for idx in range(len(pose_pred))
                },
            )

            # ===== TIMING ANALYSIS: Before Grouper Operations =====
            grouper_start_time = time.time()
            print(f"===== GROUPER TIMING ANALYSIS - Step {agent[0].l_step} =====")
            
            # Time object position extraction
            obj_pos_start = time.time()
            numpy_full_map = full_map_pred.cpu().numpy()
            object_positions = map_manager.get_object_positions(numpy_full_map, agent[0].object_category)
            obj_pos_time = time.time() - obj_pos_start
            print(f"Object Position Extraction Time: {obj_pos_time:.4f}s")
            logging.info(
                "Map manager tracked objects=%d returned positions=%d",
                len(map_manager.tracked_objects) if map_manager.tracked_objects else 0,
                len(object_positions),
            )
            
            # Time newly detected objects processing
            new_obj_start = time.time()
            newly_detected_objects = map_manager.get_newly_added_objects(object_positions)
            new_obj_time = time.time() - new_obj_start
            print(f"Newly Detected Objects Processing Time: {new_obj_time:.4f}s")
            logging.info(
                "Newly detected objects: %d (total tracked now=%d)",
                len(newly_detected_objects),
                len(map_manager.tracked_objects) if map_manager.tracked_objects else 0,
            )


            goal_text = agent[0].goal_name
            grouper.set_goal_text(goal_text)
            logging.info("Current goal text: %s", goal_text)

            # Time detection list creation
            det_list_start = time.time()
            det_list = []
            if newly_detected_objects:
                for o in newly_detected_objects:       # your merged list
                    det_list.append(Det(det_id=o["object_id"],
                                        category=o["category"],
                                        xy_m=(o["map_position"]["x"] , o["map_position"]["y"] ),
                                        status=o["object_state"],
                                        conf=float(o.get("confidence", 1.0))))
            det_list_time = time.time() - det_list_start
            print(f"Detection List Creation Time: {det_list_time:.4f}s")
            logging.info("Detections passed to grouper: %d", len(det_list))

            # Time grouper add_detections
            add_det_start = time.time()
            if det_list:
                grouper.add_detections(det_list)
            add_det_time = time.time() - add_det_start
            print(f"Grouper Add Detections Time: {add_det_time:.4f}s")
            tracked_keys = list(map_manager.tracked_objects.keys()) if map_manager.tracked_objects else []
            if hasattr(grouper, "sem_groups"):
                grouper.prune_missing_detections(tracked_keys)
                num_sem_groups = len(grouper.sem_groups)
                total_subgroups = sum(len(G.subgroups) for G in grouper.sem_groups)
                logging.info(
                    "Grouper state: sem_groups=%d subgroups=%d (active tracked=%d)",
                    num_sem_groups,
                    total_subgroups,
                    len(tracked_keys),
                )
                if num_sem_groups == 0:
                    logging.info("Grouper detail: no semantic groups currently tracked.")
                else:
                    for G in grouper.sem_groups:
                        logging.info(
                            "Grouper SemGroup gid=%s members=%s kv=%d v=%.3f h=%.4f c=%.0f",
                            G.gid,
                            list(G.member_ids),
                            int(getattr(G, "kv_bytes_est", 0)),
                            float(getattr(G, "v_score", 0.0)),
                            float(getattr(G, "h_score", 0.0)),
                            float(getattr(G, "c_score", 0.0)),
                        )
                        if not G.subgroups:
                            logging.info("  No spatial subgroups under gid=%s", G.gid)
                        else:
                            for S in G.subgroups:
                                # ensure no duplicate detections linger in subgroup state
                                if S.member_ids:
                                    seen_ids: Set[int] = set()
                                    unique_ids: List[int] = []
                                    for det_id in S.member_ids:
                                        if det_id in seen_ids:
                                            continue
                                        seen_ids.add(det_id)
                                        unique_ids.append(det_id)
                                    if len(unique_ids) != len(S.member_ids):
                                        logging.warning(
                                            "Duplicate detection IDs detected in subgroup gid=%s sid=%s; deduplicating for logging.",
                                            G.gid,
                                            S.sid,
                                        )
                                        S.member_ids = unique_ids
                                member_details: List[Dict[str, object]] = []
                                for det_id in S.member_ids:
                                    obj = map_manager.tracked_objects[det_id]
                                    member_details.append({
                                        "id": int(det_id),
                                        "category": str(obj["category"])
                                    })
                                logging.info(
                                    "  SubGroup sid=%s parent=%s members=%s centroid=%s kv=%d v=%.3f h=%.4f c=%.0f",
                                    S.sid,
                                    getattr(S, "parent_gid", None),
                                    member_details,
                                    tuple(map(float, S.centroid_xy)) if S.centroid_xy is not None else None,
                                    int(getattr(S, "kv_bytes_est", 0)),
                                    float(getattr(S, "v_score", 0.0)),
                                    float(getattr(S, "h_score", 0.0)),
                                    float(getattr(S, "c_score", 0.0)),
                                )
                                if getattr(S, "tokens", None):
                                    logging.info("    tokens: %s", S.tokens)
                                if getattr(S, "selector_entry", None):
                                    logging.info("    selector_entry: %s", S.selector_entry)
            else:
                grouper.prune_missing_detections(tracked_keys)
            
            # Time visualization
            # vis_start = time.time()
            # paths = grouper.visualize_grouping(agent[0].l_step,map_manager.tracked_objects, grouper.sem_groups,
            #                 res_m_per_px=0.05, origin_xy_m=(0.0, 0.0),
            #                 spatial_radius_m=1.2, out_prefix="demo_group")
            # vis_time = time.time() - vis_start
            # print(f"Grouper Visualization Time: {vis_time:.4f}s")
            
            # # Total grouper time
            # total_grouper_time = time.time() - grouper_start_time
            # print(f"TOTAL GROUPER TIME: {total_grouper_time:.4f}s")
            # print(f"Grouper Breakdown:")
            # print(f"  - Object Position Extraction: {obj_pos_time:.4f}s ({obj_pos_time/total_grouper_time*100:.1f}%)")
            # print(f"  - New Objects Processing: {new_obj_time:.4f}s ({new_obj_time/total_grouper_time*100:.1f}%)")
            # print(f"  - Detection List Creation: {det_list_time:.4f}s ({det_list_time/total_grouper_time*100:.1f}%)")
            # print(f"  - Grouper Add Detections: {add_det_time:.4f}s ({add_det_time/total_grouper_time*100:.1f}%)")
            # print(f"  - Visualization: {vis_time:.4f}s ({vis_time/total_grouper_time*100:.1f}%)")
            # print("="*50)


            # pdb.set_trace()

            
            if agent[0].goal_id + 4 > 24:
                break
            # logging.info(f"agent[0].l_step % args.num_local_steps == args.num_local_steps - 1: {agent[0].l_step % args.num_local_steps, args.num_local_steps - 1}")
            # logging.info(f"agent[0].l_step == 0: {agent[0].l_step }")
            if is_local_planning_step:
                # Time decision making phase
                decision_start = time.time()
                for j in range(num_agents):
                    agent[j].Perception_PR = 0
                
                # Time object extraction
                obj_extract_start = time.time()
                agents_seg_list = Objects_Extract(args, full_map_pred, args.use_sam)
                obj_extract_time = time.time() - obj_extract_start
                print(f"Object Extraction Time: {obj_extract_time:.4f}s")

                pre_goal_points.clear()
                if len(cur_goal_points) > 0:
                    pre_goal_points = cur_goal_points.copy()
                    cur_goal_points.clear()
                
                
                if len(full_target_point_map) > 0:
                    full_Frontiers_dict = {}
                    for j in range(len(full_target_point_map)):
                        full_Frontiers_dict['frontier_' + str(j)] = f"<centroid: {full_target_point_map[j][0], full_target_point_map[j][1]}, number: {full_Frontier_list[j]}>"
                    logging.info(f'=====> Frontier: {full_Frontiers_dict}')

                    if len(history_nodes) > 0:
                        logging.info(f'=====> history_nodes: {history_nodes}')
                        logging.info(f'=====> history_score: {history_score}')

                    # ------------------------------------------------------------------
                    ##### VLM Preliminaries :>
                    # ------------------------------------------------------------------
                    for j in range(num_agents):
                        agent[j].is_Frontier = True
                        rgb = observations[j]['rgb'].astype(np.uint8)
                        
                        # full_rgb1.append(full_rgb)
                        all_rgb.append(rgb)
                        goal_name = agent[j].goal_name
                        # if args.yolo == 'yolov9':
                        #     agent_objs[f"agent_{j}"] = yolo.run(rgb) # Record target detection information for each robot in a single time step.
                        # else:
                        #     yolo_output = yolo(source=rgb,conf=0.2)
                        #     yolo_mapping = [yolo_output[0].names[int(c)] for c in yolo_output[0].boxes.cls]
                        #     agent_objs[f"agent_{j}"] = {k: v for k, v in zip(yolo_mapping, yolo_output[0].boxes.conf)}
                        # logging.info(agent_objs)
                        
                        # agents_seg_list = Objects_Extract(local_map1, args.use_sam)
                        single_map = [full_map[j]]

                        full_map1.append(torch.cat([fm.unsqueeze(0) for fm in single_map], dim=0))
                        full_map_pred1, _ = torch.max(full_map1[j], 0)
                        Wall_list, Frontier_list, target_edge_map, target_point_map = Frontiers(full_map_pred1)
                        agent_FrontierList.append(Frontier_list)
                        agent_TargetEdgeMap.append(target_edge_map)
                        agent_TargetPointMap.append(target_point_map)
                        agent_MapPred.append(full_map_pred1)

                        

                        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[j].planner_pose_inputs
                        r, c = start_y, start_x
                        start = [int(r * 100.0 / args.map_resolution - gx1),
                                int(c * 100.0 / args.map_resolution - gy1)]
                        start = pu.threshold_poses(start, agent[j].local_map[0, :, :].cpu().numpy().shape)
                        
                        # if len(pre_goal_points) > 0:
                        #     # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                        #     #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points[j])
                        #     sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                        #             agent[0].goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=pre_goal_points[j])
                        # else:
                        #     # sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, agent_MapPred[j], 
                        #     #                 agent[j].goal_id, visited_vis[j], agent_TargetEdgeMap[j], history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=None)
                        #     sem_map, sem_map_frontier = Decision_Generation_Vis(args, agents_seg_list, j, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                        #             agent[0].goal_id, visited_vis, full_target_edge_map, history_nodes, full_Frontiers_dict, goal_points=[], pre_goal_point=None)
                        # full_rgb = np.hstack((rgb, sem_map))

                        # ------------------------------------------------------------------
                        #### Perception VLM
                        # ------------------------------------------------------------------
                        # Caption_Prompt, VLM_Perception_Prompt = form_prompt_for_PerceptionVLM(goal_name, agent_objs[f'agent_{j}'], args.yolo)
                        # print("+"*10)
                        # print(Caption_Prompt)
                        # print("+"*10)
                        
                        # Time the first LLM call (Scene Information)
                        # start_time = time.time()
                        # _, Scene_Information = cogvlm2.simple_image_chat(User_Prompt=Caption_Prompt, 
                                                                        # return_string_probabilities=None, img=rgb)
                        # scene_info_time = time.time() - start_time
                        # logging.info(f"Agent_{j}--LLM_Scene_Info_Time: {scene_info_time:.3f}s")
                        
                        # Time the second LLM call (Perception)
                        # start_time = time.time()
                        # Perception_Rel, Perception_Pred = cogvlm2.CoT2(User_Prompt1=Caption_Prompt, 
                                                                    #    User_Prompt2=VLM_Perception_Prompt,
                                                                    #    cot_pred1=Scene_Information,
                                                                    #    return_string_probabilities="[Yes, No]", img=rgb)
                        # perception_time = time.time() - start_time
                        # logging.info(f"Agent_{j}--LLM_Perception_Time: {perception_time:.3f}s")
                        # Perception_Rel = np.array(Perception_Rel)
                        # Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
                        # logging.info(f"Agent_{j}--VLM_PerceptionPR: {Perception_PR}")
                        # agents_VLM_Rel[f"Agent_{i}--VLM_PerceptionRel"] = Perception_Rel
                        # agents_VLM_Pred[f"Agent_{i}--VLM_PerceptionPred"] = Perception_Pred
                        # agents_VLM_PR[f"Agent_{i}--VLM_PerceptionPR"] = Perception_PR

                        

                        is_exist_oldhistory = False
                        if len(history_nodes) > 0:
                            closest_index = -1
                            min_distance = float('inf')
                            new_x, new_y = start
                            for i, (x, y) in enumerate(history_nodes):
                                distance = math.sqrt((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y))
                                if distance < 25 and distance < min_distance:
                                    min_distance = distance
                                    closest_index = i
                                    is_exist_oldhistory = True

                            if  is_exist_oldhistory == False:
                                history_nodes.append(start)
                                history_count.append(1)
                                history_state = np.zeros(360)
                            else:
                                history_count[closest_index] = history_count[closest_index] + 1

                            
                        else:
                            history_nodes.append(start)
                            history_count.append(1)
                            history_state = np.zeros(360)

                        
                        cur_goal_points.append(start)
                        if len(agent_TargetPointMap[j]) > 0:
                            
                            logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')

                            # ------------------------------------------------------------------
                            #### Judgment VLM
                            # ------------------------------------------------------------------
                            # if len(history_nodes) > 0:
                                # if len(pre_goal_points) > 0:
                                    # FN_Prompt = form_prompt_for_FN(goal_name, agents_seg_list, Perception_PR, pre_goal_points[j], full_Frontiers_dict, start, history_nodes)
                                # else:
                                    # FN_Prompt = form_prompt_for_FN(goal_name, agents_seg_list, Perception_PR, pre_goal_points, full_Frontiers_dict, start, history_nodes)
                                # logging.info(FN_Prompt)
                                
                                # Time the Judgment VLM call
                                # start_time = time.time()
                                # FN_Rel, FN_Decision = cogvlm2.simple_image_chat(User_Prompt=FN_Prompt, 
                                                                                        # return_string_probabilities="[Yes, No]", img=sem_map)
                                # judgment_time = time.time() - start_time
                                # logging.info(f"Agent_{j}--LLM_Judgment_Time: {judgment_time:.3f}s")
                                
                                # Calculate total LLM time so far for this agent
                                # total_llm_time = scene_info_time + perception_time + judgment_time
                                # logging.info(f"Agent_{j}--Total_LLM_Time_So_Far: {total_llm_time:.3f}s")

                                # FN_PR = Perception_weight_decision(FN_Rel, FN_Decision)
                                # logging.info(f"Agent_{j}--FN_PR: {FN_PR}")
                                # if FN_PR == 'Neither':
                                    # FN_PR = FN_Rel

                                
                                
                                # angle_score = Perception_PR[0] * 2 + FN_PR[0]
                            #     agent[j].angle_score = angle_score
                            #     c_angle = int(start_o % 360)

                            #     if is_exist_oldhistory == False:
                            #         if c_angle >= 39 and c_angle < 321:
                            #             history_state[c_angle-39:c_angle+39] = angle_score
                            #         elif c_angle < 39:
                            #             history_state[:c_angle+39] = angle_score
                            #             history_state[360-c_angle-39:] = angle_score

                            #         elif c_angle >= 321:
                            #             history_state[c_angle-39:] = angle_score
                            #             history_state[:c_angle+39-360] = angle_score
                            #         h_score = history_state.sum()
                            #         history_states.append(history_state)
                            #         history_score.append(h_score)
                            #     else:
                            #         if c_angle >= 39 and c_angle < 321:
                            #             history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                            #         elif c_angle < 39:
                            #             history_states[closest_index][:c_angle] = angle_score
                            #             history_states[closest_index][360-c_angle:] = angle_score
                            #         elif c_angle >= 321:
                            #             history_states[closest_index][c_angle:] = angle_score
                            #             history_states[closest_index][:360-c_angle] = angle_score
                            #         h_score = history_states[closest_index].sum() / history_count[closest_index]
                            #         history_score[closest_index] = h_score

                            # logging.info(f'=====> history_nodes: {history_nodes}')
                            # logging.info(f'=====> history_score: {history_score}')
                            # Scores = []
                            # if j == 0:
                            #     history_nodes_copy = history_nodes.copy()
                            #     history_score_copy = history_score.copy()
                            #     full_Frontiers_dict_copy = full_Frontiers_dict.copy()
                            # else:
                            #     missing_key_F = []
                            #     if len(full_Frontiers_dict) == 4:
                            #         frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                            #     elif len(full_Frontiers_dict) == 3:
                            #         frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2']
                            #     elif len(full_Frontiers_dict) == 2:
                            #         frontier_keys = ['frontier_0', 'frontier_1']
                            #     else:
                            #         frontier_keys = ['frontier_0']

                            #     for element in full_Frontiers_dict.keys():
                            #         if element not in full_Frontiers_dict_copy.keys():
                            #             missing_key_F.append(element)
                                # for element in history_nodes:
                                #     if element not in history_nodes_copy:
                                #         missing_index_H.append(element.index(element))
                            # if FN_PR[0] >= 0.5 or agent[j].l_step <= 125:
                            #     # ------------------------------------------------------------------
                            #     #### Decision VLM
                            #     # ------------------------------------------------------------------
                            #     if len(pre_goal_points) > 0:
                            #         Meta_Prompt = form_prompt_for_DecisionVLM_Frontier(Scene_Information, agents_seg_list, pre_goal_points[j], goal_name, start, full_Frontiers_dict_copy)
                            #     else:
                            #         Meta_Prompt = form_prompt_for_DecisionVLM_Frontier(Scene_Information, agents_seg_list, pre_goal_points, goal_name, start, full_Frontiers_dict_copy)
                                
                            #     # Time the Decision VLM call
                            #     start_time = time.time()
                            #     Meta_Score, Meta_Choice = cogvlm2.simple_image_chat(User_Prompt=Meta_Prompt,
                            #                                 return_string_probabilities="[A, B, C, D]", img=sem_map_frontier)
                            #     decision_time = time.time() - start_time
                            #     logging.info(f"Agent_{j}--LLM_Decision_Time: {decision_time:.3f}s")
                                
                            #     # Calculate total LLM time for this agent (including decision)
                            #     total_llm_time_with_decision = scene_info_time + perception_time + judgment_time + decision_time
                            #     logging.info(f"Agent_{j}--Total_LLM_Time_Complete: {total_llm_time_with_decision:.3f}s")

                            #     Final_PR = Perception_weight_decision4(Meta_Score, Meta_Choice)
                                
                            # else:
                            #     Final_PR = history_score_copy

                            # logging.info(f"Agent_{j}--Final_PR: {Final_PR}")

                            # Scores.append(Final_PR)
                            # Choice = Final_PR.index(max(Final_PR))
                            
                            
                            # if FN_PR[0] >= 0.5 or agent[j].l_step <= 125:
                            #     logging.info(f"VLM Choice: Agent_{j}-frontier_{Choice}")
                            #     Choice2 = Meta_Score.index(max(Meta_Score))

                            #     if len(full_Frontiers_dict) == 1:
                            #         goal_points[j] = [int(x) for x in full_Frontiers_dict['frontier_0'].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                
                            #     elif len(full_Frontiers_dict) == 2 and num_agents == 3:
                            #         if j == 0:
                            #             for i, key in enumerate(frontier_keys):
                            #                 if Choice == i:
                            #                     if key in full_Frontiers_dict_copy:
                            #                         goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                            #                         del full_Frontiers_dict_copy[key]
                            #         elif j == 1:
                            #             if len(missing_key_F) != 0:
                            #                 for keys in missing_key_F:
                            #                     frontier_keys.remove(keys)
                            #             for i, key in enumerate(frontier_keys):
                            #                 goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                            #         else:
                            #             if len(missing_key_F) != 0:
                            #                 for keys in missing_key_F:
                            #                     frontier_keys.remove(keys)
                            #             for i, key in enumerate(frontier_keys):
                            #                 goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                                
                                
                            #     else:
                            #         if j > 0:
                            #             if len(missing_key_F) != 0:
                            #                 for keys in missing_key_F:
                            #                     frontier_keys.remove(keys)
                            #         else:
                            #             if len(full_Frontiers_dict) == 4:
                            #                 frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                            #             elif len(full_Frontiers_dict) == 3:
                            #                 frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2']
                            #             elif len(full_Frontiers_dict) == 2:
                            #                 frontier_keys = ['frontier_0', 'frontier_1']
                            #             else:
                            #                 frontier_keys = ['frontier_0']

                            #         invalid_answer = False
                            #         for i, key in enumerate(frontier_keys):
                            #             if Choice == i:
                            #                 if key in full_Frontiers_dict_copy:
                            #                     goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                            #                     del full_Frontiers_dict_copy[key]
                            #                 else:
                            #                     invalid_answer = True
                            #                 break
                            #         if invalid_answer:
                            #             for i, key in enumerate(frontier_keys):
                            #                 if Choice2 == i:
                            #                     try:
                            #                         goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[key].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                            #                         del full_Frontiers_dict_copy[key]
                            #                         break
                            #                     except:
                            #                         goal_points[j] = [int(x) for x in full_Frontiers_dict_copy[frontier_keys[0]].split('centroid: ')[1].split(', number: ')[0][1:-1].split(', ')]
                            #                         del full_Frontiers_dict_copy[frontier_keys[0]]
                            #                         break
                                        
                            if not getattr(args, "aide", False):
                                # Set goals based on random semantic group objects
                                # Collect all available objects from semantic groups
                                candidates = []
                                for sem_group in grouper.sem_groups:
                                    for subgroup in sem_group.subgroups:
                                        for member_id in subgroup.member_ids:
                                            if member_id in map_manager.tracked_objects:
                                                obj = map_manager.tracked_objects[member_id]
                                                candidates.append({
                                                    "id": member_id,
                                                    "position": obj["map_position"],
                                                    "sem_group_id": sem_group.gid,
                                                    "subgroup_id": subgroup.sid,
                                                    "v_score": subgroup.v_score,
                                                    "h_score": subgroup.h_score,
                                                    "c_score": subgroup.c_score,
                                                })

                                if candidates:
                                    candidates.sort(key=lambda c: c["v_score"], reverse=True)
                                    selected_obj = candidates[0]
                                    goal_points[j] = [
                                        int(selected_obj["position"]["y"]),
                                        int(selected_obj["position"]["x"]),
                                    ]
                                    print(
                                        f"Agent {j} picked object {selected_obj['id']} "
                                        f"(group {selected_obj['sem_group_id']}, "
                                        f"subgroup {selected_obj['subgroup_id']}) "
                                        f"with V/H/C = "
                                        f"{selected_obj['v_score']:.3f}/"
                                        f"{selected_obj['h_score']:.3f}/"
                                        f"{selected_obj['c_score']:.0f}"
                                    )
                                    logging.info(
                                        "Goal assignment: agent_%d top-V semantic object %s from group %s/%s -> %s",
                                        j,
                                        selected_obj["id"],
                                        selected_obj["sem_group_id"],
                                        selected_obj["subgroup_id"],
                                        goal_points[j],
                                    )

                        else:
                            logging.info(f'===== Agent_{j} No Frontier, Random Mode =====')
                            #### Modify to history node
                            agent[j].is_Frontier = False
                            c_angle = int(start_o % 360)
                            # angle_score = Perception_PR[0] * 2
                            # agent[j].angle_score = angle_score

                            # if is_exist_oldhistory == False:
                            #     if c_angle >= 39 and c_angle < 321:
                            #         history_state[c_angle-39:c_angle+39] = angle_score
                            #     elif c_angle < 39:
                            #         history_state[:c_angle+39] = angle_score
                            #         history_state[360-c_angle-39:] = angle_score

                            #     elif c_angle >= 321:
                            #         history_state[c_angle-39:] = angle_score
                            #         history_state[:c_angle+39-360] = angle_score
                            #     h_score = history_state.sum()
                            #     history_states.append(history_state)
                            #     history_score.append(h_score)
                            # else:
                            #     if c_angle >= 39 and c_angle < 321:
                            #         history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                            #     elif c_angle < 39:
                            #         history_states[closest_index][:c_angle] = angle_score
                            #         history_states[closest_index][360-c_angle:] = angle_score
                            #     elif c_angle >= 321:
                            #         history_states[closest_index][c_angle:] = angle_score
                            #         history_states[closest_index][:360-c_angle] = angle_score
                            #     h_score = history_states[closest_index].sum() / history_count[closest_index]
                            #     history_score[closest_index] = h_score

                            if j == 0:
                                history_nodes_copy = history_nodes.copy()
                                history_score_copy = history_score.copy()
                                full_Frontiers_dict_copy = full_Frontiers_dict.copy()
                            
                            if len(full_Frontiers_dict) == 1:
                                logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                                # Set goals based on random semantic group objects
                                # Collect all available objects from semantic groups
                                available_objects = []
                                for sem_group in grouper.sem_groups:
                                    for subgroup in sem_group.subgroups:
                                        for member_id in subgroup.member_ids:
                                            if member_id in map_manager.tracked_objects:
                                                obj_data = map_manager.tracked_objects[member_id]
                                                available_objects.append({
                                                    'id': member_id,
                                                    'map_position': obj_data.get('map_position'),
                                                    'sem_group_id': sem_group.gid,
                                                    'subgroup_id': subgroup.sid
                                                })
                                            else:
                                                print(f"Warning: member_id {member_id} not found in tracked_objects (likely merged/deleted)")
                                
                                if len(available_objects) > 0:
                                    # Select a random object as goal

                                    selected_obj = random.choice(available_objects)
                                    print(f"Agent {j} selected object: ID={selected_obj['id']}, position={selected_obj['map_position']}, sem_group={selected_obj['sem_group_id']}, subgroup={selected_obj['subgroup_id']}")

                                    pos = selected_obj.get('map_position') or {}
                                    if 'y' in pos and 'x' in pos:
                                        goal_points[j] = [int(pos['y']), int(pos['x'])]
                                    print(f"Agent {j} goal set to: {goal_points[j]} (semantic object)")
                                    logging.info(
                                        "Goal assignment: agent_%d semantic object random pick -> %s (object %s)",
                                        j,
                                        goal_points[j],
                                        selected_obj['id'],
                                    )
                                else:
                                    # Fallback to center if no semantic objects available
                                    center_x, center_y = full_target_edge_map.shape[0] // 2, full_target_edge_map.shape[1] // 2
                                    center_goal = [center_x, center_y]
                                    goal_points[j] = list(center_goal)
                                    print(f"Agent {j} fallback goal: {center_goal} (center - no semantic objects)")
                                    logging.info(
                                        "Goal assignment: agent_%d fallback center -> %s",
                                        j,
                                        goal_points[j],
                                    )
                            else:
                                if  j == 0:
                                    frontier_keys = ['frontier_0', 'frontier_1', 'frontier_2', 'frontier_3']
                                logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                                # Set goals based on random semantic group objects
                                # Collect all available objects from semantic groups
                                available_objects = []
                                for sem_group in grouper.sem_groups:
                                    for subgroup in sem_group.subgroups:
                                        for member_id in subgroup.member_ids:
                                            if member_id in map_manager.tracked_objects:
                                                obj_data = map_manager.tracked_objects[member_id]
                                                available_objects.append({
                                                    'id': member_id,
                                                    'map_position': obj_data.get('map_position'),
                                                    'sem_group_id': sem_group.gid,
                                                    'subgroup_id': subgroup.sid
                                                })
                                            else:
                                                print(f"Warning: member_id {member_id} not found in tracked_objects (likely merged/deleted)")
                                
                                if len(available_objects) > 0:
                                    # Select a random object as goal
                                    import random
                                    selected_obj = random.choice(available_objects)
                                    print(f"Agent {j} selected object: ID={selected_obj['id']}, position={selected_obj['map_position']}, sem_group={selected_obj['sem_group_id']}, subgroup={selected_obj['subgroup_id']}")

                                    pos = selected_obj.get('map_position') or {}
                                    if 'y' in pos and 'x' in pos:
                                        goal_points[j] = [int(pos['y']), int(pos['x'])]
                                    print(f"Agent {j} goal set to: {goal_points[j]} (semantic object)")
                                    logging.info(
                                        "Goal assignment: agent_%d semantic object random pick -> %s (object %s)",
                                        j,
                                        goal_points[j],
                                        selected_obj['id'],
                                    )
                                else:
                                    # Fallback to center if no semantic objects available
                                    center_x, center_y = full_target_edge_map.shape[0] // 2, full_target_edge_map.shape[1] // 2
                                    center_goal = [center_x, center_y]
                                    goal_points[j] = list(center_goal)
                                    print(f"Agent {j} fallback goal: {center_goal} (center - no semantic objects)")
                                    logging.info(
                                        "Goal assignment: agent_%d fallback center -> %s",
                                        j,
                                        goal_points[j],
                                    )
                            
                            
                    
                    # all_objs.append(agent_objs) 
                    # all_VLM_Pred.append(agents_VLM_Pred)
                    # all_VLM_PR.append(agents_VLM_PR)

                else:
                    
                    logging.info(f'===== No Frontier, Random Mode===== ')
                    logging.info(f'=====> Agent_{j} state: Step: {agent[j].l_step}; Angle: {start_o}')
                    
                    for j in range(num_agents):
                        agent[j].is_Frontier = False
                        rgb = observations[j]['rgb'].astype(np.uint8)
                        
                        # full_rgb1.append(full_rgb)
                        all_rgb.append(rgb)
                        goal_name = agent[j].goal_name
                        if args.yolo == 'yolov9':
                            agent_objs[f"agent_{j}"] = yolo.run(rgb) # Record target detection information for each smart body in a single time step
                        else:
                            yolo_output = yolo(source=rgb,conf=0.2)
                            yolo_mapping = [yolo_output[0].names[int(c)] for c in yolo_output[0].boxes.cls]
                            agent_objs[f"agent_{j}"] = {k: v for k, v in zip(yolo_mapping, yolo_output[0].boxes.conf)}
                        # logging.info(agent_objs)

                        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[j].planner_pose_inputs
                        r, c = start_y, start_x
                        start = [int(r * 100.0 / args.map_resolution - gx1),
                                int(c * 100.0 / args.map_resolution - gy1)]
                        start = pu.threshold_poses(start, agent[j].local_map[0, :, :].cpu().numpy().shape)
                        
                        cur_goal_points.append(start)

                        # # ------------------------------------------------------------------
                        # #### Perception VLM
                        # # ------------------------------------------------------------------
                        # Caption_Prompt, VLM_Perception_Prompt = form_prompt_for_PerceptionVLM(goal_name, agent_objs[f'agent_{j}'], args.yolo)
                        
                        # # Time the second Scene Information LLM call
                        # start_time = time.time()
                        # _, Scene_Information = cogvlm2.simple_image_chat(User_Prompt=Caption_Prompt, 
                        #                                                 return_string_probabilities=None, img=rgb)
                        # scene_info_time2 = time.time() - start_time
                        # logging.info(f"Agent_{j}--LLM_Scene_Info_Time2: {scene_info_time2:.3f}s")
                        
                        # # Time the second Perception LLM call
                        # start_time = time.time()
                        # Perception_Rel, Perception_Pred = cogvlm2.CoT2(User_Prompt1=Caption_Prompt, 
                        #                                                User_Prompt2=VLM_Perception_Prompt,
                        #                                                cot_pred1=Scene_Information,
                        #                                                return_string_probabilities="[Yes, No]", img=rgb)
                        # perception_time2 = time.time() - start_time
                        # logging.info(f"Agent_{j}--LLM_Perception_Time2: {perception_time2:.3f}s")
                        
                        # # Calculate total LLM time for the second set of calls
                        # total_llm_time2 = scene_info_time2 + perception_time2
                        # logging.info(f"Agent_{j}--Total_LLM_Time_Set2: {total_llm_time2:.3f}s")
                        
                        # Perception_Rel = np.array(Perception_Rel)
                        # Perception_PR = Perception_weight_decision(Perception_Rel, Perception_Pred)
                        # logging.info(f"Agent_{j}--VLM_PerceptionPR: {Perception_PR}")

                        # is_exist_oldhistory = False
                        # if len(history_nodes) > 0:
                        #     closest_index = -1
                        #     min_distance = float('inf')
                        #     new_x, new_y = start
                        #     for i, (x, y) in enumerate(history_nodes):
                        #         distance = math.sqrt((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y))
                        #         if distance < 25 and distance < min_distance:
                        #             min_distance = distance
                        #             closest_index = i
                        #             is_exist_oldhistory = True

                        #     if  is_exist_oldhistory == False:
                        #         history_nodes.append(start)
                        #         history_count.append(1)
                        #         history_state = np.zeros(360)
                        #     else:
                        #         history_count[closest_index] = history_count[closest_index] + 1

                            
                        # else:
                        #     history_nodes.append(start)
                        #     history_count.append(1)
                        #     history_state = np.zeros(360)


                        # angle_score = Perception_PR[0] * 2
                        # agent[j].angle_score = angle_score
                        # c_angle = int(start_o % 360)

                        # if is_exist_oldhistory == False:
                        #     if c_angle >= 39 and c_angle < 321:
                        #         history_state[c_angle-39:c_angle+39] = angle_score
                        #     elif c_angle < 39:
                        #         history_state[:c_angle+39] = angle_score
                        #         history_state[360-c_angle-39:] = angle_score

                        #     elif c_angle >= 321:
                        #         history_state[c_angle-39:] = angle_score
                        #         history_state[:c_angle+39-360] = angle_score
                        #     h_score = history_state.sum()
                        #     history_states.append(history_state)
                        #     history_score.append(h_score)
                        # else:
                        #     if c_angle >= 39 and c_angle < 321:
                        #         history_states[closest_index][c_angle-39:c_angle+39] = angle_score
                        #     elif c_angle < 39:
                        #         history_states[closest_index][:c_angle] = angle_score
                        #         history_states[closest_index][360-c_angle:] = angle_score
                        #     elif c_angle >= 321:
                        #         history_states[closest_index][c_angle:] = angle_score
                        #         history_states[closest_index][:360-c_angle] = angle_score
                        #     h_score = history_states[closest_index].sum() / history_count[closest_index]
                        #     history_score[closest_index] = h_score


                        if not getattr(args, "aide", False):
                            # Set goals based on random semantic group objects
                            # Collect all available objects from semantic groups
                            available_objects = []
                            for sem_group in grouper.sem_groups:
                                for subgroup in sem_group.subgroups:
                                    for member_id in subgroup.member_ids:
                                        if member_id in map_manager.tracked_objects:
                                            obj_data = map_manager.tracked_objects[member_id]
                                            available_objects.append({
                                                'id': member_id,
                                                'position': obj_data['position'],
                                                'sem_group_id': sem_group.gid,
                                                'subgroup_id': subgroup.sid
                                            })
                            
                            if len(available_objects) > 0:
                                # Select a random object as goal
                                import random
                                selected_obj = random.choice(available_objects)
                                print(f"Agent {j} selected object: ID={selected_obj['id']}, position={selected_obj['position']}, sem_group={selected_obj['sem_group_id']}, subgroup={selected_obj['subgroup_id']}")
                                
                                # Convert object position from meters to map coordinates
                                map_resolution = 0.05  # meters per pixel
                                global_goal = [
                                    int(selected_obj['position']['y'] / map_resolution),
                                    int(selected_obj['position']['x'] / map_resolution)
                                ]
                                
                                # Transform to local coordinates
                                local_goal = list(global_goal)
                                goal_points[j] = local_goal
                                print(f"Agent {j} global goal: {global_goal}, local goal: {goal_points[j]} (semantic object)")
                                logging.info(
                                    "Goal assignment: agent_%d semantic object random selection -> %s (global %s, object %s)",
                                    j,
                                    goal_points[j],
                                    global_goal,
                                    selected_obj['id'],
                                )
                            else:
                                # Fallback to center if no semantic objects available
                                center_x, center_y = full_target_edge_map.shape[0] // 2, full_target_edge_map.shape[1] // 2
                                center_goal = [center_x, center_y]
                                goal_points[j] = list(center_goal)
                                print(f"Agent {j} fallback goal: {center_goal}, local goal: {goal_points[j]} (center - no semantic objects)")
                                logging.info(
                                    "Goal assignment: agent_%d fallback center -> %s (no semantic objects available)",
                                    j,
                                    goal_points[j],
                                )

                        
                # ------------------------------------------------------------------
                #### Logical Analysis
                # ------------------------------------------------------------------
                # The current scene is worth exploring and the intelligences are not in Frontier
                for i in range(num_agents):
                    if len(pre_g_points) == 0:
                        break
                    if calculate_distance(cur_goal_points[i], pre_g_points[i]) >= 25 and agent[i].is_Frontier == True:
                        # print(calculate_distance(cur_goal_points[i], pre_g_points[i]))
                        goal_points[i] = list(pre_g_points[i])
                        logging.info(
                            "Goal assignment: agent_%d keeps previous global goal %s (distance %.2f)",
                            i,
                            goal_points[i],
                            calculate_distance(cur_goal_points[i], pre_g_points[i]),
                        )

                # Local_Policy = 1
                # Determine the distance, if the distance between two intervals is too short choose a random point for navigation
                for i in range(num_agents):
                    if len(pre_goal_points) > 0 and calculate_distance(pre_goal_points[i], cur_goal_points[i]) <1:
                        actions = np.random.rand(1, 2).squeeze()*(full_target_edge_map.shape[0] - 1)
                        goal_points[i] = [int(actions[0]), int(actions[1])]
                        logging.info(
                            "Goal assignment: agent_%d jitter random target %s (close to previous %.2f)",
                            i,
                            goal_points[i],
                            calculate_distance(pre_goal_points[i], cur_goal_points[i]),
                        )
                

                # logging.info(f"pre_g_points: {pre_g_points}")        
                
                assigned_centroids = None
                # pdb.set_trace()
                
                # ------------------------------------------------------------------
                #### Check for agents that have found the target
                # ------------------------------------------------------------------
                # If an agent has found the target, it should continue navigating to it
                # and not be reassigned by AIDE or frontier logic
                agents_found_goal = []
                for agent_idx in range(num_agents):
                    if agent[agent_idx].Find_Goal:
                        agents_found_goal.append(agent_idx)
                        logging.info(
                            "Agent_%d has found the target; will continue navigating to goal (no reassignment)",
                            agent_idx
                        )
                
                # ------------------------------------------------------------------
                #### First Step Wrapper: Assign initial spread-out goals to agents
                # ------------------------------------------------------------------
                # At the first step of each episode, skip AIDE and frontier logic
                # and assign predetermined subgoals to spread agents out
                is_first_step = (agent[0].l_step == 0)
                
                if is_first_step:
                    logging.info("===== FIRST STEP OF EPISODE: Assigning initial spread-out goals =====")
                    map_dim = full_target_edge_map.shape[0]  # Map dimension (e.g., 480)
                    
                    # Define initial goals that spread agents to different areas
                    # For 2 agents: opposite corners
                    # For more agents: distributed around the map
                    initial_goals = []
                    if num_agents == 2:
                        # Two agents: opposite corners
                        initial_goals = [
                            [map_dim // 4, map_dim // 4],           # Agent 0: top-left quadrant
                            [3 * map_dim // 4, 3 * map_dim // 4],  # Agent 1: bottom-right quadrant
                        ]
                    elif num_agents == 3:
                        # Three agents: spread in triangle pattern
                        initial_goals = [
                            [map_dim // 4, map_dim // 4],           # Agent 0: top-left
                            [3 * map_dim // 4, map_dim // 4],       # Agent 1: top-right
                            [map_dim // 2, 3 * map_dim // 4],       # Agent 2: bottom-center
                        ]
                    elif num_agents == 4:
                        # Four agents: one per corner
                        initial_goals = [
                            [map_dim // 4, map_dim // 4],           # Agent 0: top-left
                            [3 * map_dim // 4, map_dim // 4],       # Agent 1: top-right
                            [map_dim // 4, 3 * map_dim // 4],       # Agent 2: bottom-left
                            [3 * map_dim // 4, 3 * map_dim // 4],  # Agent 3: bottom-right
                        ]
                    else:
                        # For any number of agents: distribute evenly in a grid pattern
                        grid_size = int(np.ceil(np.sqrt(num_agents)))
                        for i in range(num_agents):
                            row = i // grid_size
                            col = i % grid_size
                            y = int((row + 1) * map_dim / (grid_size + 1))
                            x = int((col + 1) * map_dim / (grid_size + 1))
                            initial_goals.append([y, x])
                    
                    # Assign initial goals to each agent
                    for agent_idx in range(num_agents):
                        # Skip if agent has already found the goal
                        if agent_idx in agents_found_goal:
                            logging.info(
                                "Agent_%d already found goal; keeping current goal %s",
                                agent_idx,
                                goal_points[agent_idx]
                            )
                            continue
                        
                        if agent_idx < len(initial_goals):
                            goal_points[agent_idx] = initial_goals[agent_idx]
                        else:
                            # Fallback: use map center if we somehow don't have enough goals
                            goal_points[agent_idx] = [map_dim // 2, map_dim // 2]
                        
                        logging.info(
                            "Goal assignment [initial spread]: agent_%d -> (%d, %d)",
                            agent_idx,
                            goal_points[agent_idx][1],  # x
                            goal_points[agent_idx][0],  # y
                        )
                    
                    logging.info("===== FIRST STEP: Skipping AIDE and frontier logic =====")
                
                # ------------------------------------------------------------------
                #### Normal Planning: AIDE or Frontier-based exploration
                # ------------------------------------------------------------------
                # For steps after the first, use normal AIDE + frontier logic
                elif getattr(args, "aide", False) and aird_price_coordinator is not None:
                    previous_assignments = dict(agent_active_assignments)
                    new_assignments = {i: None for i in range(num_agents)}
                    combined_deprecated = set(deprecated_assignments) | exhausted_assignments
                    air_result = _assign_groups_with_aird(
                        grouper,
                        pose_pred,
                        args,
                        aird_price_coordinator,
                        map_manager,
                        agent,
                        cogvlm_clients,
                        deprecated_assignments=combined_deprecated,
                        agent_cooldowns=agent_assignment_cooldown,
                        score_bar=aide_score_bar,
                        episode_idx=agent[0].episode_n,
                        step_idx=agent[0].l_step,
                    )
                    logging.info(
                        "AIRD call: deprecated=%s exhausted=%s active=%s cooldown=%s",
                        sorted(combined_deprecated),
                        sorted(exhausted_assignments),
                        dict(agent_active_assignments),
                        {k: sorted(v) for k, v in agent_assignment_cooldown.items()},
                    )

                    if air_result:
                        subgroup_selections = air_result["selections"]
                        assignment_ids = air_result["raw_assignments"]
                        lookup = air_result["lookup"]
                        stats = air_result["stats"]
                        logging.info(
                            "AIRD assignment: total_C=%.1f, total_H=%.3f, price_p=%.4f",
                            stats["total_C"], stats["total_H"], stats["p"]
                        )
                        groups_payload = air_result.get("groups_payload", [])
                        if groups_payload:
                            group_descriptions = []
                            for g in groups_payload:
                                try:
                                    desc = (
                                        f"g{g.get('g_id')} "
                                        f"V={g.get('V', 0.0):.3f} "
                                        f"H={g.get('H', 0.0):.4f} "
                                        f"C={g.get('C', 0.0):.0f} "
                                        f"centroid=({g.get('cx', 0.0):.1f},{g.get('cy', 0.0):.1f})"
                                    )
                                except Exception:
                                    desc = str(g)
                                group_descriptions.append(desc)
                            logging.info("AIRD groups: %s", "; ".join(group_descriptions))
                        elif lookup:
                            lookup_descriptions = []
                            for gid, info in lookup.items():
                                centroid = info.get("centroid")
                                sem_gid = info.get("sem_group")
                                subgroup = info.get("subgroup")
                                members = info.get("members") or []
                                lookup_descriptions.append(
                                    f"g{gid} sem={sem_gid} sub={subgroup} centroid={centroid} members={members}"
                                )
                            logging.info("AIRD group lookup (no payload available): %s", "; ".join(lookup_descriptions))

                        debug_payload = air_result.get("debug") or {}
                        winners = debug_payload.get("winners") or {}
                        raw_results = debug_payload.get("raw_results") or {}
                        for agent_key in sorted(raw_results.keys()):
                            agent_res = raw_results[agent_key]
                            logging.info(
                                "AIRD agent %s | selected=%s U_sum=%.3f C_sum=%.3f H_sum=%.4f lambda=%.3f ranklist=%s",
                                agent_key,
                                agent_res.get("selected"),
                                agent_res.get("U_sum", 0.0),
                                agent_res.get("C_sum", 0.0),
                                agent_res.get("H_sum", 0.0),
                                agent_res.get("lambda_star", 0.0),
                                agent_res.get("ranklist"),
                            )
                        logging.info("AIRD assignments per agent: %s", assignment_ids)
                        logging.info("AIRD subgroup selections: %s", subgroup_selections)
                        logging.info("AIRD winners: %s", winners)
                        processed_assignments: Set[int] = set()
                        for gid, win_agents in winners.items():
                            if not win_agents:
                                continue
                            keep_agent = win_agents[0]
                            processed_assignments.add(gid)
                            for agent_idx in range(num_agents):
                                key = f"agent_{agent_idx}"
                                assigned_list = assignment_ids.get(key, [])
                                if not assigned_list:
                                    continue
                                if assigned_list[0] != gid:
                                    continue
                                if key == keep_agent:
                                    continue
                                logging.info(
                                    "Assignment %s reserved for %s; %s will explore frontier",
                                    gid,
                                    keep_agent,
                                    key,
                                )
                                assignment_ids[key] = []
                                subgroup_selections[key] = []

                        for agent_idx in range(num_agents):
                            key = f"agent_{agent_idx}"
                            assigned_list = assignment_ids.get(key, [])
                            if not assigned_list:
                                continue
                            gid = assigned_list[0]
                            if gid in processed_assignments:
                                continue
                            processed_assignments.add(gid)
                            entries = [f"agent_{agent_idx}"]
                            keep_agent = entries[0]
                            for other_idx in range(num_agents):
                                other_key = f"agent_{other_idx}"
                                if other_key == keep_agent:
                                    continue
                                other_list = assignment_ids.get(other_key, [])
                                if other_list and other_list[0] == gid:
                                    logging.info(
                                        "Assignment %s reserved for %s; %s will explore frontier",
                                        gid,
                                        keep_agent,
                                        other_key,
                                    )
                                    assignment_ids[other_key] = []
                                    subgroup_selections[other_key] = []

                        assigned_info: Dict[str, Dict] = {}
                        subgroup_counter: Counter = Counter()
                        assignment_counter: Counter = Counter()
                        for agent_idx in range(num_agents):
                            key = f"agent_{agent_idx}"
                            assigned_list = assignment_ids.get(key, [])
                            if not assigned_list:
                                continue
                            gid = assigned_list[0]
                            info = lookup.get(gid, {})
                            subgroup_ids = subgroup_selections.get(key, [])
                            subgroup_id = subgroup_ids[0] if subgroup_ids else None
                            centroid = info.get("centroid")
                            assigned_info[key] = {
                                "assignment_id": gid,
                                "centroid": centroid,
                                "subgroup_id": subgroup_id,
                                "sem_group": info.get("sem_group"),
                            }
                            assignment_counter[gid] += 1
                            if subgroup_id is not None:
                                subgroup_counter[subgroup_id] += 1

                        fallback_agents = air_result.get("fallback_agents") or []
                        if fallback_agents:
                            logging.info("Agents falling back to frontier due to low score: %s", fallback_agents)
                            for agent_key in fallback_agents:
                                if agent_key in assignment_ids:
                                    assignment_ids[agent_key] = []
                                if agent_key in subgroup_selections:
                                    subgroup_selections[agent_key] = []
                                if agent_key in assigned_info:
                                    removed = assigned_info.pop(agent_key)
                                    aid = removed.get("assignment_id")
                                    sid = removed.get("subgroup_id")
                                    if aid is not None and aid in assignment_counter:
                                        assignment_counter[aid] -= 1
                                        if assignment_counter[aid] <= 0:
                                            assignment_counter.pop(aid, None)
                                    if sid is not None and sid in subgroup_counter:
                                        subgroup_counter[sid] -= 1
                                        if subgroup_counter[sid] <= 0:
                                            subgroup_counter.pop(sid, None)

                        llm_objects = air_result.get("llm_objects", {})
                        used_assignments: Set[int] = set()
                        used_subgroups: Set[int] = set()
                        used_frontiers: Set[Tuple[int, int]] = set()
                        used_goals: Set[Tuple[int, int]] = set()

                        for agent_idx in range(num_agents):
                            key = f"agent_{agent_idx}"
                            
                            # Skip goal reassignment if agent has already found the target
                            # The agent's act() method will handle navigation to the detected goal
                            if agent_idx in agents_found_goal:
                                logging.info("%s has found the goal; keeping current goal (no AIDE reassignment)", key)
                                new_assignments[agent_idx] = previous_assignments.get(agent_idx)
                                # Ensure goal_points is not overwritten for this agent
                                continue
                            
                            info = assigned_info.get(key)
                            assignment_id = None
                            selected = False
                            if info:
                                subgroup_id = info["subgroup_id"]
                                assignment_id = info["assignment_id"]
                                used_assignments.add(assignment_id)
                                unique_assignment = assignment_counter[assignment_id] == 1
                                unique_subgroup = subgroup_id is not None and subgroup_counter[subgroup_id] == 1
                                already_taken = subgroup_id in used_subgroups if subgroup_id is not None else False
                                if unique_assignment and unique_subgroup and not already_taken:
                                    payload = llm_objects.get(key)
                                    obj = payload.get("object") if payload else None
                                    if obj:
                                        pos = obj.get("map_position", {})
                                        x = pos.get("x")
                                        y = pos.get("y")
                                        if x is not None and y is not None:
                                            gy, gx = int(round(y)), int(round(x))
                                            goal_tuple = (gy, gx)
                                            if goal_tuple in used_goals:
                                                logging.info(
                                                    "LLM object goal %s already taken; %s will fallback",
                                                    goal_tuple, key
                                                )
                                            else:
                                                goal_points[agent_idx] = [gy, gx]
                                                used_goals.add(goal_tuple)
                                                logging.info(
                                                    "Goal assignment [LLM object]: %s -> object %s (%s) at (%d, %d)",
                                                    key,
                                                    obj.get('object_id'),
                                                    obj.get('category'),
                                                    gx,
                                                    gy
                                                )
                                                selected = True
                                                if subgroup_id is not None:
                                                    used_subgroups.add(subgroup_id)
                                    if not selected:
                                        logging.info(
                                            "No LLM object available for %s; will fallback to frontier",
                                            key
                                        )
                                else:
                                    if already_taken:
                                        logging.info(
                                            "Subgroup %s already taken by another agent; %s will explore frontier",
                                            subgroup_id, key
                                        )
                                    if not already_taken and subgroup_id is not None and subgroup_counter[subgroup_id] > 1:
                                        logging.info(
                                            "Subgroup %s assigned to multiple agents; %s will explore frontier",
                                            subgroup_id, key
                                        )
                                    if assignment_counter[assignment_id] > 1:
                                        logging.info(
                                            "Assignment %s shared by multiple agents; %s will explore frontier",
                                            assignment_id, key
                                        )
                                    elif subgroup_id is not None and subgroup_counter[subgroup_id] > 1:
                                        logging.info(
                                            "Subgroup %s assigned to multiple agents; %s will explore frontier",
                                            subgroup_id, key
                                        )
                            new_assignments[agent_idx] = assignment_id if selected else None
                            if not selected:
                                frontier_goal = _nearest_frontier_goal(
                                    agent_idx,
                                    agent_TargetPointMap,
                                    cur_goal_points,
                                    used_frontiers,
                                    global_frontiers=full_target_point_map,
                                )
                                if frontier_goal:
                                    gy, gx = frontier_goal[0], frontier_goal[1]
                                    goal_points[agent_idx] = [gy, gx]
                                    used_goals.add((gy, gx))
                                    logging.info("Goal assignment [frontier fallback]: %s -> (%d, %d)", key, gx, gy)
                                else:
                                    logging.info("No frontier goal available for %s; retaining previous goal", key)

                        assigned_centroids = None

                        agent_assignment_cooldown.clear()
                        for agent_idx in range(num_agents):
                            key = f"agent_{agent_idx}"
                            assigned_list = assignment_ids.get(key, [])
                            if assigned_list:
                                agent_assignment_cooldown[key] = set(assigned_list)

                        deprecated_assignments = used_assignments
                        logging.info("AIRD deprecated assignments (updated): %s", sorted(deprecated_assignments))
                        logging.info("AIRD exhausted assignments (current): %s", sorted(exhausted_assignments))
                    else:
                        agent_assignment_cooldown.clear()
                        deprecated_assignments = set()
                        logging.info("AIRD produced no assignments; reset deprecated set.")
                        
                        # Fallback to frontier-based exploration when AIRD produces no assignments
                        logging.info("Assigning frontier goals to all agents (no AIRD assignments available)")
                        used_frontiers: Set[Tuple[int, int]] = set()
                        for agent_idx in range(num_agents):
                            key = f"agent_{agent_idx}"
                            
                            # Skip goal reassignment if agent has already found the target
                            # The agent's act() method will handle navigation to the detected goal
                            if agent_idx in agents_found_goal:
                                logging.info("%s has found the goal; keeping current goal (no frontier assignment)", key)
                                continue
                            
                            frontier_goal = _nearest_frontier_goal(
                                agent_idx,
                                agent_TargetPointMap,
                                cur_goal_points,
                                used_frontiers,
                                global_frontiers=full_target_point_map,
                            )
                            if frontier_goal:
                                gy, gx = frontier_goal[0], frontier_goal[1]
                                goal_points[agent_idx] = [gy, gx]
                                logging.info("Goal assignment [frontier fallback]: %s -> (%d, %d)", key, gx, gy)
                            else:
                                logging.info("No frontier goal available for %s; retaining previous goal", key)

                    for agent_idx in range(num_agents):
                        prev_gid = previous_assignments.get(agent_idx)
                        new_gid = new_assignments.get(agent_idx)
                        if prev_gid is not None and prev_gid != new_gid and prev_gid not in exhausted_assignments:
                            if not agent[agent_idx].Find_Goal:
                                exhausted_assignments.add(prev_gid)
                                logging.info(
                                    "Group %s exhausted after agent_%d failed to locate the target",
                                    prev_gid,
                                    agent_idx,
                                )
                        agent_active_assignments[agent_idx] = new_gid
                
                else:
                    # ------------------------------------------------------------------
                    #### Fallback: Frontier exploration when AIDE is not enabled
                    # ------------------------------------------------------------------
                    # If AIDE is disabled and not first step, use frontier-based exploration
                    if not is_first_step:
                        logging.info("AIDE not enabled; assigning frontier goals to all agents")
                        used_frontiers: Set[Tuple[int, int]] = set()
                        for agent_idx in range(num_agents):
                            key = f"agent_{agent_idx}"
                            
                            # Skip goal reassignment if agent has already found the target
                            # The agent's act() method will handle navigation to the detected goal
                            if agent_idx in agents_found_goal:
                                logging.info("%s has found the goal; keeping current goal (no frontier assignment)", key)
                                continue
                            
                            frontier_goal = _nearest_frontier_goal(
                                agent_idx,
                                agent_TargetPointMap,
                                cur_goal_points,
                                used_frontiers,
                                global_frontiers=full_target_point_map,
                            )
                            if frontier_goal:
                                gy, gx = frontier_goal[0], frontier_goal[1]
                                goal_points[agent_idx] = [gy, gx]
                                logging.info("Goal assignment [no AIDE fallback]: %s -> (%d, %d)", key, gx, gy)
                            else:
                                logging.info("No frontier goal available for %s; retaining previous goal", key)

                logging.info(f"goal_points: {goal_points}")
                pre_g_points = [gp[:] for gp in goal_points]
                logging.info("===== Global Planning Done ===== ")
                logging.info(
                    "===== END STEP %d (local_planning=%s) =====",
                    agent[0].l_step,
                    is_local_planning_step,
                )
                
                # End decision making timing
                decision_time = time.time() - decision_start
                print(f"Decision Making Phase Time: {decision_time:.4f}s")
            
            

            # Time agent action execution
            action_start = time.time()
            
            # # Debug: Print goal information before agent actions
            # print(f"===== GOAL DEBUG - Step {agent[0].l_step} =====")
            # for i in range(num_agents):
            #     print(f"Agent {i} goal_points: {goal_points[i]}")
            #     if hasattr(agent[i], 'planner_pose_inputs'):
            #         start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
            #         print(f"Agent {i} pose: start=({start_x:.2f}, {start_y:.2f}), orientation={start_o:.2f}")
            #         print(f"Agent {i} planning window: gx1={gx1}, gx2={gx2}, gy1={gy1}, gy2={gy2}")
            # print("="*50)
            
            if target_point:
                logging.info("Goal assignment [shared target_point]: %s applied to all agents", target_point)
                for j in range(num_agents):
                    goal_points[j] = list(target_point)
                    logging.info(
                        "Goal assignment: agent_%d shared target_point -> %s",
                        j,
                        goal_points[j],
                    )

            for i in range(num_agents):
                action[i] = agent[i].act(goal_points[i])
                if 'objectnav_hm3d' in args.task_config:
                    _ = agent_GT[i].act(goal_points[i])
                if action[i] == 0:
                    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                    r, c = start_y, start_x
                    start = [int(r * 100.0 / args.map_resolution - gx1),
                            int(c * 100.0 / args.map_resolution - gy1)]
                    start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)
                    target_point = start.copy()
            # logging.info(f"actions: {action}")
            print(f"Agent get actions time: {time.time() - action_start:.4f}s")
            time_start = time.time()
            observations = env.step(action)
            print(f"Agent Action Execution Time:{time.time() - time_start:.4f}s")
            action_time = time.time() - action_start
            
            # exit(0)
                    
            
                        
            # if count_rotating == 2:
            #     exit(0)
            # ------------------------------------------------------------------

            # Time visualization operations
            vis_start = time.time()
            if args.visualize or args.print_images: 
                if num_agents == 2:
                    vis_ep_dir = '{}/episodes/eps_{}/Agent0_vis'.format(
                        dump_dir, agent[0].episode_n)
                    vis_ep_dir2 = '{}/episodes/eps_{}/Agent1_vis'.format(
                        dump_dir, agent[0].episode_n)
                    if not os.path.exists(vis_ep_dir):
                        os.makedirs(vis_ep_dir)
                    if not os.path.exists(vis_ep_dir2):
                        os.makedirs(vis_ep_dir2)
                    # Legend = cv2.imread("img/legend.png")
                    # height, _ = sem_map.shape[:2]
                    # legend_resized = cv2.resize(Legend, (Legend.shape[1], height))
                    # img_show = np.hstack((sem_map, legend_resized))
                    img_show = observations[0]['rgb'].astype(np.uint8)
                    img_show2 = observations[1]['rgb'].astype(np.uint8)
                    fn = '{}/episodes/eps_{}/Agent0_vis/VisStep-{}.png'.format(
                        dump_dir, agent[0].episode_n,
                        agent[0].l_step)
                    fn2 = '{}/episodes/eps_{}/Agent1_vis/VisStep-{}.png'.format(
                        dump_dir, agent[0].episode_n,
                        agent[0].l_step)
                    # print(fn)
                    cv2.imwrite(fn, img_show)
                    cv2.imwrite(fn2, img_show2)    


                # Visualize(args, agent[0].episode_n, agent[0].l_step, pose_pred, full_map_pred, 
                #         agent[0].goal_id, visited_vis, full_target_edge_map, Frontiers_dict=None, 
                #         goal_points=goal_points)

                # Only visualize even-numbered episodes
                # if (args.visualize or args.print_images) and agent[0].episode_n % 2 == 0:
                if (args.visualize or args.print_images) :
                    Visualize_obj(
                        args,
                        agent[0].episode_n,
                        agent[0].l_step,
                        pose_pred,
                        full_map_pred,
                        agent[0].goal_id,
                        visited_vis,
                        full_target_edge_map,
                        Frontiers_dict=full_Frontiers_dict,
                        goal_points=goal_points,
                        object_positions=object_positions,
                        tracked_objects=map_manager.tracked_objects,
                        assigned_centroids=None,
                        frontier_points=full_target_point_map,
                    )
                    # assigned_centroids=assigned_centroids)
                
                # Clear object states after visualization to prevent state persistence
                map_manager.clear_object_states()

                # exit(0)
            
            # End visualization timing
            vis_time = time.time() - vis_start
            print(f"Visualization Operations Time: {vis_time:.4f}s")
            
            # Total step timing
            total_step_time = time.time() - step_start_time
            print(f"TOTAL STEP TIME: {total_step_time:.4f}s")
            print(f"Step Breakdown:")
            print(f"  - Agent Mapping: {mapping_time:.4f}s ({mapping_time/total_step_time*100:.1f}%)")
            print(f"  - Map Processing: {map_processing_time:.4f}s ({map_processing_time/total_step_time*100:.1f}%)")
            # print(f"  - Grouper Operations: {total_grouper_time:.4f}s ({total_grouper_time/total_step_time*100:.1f}%)")
            if is_local_planning_step:
                print(f"  - Object Extraction: {obj_extract_time:.4f}s ({obj_extract_time/total_step_time*100:.1f}%)")
                print(f"  - Decision Making: {decision_time:.4f}s ({decision_time/total_step_time*100:.1f}%)")
            print(f"  - Action Execution: {action_time:.4f}s ({action_time/total_step_time*100:.1f}%)")
            print(f"  - Visualization: {vis_time:.4f}s ({vis_time/total_step_time*100:.1f}%)")
            print("="*60)
            

            # logging.info(f"full_map_pred.shape: {full_map_pred.shape}") # [20,480,480] HM-3D

##############################################===Metrics===##############################################

        count_episodes += 1
        # obj_SR['num_'+agent[0].goal_name] += 1
        count_step += agent[0].l_step

        # ------------------------------------------------------------------
        ##### Logging
        # ------------------------------------------------------------------
        log_end = time.time()
        time_elapsed = time.gmtime(log_end - log_start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_step),
            "FPS {},".format(int(count_step / (log_end - log_start)))
        ]) + '\n'

        # Set SR to 0 if unknown tags are present
        if agent[0].goal_id + 4 > 24:
            log += '==========Unknown Label=========='
            log += '\n'
            for k, v in agg_metrics.items():
                if k == 'multi_Total_SR':
                    for i in range(num_agents):
                        if 'objectnav_hm3d' in args.task_config:
                            if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                                agg_metrics[k] += 1
                                if agg_metrics[k] > count_episodes:
                                    agg_metrics[k] = count_episodes
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
                            elif agent[i].Find_Goal and agent_GT[i].Find_Goal == False:
                                agg_metrics[k] += 0
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
                        else:
                            if agent[i].Find_Goal:
                                agg_metrics[k] += 1
                                if agg_metrics[k] > count_episodes:
                                    agg_metrics[k] = count_episodes
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break

            gt_semantic_success = any(
                agent_near_gt_target(agent[i], args)
                for i in range(num_agents)
            )
            if gt_semantic_success:
                agg_metrics['multi_GTCategory_SR'] += 1
                if agg_metrics['multi_GTCategory_SR'] > count_episodes:
                    agg_metrics['multi_GTCategory_SR'] = count_episodes

            spls = []
            for i in range(num_agents):
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                r, c = start_y, start_x
                start = [int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1)]
                start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)

                # Ensure Start_Location is set at the first valid pose window
                if getattr(agent[i], 'Start_Location', None) is None and start is not None:
                    agent[i].Start_Location = start

                # Ensure Start_Location is set at the first valid pose window
                if getattr(agent[i], 'Start_Location', None) is None and start is not None:
                    agent[i].Start_Location = start
                if 'objectnav_hm3d' in args.task_config:
                    if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                        spl = agent[i].get_spl(success=1,cur_loc=start)
                    else:
                        spl = agent[i].get_spl(success=0,cur_loc=start)
                else:
                    if agent[i].Find_Goal:
                        spl = agent[i].get_spl(success=1,cur_loc=start)
                    else:
                        spl = agent[i].get_spl(success=0,cur_loc=start)

                # Skip if SPL could not be computed (e.g., missing Start_Location)
                if spl is None:
                    continue

                agg_metrics['multi_SPL'][f'Agent_{i}'] = spl
                agg_metrics['multi_SoftSPL'][f'Agent_{i}'] += spl
                agg_metrics['multi_SPL_valid'][f'Agent_{i}'] += 1
                spls.append(spl)

            # Handle case where all agents were skipped this episode
            if spls:
                agg_metrics['SPL'] = max(spls)
                agg_metrics['SoftSPL'] += max(spls)
                agg_metrics['SPL_valid'] += 1
            else:
                agg_metrics['SPL'] = 0.0
            for agent_name, SPL in agg_metrics['multi_SPL'].items():
                valid_cnt = int(agg_metrics['multi_SPL_valid'].get(agent_name, 0))
                denom = valid_cnt if valid_cnt > 0 else 1
                SoftSPL = agg_metrics['multi_SoftSPL'][agent_name] / denom
                log += f"{agent_name}" + "---SPL: {:.3f}, SoftSPL: {:.3f}".format(SPL, SoftSPL)
                log += '\n'

            log += "multi_Total_SR: {:.3f}, ".format(agg_metrics['multi_Total_SR'] / count_episodes)
            log += "multi_Navigation_SR/SR: {:.0f}/{:.0f}, ".format(agg_metrics['multi_Navigation_SR'], agg_metrics['multi_Total_SR'])
            log += "multi_GTCategory_SR: {:.3f}, ".format(agg_metrics['multi_GTCategory_SR'] / count_episodes)
            log += "multi_SPL: {:.3f}, ".format(agg_metrics['SPL'])
            valid_episodes = int(agg_metrics['SPL_valid']) if int(agg_metrics['SPL_valid']) > 0 else 1
            log += "multi_SoftSPL: {:.3f}, ".format(agg_metrics['SoftSPL'] / valid_episodes)
            log += "Skipped: {:.0f} ".format(agg_metrics['Skipped_Episodes'])
            log += " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)
        else:
            # metrics = env.get_metrics()

            for k, v in agg_metrics.items():
                if k == 'multi_Total_SR':
                    for i in range(num_agents):
                        if 'objectnav_hm3d' in args.task_config:
                            pdb.set_trace()
                            if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                                agg_metrics[k] += 1
                                if agg_metrics[k] > count_episodes:
                                    agg_metrics[k] = count_episodes
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
                            elif agent[i].Find_Goal and agent_GT[i].Find_Goal == False:
                                agg_metrics[k] += 0
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
                        else:
                            if agent[i].Find_Goal:
                                agg_metrics[k] += 1
                                if agg_metrics[k] > count_episodes:
                                    agg_metrics[k] = count_episodes
                                agg_metrics['multi_Navigation_SR'] += 1
                                if agg_metrics['multi_Navigation_SR'] > count_episodes:
                                    agg_metrics['multi_Navigation_SR'] = count_episodes
                                break
            gt_semantic_success = any(
                agent_near_gt_target(agent[i], args)
                for i in range(num_agents)
            )
            if gt_semantic_success:
                agg_metrics['multi_GTCategory_SR'] += 1
                if agg_metrics['multi_GTCategory_SR'] > count_episodes:
                    agg_metrics['multi_GTCategory_SR'] = count_episodes

            spls = []
            for i in range(num_agents):
                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = agent[i].planner_pose_inputs
                r, c = start_y, start_x
                start = [int(r * 100.0 / args.map_resolution - gx1),
                int(c * 100.0 / args.map_resolution - gy1)]
                start = pu.threshold_poses(start, agent[i].local_map[0, :, :].cpu().numpy().shape)
                if 'objectnav_hm3d' in args.task_config:
                    if agent[i].Find_Goal and agent_GT[i].Find_Goal:
                        spl = agent[i].get_spl(success=1,cur_loc=start)
                    else:
                        spl = agent[i].get_spl(success=0,cur_loc=start)
                else:
                    if agent[i].Find_Goal:
                        spl = agent[i].get_spl(success=1,cur_loc=start)
                    else:
                        spl = agent[i].get_spl(success=0,cur_loc=start)
                # Skip if SPL could not be computed (e.g., missing Start_Location)
                if spl is None:
                    continue

                agg_metrics['multi_SPL'][f'Agent_{i}'] = spl
                agg_metrics['multi_SoftSPL'][f'Agent_{i}'] += spl
                agg_metrics['multi_SPL_valid'][f'Agent_{i}'] += 1
                spls.append(spl)

            # Handle case where all agents were skipped this episode
            if spls:
                agg_metrics['SPL'] = max(spls)
                agg_metrics['SoftSPL'] += max(spls)
                agg_metrics['SPL_valid'] += 1
            else:
                agg_metrics['SPL'] = 0.0
            for agent_name, SPL in agg_metrics['multi_SPL'].items():
                valid_cnt = int(agg_metrics['multi_SPL_valid'].get(agent_name, 0))
                denom = valid_cnt if valid_cnt > 0 else 1
                SoftSPL = agg_metrics['multi_SoftSPL'][agent_name] / denom
                log += f"{agent_name}" + "---SPL: {:.3f}, SoftSPL: {:.3f}".format(SPL, SoftSPL)
                log += '\n'

            log += "multi_Total_SR: {:.3f}, ".format(agg_metrics['multi_Total_SR'] / count_episodes)
            log += "multi_Navigation_SR/SR: {:.0f}/{:.0f}, ".format(agg_metrics['multi_Navigation_SR'], agg_metrics['multi_Total_SR'])
            log += "multi_GTCategory_SR: {:.3f}, ".format(agg_metrics['multi_GTCategory_SR'] / count_episodes)
            log += "multi_SPL: {:.3f}, ".format(agg_metrics['SPL'])
            valid_episodes = int(agg_metrics['SPL_valid']) if int(agg_metrics['SPL_valid']) > 0 else 1
            log += "multi_SoftSPL: {:.3f}, ".format(agg_metrics['SoftSPL'] / valid_episodes)
            log += "Skipped: {:.0f} ".format(agg_metrics['Skipped_Episodes'])
            log += " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)
        # log += "Total usage: " + str(sum(total_usage)) + ", average usage: " + str(np.mean(total_usage))
        # print(log)
        
        # ------------------------------------------------------------------
        # LLM Timing Summary for Episode
        # ------------------------------------------------------------------
        logging.info("="*50)
        logging.info(f"EPISODE {count_episodes} LLM TIMING SUMMARY:")
        logging.info("="*50)
        
        # Note: Individual LLM timing logs are already printed during execution
        # This provides a summary at the end of each episode
        logging.info("LLM timing breakdown:")
        logging.info("- Scene Information: Initial scene understanding")
        logging.info("- Perception: Object and scene analysis") 
        logging.info("- Judgment: Frontier evaluation and decision making")
        logging.info("- Decision: Final navigation choice")
        logging.info("="*50)
        
        logging.info(log)
        fn = '{}/MCoCoNav_history.log'.format(log_dir)
        if count_episodes == 1:
            with open(fn,'w', encoding='utf-8') as f:
                f.write(log)
                f.write('\n')
        else:
            with open(fn,'a', encoding='utf-8') as f:
                f.write(log)
                f.write('\n')
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # LLM Server Keepalive - Prompt every 10 seconds with last prompt
    # ------------------------------------------------------------------
    
    # Define a simple last prompt for keepalive
    last_prompt = "what is 1+2? only give the answer, no other words."
    last_messages = [
        {
            "role": "system",
            "content": "You help robots choose navigation targets. Always reply with a single integer object_id and nothing else.",
        },
        {
            "role": "user",
            "content": last_prompt,
        },
    ]
    
    logging.info("="*50)
    logging.info("Starting LLM server keepalive (10 second intervals)")
    logging.info("="*50)
    

    while True:
        # time.sleep(random.uniform(5, 10))
        time.sleep(10)
        for idx, llm_client in enumerate(cogvlm_clients):
            try:
                _, response = llm_client.create_chat_completion(
                    "cogvlm2",
                    messages=last_messages,
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=256,
                    use_stream=False,
                )
                logging.info(f"[Keepalive] LLM client {idx} responded: {response[:50] if response else 'None'}")
            except Exception as exc:
                logging.warning(f"[Keepalive] LLM client {idx} failed: {exc}")

    
    # ------------------------------------------------------------------
    
    # avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    # return avg_metrics
    

if __name__ == "__main__":
    main()
