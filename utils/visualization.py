import cv2
import numpy as np
from constants import color_palette
from matplotlib import pyplot as plt

def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])

def save_legend(categories):
    full_cat = ['Unexplored','Obstacle','Explored','Trajectory','Goal'] + categories
    colors = np.array(color_palette).reshape(-1, 3)
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                             markerfacecolor=color, markersize=10) for cat, color in zip(full_cat, colors[:len(full_cat)-1])]

    # Display the legend
    plt.legend(handles=legend_handles, loc='center')

    # To remove the x and y axis labels and ticks
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(4/3,12.0/3) #dpi = 300
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    fig.savefig("img/legend.png", format='png', transparent=True, dpi=300, pad_inches = 0, bbox_inches="tight")


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image(goal_name, action):
    vis_image = np.ones((537, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations" 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Find {}  Action {}".format(goal_name, str(action))
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
    vis_image[49, 15:655] = color
    vis_image[49, 670:1150] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 655] = color
    vis_image[50:530, 669] = color
    vis_image[50:530, 1150] = color
    vis_image[530, 15:655] = color
    vis_image[530, 670:1150] = color


#     # draw legend
#     lx, ly, _ = legend.shape
#     vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image

def init_multi_vis_image(goal_name, multi_color):
    vis_image = np.ones((537, 670, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Find {}".format(goal_name) 
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 50
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    for i in range(len(multi_color)):
        text = "Agent {}".format(i) 
        vis_image = cv2.putText(vis_image, text, (textX+200+150*i, textY),
                                font, fontScale, multi_color[i], thickness,
                                cv2.LINE_AA)
    # draw outlines
    color = [100, 100, 100]
    vis_image[49, 15:600] = color
    vis_image[50:530, 14] = color
    vis_image[50:530, 600] = color
    vis_image[530, 15:600] = color


#     # draw legend
#     lx, ly, _ = legend.shape
#     vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image


def visualize_localmap(inputs, save_path):
    args = self.args


    local_w = inputs['map_pred'].shape[0]

    map_pred = inputs['map_pred']
    exp_pred = inputs['exp_pred']
    map_edge = inputs['map_edge']
    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

    goal = inputs['goal']
    sem_map = inputs['sem_map_pred']

    gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

    sem_map += 5

    no_cat_mask = sem_map == 20
    map_mask = np.rint(map_pred) == 1
    exp_mask = np.rint(exp_pred) == 1
    edge_mask = map_edge == 1

    sem_map[no_cat_mask] = 0
    m1 = np.logical_and(no_cat_mask, exp_mask)
    sem_map[m1] = 2

    m2 = np.logical_and(no_cat_mask, map_mask)
    sem_map[m2] = 1

    sem_map[edge_mask] = 3

    selem = skimage.morphology.disk(4)
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal, selem) != True

    goal_mask = goal_mat == 1
    sem_map[goal_mask] = 4
    if np.sum(goal) == 1:
        f_pos = np.argwhere(goal == 1)
        # fmb = get_frontier_boundaries((f_pos[0][0], f_pos[0][1]))
        # goal_fmb = skimage.draw.circle_perimeter(int((fmb[0]+fmb[1])/2), int((fmb[2]+fmb[3])/2), 23)
        goal_fmb = skimage.draw.circle_perimeter(f_pos[0][0], f_pos[0][1], int(local_w/8 -1))
        goal_fmb[0][goal_fmb[0] > local_w-1] = local_w-1
        goal_fmb[1][goal_fmb[1] > local_w-1] = local_w-1
        goal_fmb[0][goal_fmb[0] < 0] = 0
        goal_fmb[1][goal_fmb[1] < 0] = 0
        # goal_fmb[goal_fmb < 0] =0
        goal_mask[goal_fmb[0], goal_fmb[1]] = 1
        sem_map[goal_mask] = 4


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

    vis_image = vu.init_vis_image("test", "null")

    vis_image[50:530, 15:655] = self.rgb_vis
    vis_image[50:530, 670:1150] = sem_map_vis

    pos = (
        (start_x * 100. / args.map_resolution - gy1)
        * 480 / map_pred.shape[0],
        (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
        * 480 / map_pred.shape[1],
        np.deg2rad(-start_o)
    )

    agent_arrow = vu.get_contour_points(pos, origin=(670, 50), size=10)
    color = (int(color_palette[11] * 255),
                int(color_palette[10] * 255),
                int(color_palette[9] * 255))
    cv2.drawContours(vis_image, [agent_arrow], 0, color, -1)

    cv2.imwrite(save_path, vis_image)
    # if args.print_images:
    #     fn = '{}/episodes/eps_{}/agent-{}-Vis-{}.png'.format(
    #         dump_dir, self.episode_n,
    #         self.agent_id, self.l_step)
    #     cv2.imwrite(fn, self.vis_image)


def visualize_semantic_map(full_map, object_category, save_path=None, map_size=480):
    """
    Visualize the full semantic map with different colors for different semantic categories.
    
    Args:
        full_map: torch.Tensor of shape [channels, height, width] where:
                 - Channel 0: Obstacle Map
                 - Channel 1: Explored Area  
                 - Channel 2: Current Agent Location
                 - Channel 3: Past Agent Locations
                 - Channels 4+: Semantic Categories
        object_category: List of semantic category names
        save_path: Optional path to save the visualization
        map_size: Size of the output visualization
    
    Returns:
        vis_image: RGB image of the semantic map visualization
    """
    import torch
    import numpy as np
    from PIL import Image
    from constants import color_palette
    
    # Convert to numpy if it's a torch tensor
    if isinstance(full_map, torch.Tensor):
        full_map = full_map.cpu().numpy()
    
    # Get map dimensions
    channels, height, width = full_map.shape
    
    # Create the semantic map visualization
    # Initialize with unexplored color (white)
    semantic_map = np.zeros((height, width), dtype=np.uint8)
    
    # Channel 0: Obstacle Map (gray)
    obstacle_mask = full_map[0] > 0.5
    semantic_map[obstacle_mask] = 1
    
    # Channel 1: Explored Area (light gray) - only where not obstacle
    explored_mask = (full_map[1] > 0.5) & (~obstacle_mask)
    semantic_map[explored_mask] = 2
    
    # Channel 2: Current Agent Location (red)
    current_agent_mask = full_map[2] > 0.5
    semantic_map[current_agent_mask] = 3
    
    # Channel 3: Past Agent Locations (blue)
    past_agent_mask = full_map[3] > 0.5
    semantic_map[past_agent_mask] = 4
    
    # Channels 4+: Semantic Categories
    for ch in range(4, min(channels, 4 + len(object_category))):
        semantic_mask = full_map[ch] > 0.5
        # Use different colors for each semantic category
        category_id = ch - 4
        if category_id < len(object_category):
            semantic_map[semantic_mask] = 5 + category_id
    
    # Create color palette for visualization
    # Base colors: Unexplored, Obstacle, Explored, Current Agent, Past Agent, + Semantic Categories
    num_categories = len(object_category)
    total_colors = 5 + num_categories
    
    # Ensure we have enough colors in the palette
    if len(color_palette) < total_colors * 3:
        # Extend palette if needed
        extended_palette = list(color_palette)
        while len(extended_palette) < total_colors * 3:
            extended_palette.extend([0.5, 0.5, 0.5])  # Default gray
        color_pal = [int(x * 255.) for x in extended_palette[:total_colors * 3]]
    else:
        color_pal = [int(x * 255.) for x in color_palette[:total_colors * 3]]
    
    # Create PIL image with palette
    sem_map_vis = Image.new("P", (width, height))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
    
    # Convert to RGB
    sem_map_vis = sem_map_vis.convert("RGB")
    sem_map_vis = np.flipud(sem_map_vis)  # Flip vertically for proper orientation
    
    # Convert BGR to RGB for OpenCV compatibility
    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
    
    # Resize to desired size
    sem_map_vis = cv2.resize(sem_map_vis, (map_size, map_size), interpolation=cv2.INTER_NEAREST)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, sem_map_vis)
        print(f"Semantic map visualization saved to: {save_path}")
    
    return sem_map_vis


def create_semantic_map_legend(object_category, save_path=None):
    """
    Create a legend for the semantic map visualization.
    
    Args:
        object_category: List of semantic category names
        save_path: Optional path to save the legend
    
    Returns:
        legend_image: RGB image of the legend
    """
    from constants import color_palette
    
    # Create legend categories
    legend_categories = ['Unexplored', 'Obstacle', 'Explored', 'Current Agent', 'Past Agent'] + object_category
    num_categories = len(legend_categories)
    
    # Create legend image
    legend_height = max(400, num_categories * 25 + 50)
    legend_width = 300
    legend_image = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
    
    # Get colors
    if len(color_palette) < num_categories * 3:
        extended_palette = list(color_palette)
        while len(extended_palette) < num_categories * 3:
            extended_palette.extend([0.5, 0.5, 0.5])
        colors = np.array(extended_palette[:num_categories * 3]).reshape(-1, 3)
    else:
        colors = np.array(color_palette[:num_categories * 3]).reshape(-1, 3)
    
    # Draw legend items
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_color = (0, 0, 0)
    
    for i, (category, color) in enumerate(zip(legend_categories, colors)):
        y_pos = 30 + i * 25
        
        # Draw color square
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        cv2.rectangle(legend_image, (20, y_pos - 8), (40, y_pos + 8), color_bgr, -1)
        cv2.rectangle(legend_image, (20, y_pos - 8), (40, y_pos + 8), (0, 0, 0), 1)
        
        # Draw text
        cv2.putText(legend_image, category, (50, y_pos + 5), font, font_scale, text_color, thickness)
    
    # Add title
    cv2.putText(legend_image, "Semantic Map Legend", (10, 20), font, 0.7, (0, 0, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, legend_image)
        print(f"Legend saved to: {save_path}")
    
    return legend_image
