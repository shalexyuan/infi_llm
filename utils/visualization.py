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
