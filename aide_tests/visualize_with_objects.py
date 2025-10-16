# 画出所有的Frontier和检测到的物体
def Visualize(args, episode_n, l_step, pose_pred, full_map_pred, goal_name, visited_vis, map_edge, Frontiers_dict, goal_points, object_positions=None, tracked_objects=None):
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
                label = f"{alpha[alpha0]}"
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
            if 'object_id' in obj:
                # This is a tracked object (existing or updated)
                if 'merged_from' in obj:
                    # This is a merged/updated object
                    color = color_updated_object
                    label_prefix = "U"
                else:
                    # This is an existing tracked object
                    color = color_existing_object
                    label_prefix = "E"
            else:
                # This is a completely new object
                color = color_new_object
                label_prefix = "N"
            
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
