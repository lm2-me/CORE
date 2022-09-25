import ghhops_server as hs
import rhino3dm as r3d
import prog_place.helpers as h
import math as m
import numpy as np
import copy

#see app.py file for description of each gh node and more specific information regarding the inputs and outputs

def divide_site(site: r3d.Surface, road_lines_tree, sidewalk_lines_tree, grid_size: float):
    sidewalk_lines: list[r3d.Line] = sidewalk_lines_tree['{0}']
    road_lines: list[r3d.Line] = road_lines_tree['{0}']

    srfpts = []
    values = []
    
    srf_u_start = int(site.Domain(0).T0)
    srf_u_end = int(site.Domain(0).T1)
    srf_v_start = int(site.Domain(1).T0)
    srf_v_end = int(site.Domain(1).T1)

    srf_btm_left_corner = site.PointAt(srf_u_start, srf_v_start)

    print('pre-loop')

    for y in range(srf_v_end, srf_v_start-int(grid_size),  -int(grid_size)):
        rowpts = []
        rowpts_vals = []
        for x in range(srf_u_start, srf_u_end+int(grid_size), int(grid_size)):
            road_weight = m.inf
            sidewalk_weight = m.inf
            point = r3d.Point3d(site.PointAt(x,y).X,site.PointAt(x,y).Y,0)
            rowpts.append(point)

            for r in road_lines:
                new_road_weight = h.line_closest_point(r, point)
                if new_road_weight < road_weight:
                    road_weight = new_road_weight    

            for s in sidewalk_lines:
                new_sidewalk_weight = h.line_closest_point(s, point)
                if new_sidewalk_weight < sidewalk_weight:
                    sidewalk_weight = new_sidewalk_weight
                        
            rowpts_vals.append('{0},{1}'.format(road_weight, sidewalk_weight))
        srfpts.append(rowpts)
        values.append(rowpts_vals)
    
    label_array_np = np.full_like(np.array(srfpts), 's', str)

    site_grid, site_edge  = h.divide_surface(site, grid_size)
    for row, list in enumerate(srfpts):
        for column1, pt1 in enumerate(list):
            for pt2 in site_edge:
                distance = abs(m.dist((pt1.X, pt1.Y, pt1.Z), (pt2.X, pt2.Y, pt2.Z)))
                if distance < 0.05:
                    label_array_np[row][column1] = 'e'
                else:
                    continue
    
    return h.list_to_tree(srfpts), h.list_to_tree(values), h.list_to_tree(label_array_np.tolist())

def place_packages(srfpts_tree, cost_function_tree, labels_tree, module_use, module_geometry: r3d.Surface, module_mask):

    ### width and height of packages module
    point1 = module_geometry.PointAt(int(module_geometry.Domain(0).T0), int(module_geometry.Domain(1).T0))
    point2 = module_geometry.PointAt(int(module_geometry.Domain(0).T1), int(module_geometry.Domain(1).T1))

    width = int(abs(point1.X - point2.X))
    height = int(abs(point1.Y - point2.Y))
    rotation = 0
    center = r3d.Point3d(width/2, height/2, 0)
    axis = h.vector3d_2pts(center, r3d.Point3d(center.X, center.Y, 10))
    _, tp_left, tp_right, _,_ = h.corners(module_geometry)
    move_from = tp_left

    ### convert cost function and srf points information from gh trees to np array
    srfpts_matrix = np.array(h.tree_to_matrix(srfpts_tree))
    grid_size = abs(srfpts_matrix[0,0].X - srfpts_matrix[0,1].X)
    cost_function_matrix = h.matrix_str2floats(h.tree_to_matrix(cost_function_tree))
    out_cost_function_matrix = copy.deepcopy(cost_function_matrix)

    ### generate array of labels for each point
    ### s - site
    ### e - site edge
    ### b - boundary
    ### i - interior
    ### x - outside of site edge
    label_array_np = np.array(h.tree_to_matrix(labels_tree))

    ### convert module cost function information from strings to np array 
    module_mask_np = np.array([int(v) for v in module_mask.split(',')])

    ### mask cost function with module parameters
    module_masked_cost = np.dot(np.array(cost_function_matrix), module_mask_np)

    ### remove locations that place packages module outside of the site area
    ### TODO: update to incorporate rotations 0, 90, 180, 270 deg confirm mask is correct for each rotation
    min_cost_value = []
    masked_cost_placement_zerodeg = np.copy(module_masked_cost)
    masked_cost_placement_zerodeg[width:, :] = m.inf
    masked_cost_placement_zerodeg[:, height:] = m.inf
    min_cost_value.append([masked_cost_placement_zerodeg, masked_cost_placement_zerodeg.min(), 0, tp_left])

    masked_cost_placement_ninetydeg = np.copy(module_masked_cost)
    masked_cost_placement_ninetydeg[height:, :] = m.inf
    masked_cost_placement_ninetydeg[:, width:] = m.inf
    min_cost_value.append([masked_cost_placement_ninetydeg, masked_cost_placement_ninetydeg.min(), 1.5708, tp_right])

    masked_cost_placement_onetwentydeg = np.copy(module_masked_cost)
    masked_cost_placement_onetwentydeg[:height, :] = m.inf
    masked_cost_placement_onetwentydeg[:, :width] = m.inf
    min_cost_value.append([masked_cost_placement_onetwentydeg, masked_cost_placement_onetwentydeg.min(), 3.14159, tp_left])
 
    masked_cost_placement_twoseventydeg = np.copy(module_masked_cost)
    masked_cost_placement_twoseventydeg[:width, :] = m.inf
    masked_cost_placement_twoseventydeg[:, :height] = m.inf
    min_cost_value.append([masked_cost_placement_twoseventydeg, masked_cost_placement_twoseventydeg.min(), 4.712392, tp_right])

    min_value = m.inf

    for func, value, module_rotation, move_pt in min_cost_value:
        if value < min_value:
            min_value = value
            masked_cost_placement = func
            rotation = module_rotation
            move_from = move_pt

    best_points_locs = np.where(masked_cost_placement == masked_cost_placement.min())
    best_points = [(x,y) for y,x in zip(best_points_locs[0], best_points_locs[1])]
    best_points.sort()
    best_point = best_points[int(len(best_points) / 2)]

    top_left_corner = srfpts_matrix[best_point[1], best_point[0]]

    success_rot = module_geometry.Rotate(rotation, axis, tp_left)
    rotation_transform = r3d.Transform.Rotation(rotation, axis, tp_left)
    print('rotate packages module?', success_rot)

    rot_mat_flat = r3d.Transform.ToFloatArray(rotation_transform, True)
    rot_mat = np.asarray(rot_mat_flat).reshape(4, 4)
    move_from_np = np.array([move_from.X, move_from.Y, move_from.Z, 1])
    move_from_new = np.dot(rot_mat, move_from_np.T)
    move_from = r3d.Point3d(move_from_new[0], move_from_new[1], 0)

    move_vector = h.vector3d_2pts(move_from, top_left_corner)
    success_move = module_geometry.Translate(move_vector)
    print('moved packages module?', success_move)

    module_grid, module_edge  = h.divide_surface(module_geometry, grid_size)
    module_grid_np = np.array(module_grid)
  
    for row, list in enumerate(srfpts_matrix):
        for column1, pt1 in enumerate(list):
            for i in module_grid_np:
                for j, pt2 in enumerate(i):
                    distance = abs(m.dist((pt1.X, pt1.Y, pt1.Z), (pt2.X, pt2.Y, pt2.Z)))
                    if distance < 0.05:
                        out_cost_function_matrix[row][column1][0] = str(m.inf)
                        out_cost_function_matrix[row][column1][1] = str(m.inf)
                        label_array_np[row][column1] = 'i'
                    else:
                        continue
    
    label_array_np = h.update_label(label_array_np, srfpts_matrix, module_edge, 'b')

    label_array = label_array_np.tolist()

    return module_geometry, h.list_to_tree(h.matrix_floats2str(out_cost_function_matrix)), module_edge, h.list_to_tree(label_array)

def place_modules(srfpts_tree, cost_function_tree, label_array, module_use_tree, module_geometry_list, module_mask_tree):
    ### convert cost function and srf points information from gh trees to np array
    srfpts_matrix = np.array(h.tree_to_matrix(srfpts_tree))
    grid_size = abs(srfpts_matrix[0,0].X - srfpts_matrix[0,1].X)
    module_mask_matrix = h.tree_to_matrix(module_mask_tree)
    cost_function_matrix = h.matrix_str2floats(h.tree_to_matrix(cost_function_tree))
    label_array_matrix = np.array(h.tree_to_matrix(label_array))

    ### convert module cost function information from strings to np array 
    module_mask_np = []

    for i in module_mask_matrix:
        module_mask_np.append(np.array([int(v) for v in i[0].split(',')]))

    ### mask cost function with module parameters
    module_masked_cost = []
    for i, row in enumerate(module_mask_matrix):
        module_masked_cost.append(np.dot(np.array(cost_function_matrix), module_mask_np[i]))

    label_array_matrix_revised = copy.deepcopy(label_array_matrix)

    out_module_edges = []
    #place each module, update label array,
    for num, geo in enumerate(module_geometry_list):
        use_cost_function_matrix = copy.deepcopy(module_masked_cost[num])

        # if isinstance(use_cost_function_matrix[0][0], list):
        #     for i, row in enumerate(use_cost_function_matrix):
        #         for j, values in enumerate(row):
        #             sum = 0
        #             for k, value in enumerate(values):
        #                 sum += use_cost_function_matrix[i][j][k]
        #             use_cost_function_matrix[i][j] = sum

        for i, row in enumerate(label_array_matrix):
            for j, label in enumerate(row):
                if label == 's' or label == 'i' or label == 'e' or label == 'x':
                    use_cost_function_matrix[i][j] = m.inf

        points_to_order = []
        values_to_order = []
        for i, row in enumerate(use_cost_function_matrix):
            for j, cost in enumerate(row):
                if cost != m.inf:
                    values_to_order.append(cost)
                    points_to_order.append(srfpts_matrix[i,j])

        ### order best locations in order best to worst 
        values_ordered = np.array(values_to_order)
        points_ordered = np.array(points_to_order)
        idx   = np.argsort(values_ordered)

        values_ordered = values_ordered[idx]
        points_ordered = points_ordered[idx]

        valid_placement = False

        for i, corner in enumerate(points_ordered):
            _, tp_left, _, _,_ = h.corners(geo)

            #move geometry to point
            move_vector = h.vector3d_2pts(tp_left, corner)
            success_move = geo.Translate(move_vector)
            print('move module first time?', success_move)
            
            _, mvd_tp_left, _, _, _ = h.corners(geo)

            for j in range(4):
                
                _, _, _, _, corners_list = h.corners(geo)
                corner_locations = h.location_closest_grid_point(srfpts_matrix, corners_list)

                corner_labels = []
                for c in corner_locations:
                    if c != None:
                        corner_labels.append(label_array_matrix_revised[c[0],c[1]])
                    else:
                        corner_labels.append(None)

                corner_labels_np = np.array(corner_labels)

                if (np.isin('i', corner_labels_np) or
                    np.isin('x', corner_labels_np) or
                    np.isin(None, corner_labels_np) or
                    np.count_nonzero(corner_labels_np == 'b') < 2
                ):
                    
                    axis = h.vector3d_2pts(mvd_tp_left, r3d.Point3d(mvd_tp_left.X, mvd_tp_left.Y, 10))
                    success_rot = geo.Rotate(m.pi / 2., axis, mvd_tp_left)
                    # print('rotate module?', success_rot)
                else:
                    valid_placement = True
                    # print('valid placement 2: ', valid_placement)
                    break
                    
            if valid_placement:
                break

            print("no valid position for this module at this location")
        
        if not valid_placement:
            print("no valid position for this module")

        module_grid, module_edge  = h.divide_surface(geo, grid_size)
        ### list of module edges to output
        out_module_edges.append(module_edge)

        ### update label array matrix
        label_array_matrix_revised = h.update_label(label_array_matrix_revised, srfpts_matrix, module_edge, 'b')
        label_array_matrix_revised = h.update_label(label_array_matrix_revised, srfpts_matrix, module_grid, 'i')
        ### check if diagonal of each b contains at least one 's' if yes, it is still a boundary, if no, then it becomes an 'i'
        label_array_matrix_revised = h.convert_interior_boundaries(label_array_matrix_revised)

        print(label_array_matrix_revised)

    #return "surface", [excluded] "cost", "module edges", "label"
    return module_geometry_list, h.list_to_tree(out_module_edges), h.list_to_tree(label_array_matrix_revised.tolist())