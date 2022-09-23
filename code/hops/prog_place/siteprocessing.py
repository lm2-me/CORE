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

    return h.list_to_tree(srfpts), h.list_to_tree(values)

def place_packages(srfpts_tree, cost_function_tree, module_use, module_geometry: r3d.Surface, module_mask):

    ### width and height of packages module
    point1 = module_geometry.PointAt(int(module_geometry.Domain(0).T0), int(module_geometry.Domain(1).T0))
    point2 = module_geometry.PointAt(int(module_geometry.Domain(0).T1), int(module_geometry.Domain(1).T1))

    width = int(abs(point1.X - point2.X))
    height = int(abs(point1.Y - point2.Y))
    rotation = 0
    center = r3d.Point3d(width/2, height/2, 0)
    axis = h.vector3d_2pts(center, r3d.Point3d(center.X, center.Y, 10))
    btm_left, tp_left, tp_right, btm_right = h.corners(module_geometry)
    move_from = tp_left

    ### convert cost function and srf points information from gh trees to np array
    srfpts_matrix = np.array(h.tree_to_matrix(srfpts_tree))
    grid_size = abs(srfpts_matrix[0,0].X - srfpts_matrix[0,1].X)
    cost_function_matrix = h.matrix_str2floats(h.tree_to_matrix(cost_function_tree))
    out_cost_function_matrix = copy.deepcopy(cost_function_matrix)

    ### generate array of lables for each point
    lable_array_np = np.full_like(srfpts_matrix, 's', str)

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
                        lable_array_np[row][column1] = 'i'
                    else:
                        continue
    
    for row, list in enumerate(srfpts_matrix):
        for column1, pt1 in enumerate(list):
            for pt2 in module_edge:
                distance = abs(m.dist((pt1.X, pt1.Y, pt1.Z), (pt2.X, pt2.Y, pt2.Z)))
                if distance < 0.05:
                    lable_array_np[row][column1] = 'b'
                else:
                    continue
    
    lable_array = lable_array_np.tolist()

    return module_geometry, h.list_to_tree(h.matrix_floats2str(out_cost_function_matrix)), module_edge, h.list_to_tree(lable_array)

def place_modules(srfpts_tree, cost_function_tree, lable_array, module_use_tree, module_geometry, module_mask_tree):
    ### convert cost function and srf points information from gh trees to np array
    srfpts_matrix = np.array(h.tree_to_matrix(srfpts_tree))
    grid_size = abs(srfpts_matrix[0,0].X - srfpts_matrix[0,1].X)
    cost_function_matrix = h.matrix_str2floats(h.tree_to_matrix(cost_function_tree))
    lable_array_matrix = h.tree_to_matrix(lable_array)
    
    return 1