
"""
functions related to creating and optimizing hub locations

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - NetworkClustering

Methods:
    >>> divide_site: Turn site into grid
    >>> place_packages: Place package module on site
    >>> place_modules: Place modules on site
    >>> labelstopoints: Convert labels to points
"""

from cmath import nan
import ghhops_server as hs
import rhino3dm as r3d
import prog_place.helpers as h
import math as m
import numpy as np
import copy

#see app.py file for description of each gh node and more specific information regarding the inputs and outputs

def divide_site(site: r3d.Surface, road_lines_tree, sidewalk_lines_tree, grid_size: float, context, sun_hours):
    """Grasshopper component: Turn site into grid
        
        Developed by Lisa-Marie Mueller

        Parameters
        ----------
        site : r3d.Surface
            Surface of site

        road_lines_tree : tree
            Lines representing roads.
        
        sidewalk_lines_tree : tree
            Lines representing sidewalks.
        
        grid_size : integer
            Distance of grid step size.
        
        context : list
            Points locating context.
        
        site_hours : tree
            number of sun hours per point

        """  

    sidewalk_lines: list[r3d.Line] = sidewalk_lines_tree['{0}']
    road_lines: list[r3d.Line] = road_lines_tree['{0}']
    context_matrix = np.array(h.tree_to_matrix(context))
    sun_hours_matrix = np.array(h.tree_to_matrix(sun_hours))
    
    nearby_buildings = context_matrix[0]
    landmarks = context_matrix[1]
    views = context_matrix[2]

    srfpts = []
    values = []
    values_floats = []
    
    srf_u_start = int(site.Domain(0).T0)
    srf_u_end = int(site.Domain(0).T1)
    srf_v_start = int(site.Domain(1).T0)
    srf_v_end = int(site.Domain(1).T1)

    srf_btm_left_corner = site.PointAt(srf_u_start, srf_v_start)

    ### cost function includes: road distance, sidewalk distance, sun hours, nearby buildings, distance to landmark, distance to view
    for y in range(srf_v_end, srf_v_start-int(grid_size), -int(grid_size)):
        rowpts = []
        rowpts_vals = []
        rowpts_floats = []
        for x in range(srf_u_start, srf_u_end+int(grid_size), int(grid_size)):
            road_weight = m.inf
            sidewalk_weight = m.inf
            sun_hours = m.inf
            nearby_buildings_weight = m.inf
            landmarks_weight = m.inf
            views_weight = m.inf

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
            
            sun_weight = sun_hours_matrix[x][y]

            for b in nearby_buildings:
                new_nearby_buildings_weight = b.DistanceTo(point)
                if new_nearby_buildings_weight < nearby_buildings_weight:
                    nearby_buildings_weight = new_nearby_buildings_weight
            
            for l in landmarks:
                new_landmarks_weight = l.DistanceTo(point)
                if new_landmarks_weight < landmarks_weight:
                    landmarks_weight = new_landmarks_weight
            
            for v in views:
                new_views_weight = v.DistanceTo(point)
                if new_views_weight < views_weight:
                    views_weight = new_views_weight

            rowpts_floats.append([road_weight, sidewalk_weight, sun_weight, nearby_buildings_weight, landmarks_weight, views_weight])           
            rowpts_vals.append('{0},{1},{2},{3},{4},{5}'.format(road_weight, sidewalk_weight, sun_weight, nearby_buildings_weight, landmarks_weight, views_weight))
        srfpts.append(rowpts)
        values.append(rowpts_vals)
        values_floats.append(rowpts_floats)
    
    values_floats_normalized = h.normalize_weight(values_floats)

    values_normalized = []
    for row in values_floats_normalized:
        row_norm = []
        for cost in row:
            row_norm.append('{0},{1},{2},{3},{4},{5}'.format(cost[0], cost[1], cost[2], cost[3], cost[4], cost[5]))
        values_normalized.append(row_norm)

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
    
    return h.list_to_tree(srfpts), h.list_to_tree(values_normalized), h.list_to_tree(label_array_np.tolist())

def place_packages(srfpts_tree, cost_function_tree, labels_tree, module_use, module_geometry: r3d.Surface, module_mask, door):
    """Grasshopper component: Place package module on site
    
    Developed by Lisa-Marie Mueller

    Parameters
    ----------
    srfpts_tree : tree
        points dividing site surface

    cost_function_tree : tree
        cost function for entire site.
    
    labels_tree : tree
        labels of each point in grid.
    
    module_use : tree
        human readable designation for module.
    
    module_geometry : r3d.Geometry
        surface for module area.
    
    module_mask : tree
        module mask to mask cost function for relevant values
    
    door : tree
        point of door locations

    """  

    ### convert door locations to points
    door_points = []
    door_floats = [float(v) for v in door.split(',')]

    for j in range(0, len(door_floats), 2):
            door_points.append(r3d.Point3d(door_floats[j], door_floats[j+1], 0))

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
    ### o - outside of site edge
    ### x - door
    label_array_np = np.array(h.tree_to_matrix(labels_tree))

    ### convert module cost function information from strings to np array 
    module_mask_np = np.array([float(v) for v in module_mask.split(',')])

    for i, val in enumerate(module_mask_np):
        if val == 0:
            module_mask_np[i] = m.inf

    module_mask_expanded = copy.deepcopy(cost_function_matrix)
    for i, row in enumerate(cost_function_matrix):
        for j, _ in enumerate(row):
            module_mask_expanded[i][j] = module_mask_np

    ### mask cost function with module parameters
    module_masked_cost = copy.deepcopy(cost_function_matrix)
    num_weights = len(cost_function_matrix[0][0])
    for num in range(num_weights):
        for i, row in enumerate(cost_function_matrix):
            for j, weights in enumerate(row):
                if module_mask_expanded[i][j][num] == -1.:
                    module_masked_cost[i][j][num] = m.pow(weights[num], module_mask_expanded[i][j][num])
                else:
                    if weights[num] < 1 and module_mask_expanded[i][j][num] == m.inf:
                        module_masked_cost[i][j][num] = m.inf
                    else:
                        module_masked_cost[i][j][num] = weights[num] * module_mask_expanded[i][j][num]

    ### remove locations that place packages module outside of the site area
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
        
    ### rotate door location
    for i, d in enumerate(door_points):
        rotated_x, rotated_y = h.rotate_point(tp_left, d, rotation)
        door_points[i] = r3d.Point3d(rotated_x, rotated_y, 0)
        print('rotated packages module doors')

    rot_mat_flat = r3d.Transform.ToFloatArray(rotation_transform, True)
    rot_mat = np.asarray(rot_mat_flat).reshape(4, 4)
    move_from_np = np.array([move_from.X, move_from.Y, move_from.Z, 1])
    move_from_new = np.dot(rot_mat, move_from_np.T)
    move_from = r3d.Point3d(move_from_new[0], move_from_new[1], 0)

    move_vector = h.vector3d_2pts(move_from, top_left_corner)
    success_move = module_geometry.Translate(move_vector)
    print('moved packages module?', success_move)

    ### move door location
    for i, d in enumerate(door_points):
        door_points[i] = r3d.Point3d(d.X + move_vector.X, d.Y + move_vector.Y, 0)
        print('moved packages module doors')

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

    for row, list in enumerate(srfpts_matrix):
        for column1, pt1 in enumerate(list):
            for d in door_points:
                distance = abs(m.dist((pt1.X, pt1.Y, pt1.Z), (d.X, d.Y, d.Z)))
                if distance < 0.05:
                    label_array_np[row][column1] = 'x'

    label_array = label_array_np.tolist()

    return module_geometry, h.list_to_tree(h.matrix_floats2str(out_cost_function_matrix)), module_edge, h.list_to_tree(label_array)

def place_modules(srfpts_tree, cost_function_tree, label_array, module_use_tree, module_geometry_list, module_mask_tree, door_tree):
    """Grasshopper component: Place modules on site
    
    Developed by Lisa-Marie Mueller

    Parameters
    ----------
    srfpts_tree : tree
        points dividing site surface

    cost_function_tree : tree
        cost function for entire site.
    
    label_array : list
        labels of each point in grid.
    
    module_use : tree
        human readable designation for module.
    
    module_geometry : list
        surface for module area.
    
    module_mask : tree
        module mask to mask cost function for relevant values
    
    door : tree
        point of door locations

    """  

    ### convert cost function and srf points information from gh trees to np array
    srfpts_matrix = np.array(h.tree_to_matrix(srfpts_tree))
    grid_size = abs(srfpts_matrix[0,0].X - srfpts_matrix[0,1].X)
    module_mask_matrix = h.tree_to_matrix(module_mask_tree)
    cost_function_matrix = h.matrix_str2floats(h.tree_to_matrix(cost_function_tree))
    label_array_matrix = np.array(h.tree_to_matrix(label_array))
    module_use = h.tree_to_list(module_use_tree)

    door_array = h.matrix_str2floats(h.tree_to_matrix(door_tree))
    ### convert door locations to points
    door_points = []
    for i in range(len(door_array)):
        if door_array[i] == None:
            door_points.append([None])
        else:
            points = []
            for j in range(0, len(door_array[i][0]), 2):
                points.append(r3d.Point3d(door_array[i][0][j], door_array[i][0][j+1], 0))
            door_points.append(points)

    ### convert module cost function information from strings to np array 
    module_mask_np = []
    module_masked_cost_all = []

    for i, item in enumerate(module_mask_matrix):
        mask = [float(v) for v in item[0].split(',')]

        for j, val in enumerate(mask):
            if val == 0:
                mask[j] = m.inf

        module_mask_np.append(mask)

        module_mask_expanded = copy.deepcopy(cost_function_matrix)
        for k, row in enumerate(cost_function_matrix):
            for l, _ in enumerate(row):
                module_mask_expanded[k][l] = mask

        ### mask cost function with module parameters
        module_masked_cost = copy.deepcopy(cost_function_matrix)
        num_weights = len(cost_function_matrix[0][0])
        for num in range(num_weights):
            for k, row in enumerate(cost_function_matrix):
                for l, weights in enumerate(row):
                    if module_mask_expanded[k][l][num] == -1.:
                        if weights[num] == 0:
                            module_masked_cost[k][l][num] = m.inf
                        else:
                            module_masked_cost[k][l][num] = m.pow(weights[num], module_mask_expanded[i][j][num])
                    else:
                        if weights[num] < 1 and module_mask_expanded[k][l][num] == m.inf:
                            module_masked_cost[k][l][num] = m.inf
                        else:
                            module_masked_cost[k][l][num] = weights[num] * module_mask_expanded[i][j][num]
        module_masked_cost_all.append(module_masked_cost)

    combined_module_masked_cost_all = copy.deepcopy(module_masked_cost_all)
    for num, matrix in enumerate(module_masked_cost_all):
        for i, row in enumerate(matrix):
            for j, weights in enumerate(row):
                total_weight = 0
                for weight in weights:
                    if weight != m.inf:
                        total_weight += weight
                if total_weight == 0:
                    combined_module_masked_cost_all[num][i][j] = m.inf
                else:
                    combined_module_masked_cost_all[num][i][j] = total_weight

    label_array_matrix_revised = copy.deepcopy(label_array_matrix)

    out_module_edges = []
    
    #place each module, update label array,
    for num, geo in enumerate(module_geometry_list):
        print('placing module ',  module_use[num])
        use_cost_function_matrix = copy.deepcopy(combined_module_masked_cost_all[num])

        for i, row in enumerate(label_array_matrix_revised):
            for j, label in enumerate(row):
                if label == 's' or label == 'i' or label == 'e' or label == 'x':
                    use_cost_function_matrix[i][j] = m.inf

        print(label_array_matrix_revised)

        points_to_order = []
        values_to_order = []
        for i, row in enumerate(use_cost_function_matrix):
            for j, cost in enumerate(row):
                if cost != m.inf:
                    values_to_order.append(cost)
                    points_to_order.append(srfpts_matrix[i][j])
                else:
                    continue

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
            
            _, mvd_tp_left, _, _, _ = h.corners(geo)

            rotation_count = 0
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

                centers, edges = h.divide_surface(geo, grid_size)

                edge_labels = []
                for point in edges:
                    valid_index = False

                    if (0 <= round(point.X) < len(label_array_matrix_revised[0])) and (0 <= round(point.Y) < len(label_array_matrix_revised)):
                        valid_index = True

                    if point != None and valid_index:
                        edge_labels.append(label_array_matrix_revised[round(point.X), round(point.Y)])
                    else:
                        edge_labels.append(None)
                edge_labels_np = np.array(edge_labels)

                center_labels = []
                
                if isinstance(centers[0], list):
                    center_list = centers[0]
                
                else:
                    center_list = centers

                for cntr_pt in center_list:
                    valid_index = False

                    if 0 <= round(cntr_pt.X) < len(label_array_matrix_revised[0]) and (0 <= round(cntr_pt.Y) < len(label_array_matrix_revised)):
                        valid_index = True

                    if point != None and valid_index:
                        center_labels.append(label_array_matrix_revised[round(cntr_pt.X), round(cntr_pt.Y)])
                    else:
                        center_labels.append(None)
                center_labels_np = np.array(center_labels)

                if ((np.isin('b', center_labels_np)) or
                    (np.isin('x', center_labels_np)) or
                    (np.isin('i', center_labels_np)) or
                    (np.isin('o', center_labels_np)) or
                    (np.isin('i', corner_labels_np)) or
                    (np.isin('o', corner_labels_np)) or
                    (np.isin('x', corner_labels_np)) or
                    (np.isin(None, corner_labels_np)) or
                    (np.count_nonzero(corner_labels_np == 'b') < 2) or
                    (np.isin('x', edge_labels_np)) or
                    (np.count_nonzero(edge_labels_np == 'b') == len(edge_labels_np)) or
                    ((np.count_nonzero(edge_labels_np == 'b') + np.count_nonzero(edge_labels_np == 'i')) == len(edge_labels_np))
                ):
                    
                    axis = h.vector3d_2pts(mvd_tp_left, r3d.Point3d(mvd_tp_left.X, mvd_tp_left.Y, 10))
                    success_rot = geo.Rotate(m.pi / 2., axis, mvd_tp_left)
                    rotation_count +=1
                else:
                    valid_placement = True
                    break

            if valid_placement:
                corner_rot_point = mvd_tp_left
                rotation_degree = rotation_count * (m.pi / 2.)
                move_amount = move_vector

                break

            print("no valid position for this module at this location")
        
        if not valid_placement:
            print("no valid position for this module")

        ### move door locations
        for i, d in enumerate(door_points[num]):
            if d == None:
                break
            else:
                moved_point = r3d.Point3d(d.X + move_amount.X, d.Y + move_amount.Y, 0)
                door_points[num][i] = moved_point
                print('moved module doors')

        ### rotate door location
        for i, d in enumerate(door_points[num]):
            if d == None:
                break
            else:
                rotated_x, rotated_y = h.rotate_point(corner_rot_point, d, rotation_degree)
                door_points[num][i] = r3d.Point3d(rotated_x, rotated_y, 0)
                print('rotated module doors')

        module_grid, module_edge  = h.divide_surface(geo, grid_size)
        ### list of module edges to output
        out_module_edges.append(module_edge)

        ### update label array matrix
        label_array_matrix_revised = h.update_label(label_array_matrix_revised, srfpts_matrix, module_edge, 'b')
        label_array_matrix_revised = h.update_label(label_array_matrix_revised, srfpts_matrix, module_grid, 'i')
        if door_points[num][0] != None and door_points[num] != None:
            label_array_matrix_revised = h.update_label(label_array_matrix_revised, srfpts_matrix, door_points[num], 'x')
        ### check if diagonal of each b contains at least one 's' if yes, it is still a boundary, if no, then it becomes an 'i'
        label_array_matrix_revised = h.convert_interior_boundaries(label_array_matrix_revised)

    #return "surface", [excluded] "cost", "module edges", "label"
    return module_geometry_list, h.list_to_tree(out_module_edges), h.list_to_tree(label_array_matrix_revised.tolist())

def labelstopoints(srfpts_tree, label_array):
    """Grasshopper component: conver labels to points
    
    Developed by Lisa-Marie Mueller

    Parameters
    ----------
    srfpts_tree : tree
        points dividing site surface

    label_array : tree
        labels of each point in grid.

    """  

    srfpts_matrix = np.array(h.tree_to_matrix(srfpts_tree))
    label_array_matrix = np.array(h.tree_to_matrix(label_array))

    points_site = []
    points_edges = []
    points_boundary = []
    points_interior = []
    points_doors = []
    points_outside_site = []

    for i, row in enumerate(label_array_matrix):
        for j, label in enumerate(row):
            if label == 's':
                points_site.append(srfpts_matrix[i, j])
            if label == 'e':
                points_edges.append(srfpts_matrix[i, j])
            if label == 'b':
                points_boundary.append(srfpts_matrix[i, j])
            if label == 'i':
                points_interior.append(srfpts_matrix[i, j])
            if label == 'x':
                points_doors.append(srfpts_matrix[i, j])
            if label == 'o':
                points_outside_site.append(srfpts_matrix[i, j])
            else:
                continue

    return points_site, points_edges, points_boundary, points_interior, points_doors, points_outside_site