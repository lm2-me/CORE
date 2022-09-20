import ghhops_server as hs
import rhino3dm as r3d
import prog_place.helpers as h
import math as m
import numpy as np

#see app.py file for description of each gh node and more specific information regarding the inputs and outputs

def divide_site(site: r3d.Surface, road_lines_tree, sidewalk_lines_tree, grid_size: float):
    sidewalk_lines: list[r3d.Line] = sidewalk_lines_tree['{0}']
    road_lines: list[r3d.Line] = road_lines_tree['{0}']

        
    print('called method, sidewalk, road')
    print(sidewalk_lines)
    print(road_lines)

    srfpts = []
    values = []
    
    srf_u_start = int(site.Domain(0).T0)
    srf_u_end = int(site.Domain(0).T1)
    srf_v_start = int(site.Domain(1).T0)
    srf_v_end = int(site.Domain(1).T1)

    srf_btm_left_corner = site.PointAt(srf_u_end, srf_v_end)

    print('pre-loop')

    for y in range(srf_v_start, srf_v_end, int(grid_size)):
        rowpts = []
        rowpts_vals = []
        for x in range(srf_u_start, srf_u_end, int(grid_size)):
            road_weight = m.inf
            sidewalk_weight = m.inf
            point = r3d.Point3d((srf_btm_left_corner.X-x-grid_size),(srf_btm_left_corner.Y-y-grid_size),0)
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

    ### convert cost function and srf points information from gh trees to np array
    srfpts_matrix = np.array(h.tree_to_matrix(srfpts_tree))
    cost_function_matrix = h.matrix_str2floats(h.tree_to_matrix(cost_function_tree))

    ### convert module cost function information from strings to np array 
    module_mask_np = np.array([int(v) for v in module_mask.split(',')])

    ### mask cost function with module parameters
    module_masked_cost = np.dot(np.array(cost_function_matrix), module_mask_np)

    ### remove locations that place packages module outside of the site area
    ### TODO: update to incorporate rotations 0, 90, 180, 270 deg, repeat 4 times and look for lowest min from the 4 situations
    masked_cost_placement_zerodeg = np.copy(module_masked_cost)
    if (width > 1):
        masked_cost_placement_zerodeg[-(1 - width):, :] = m.inf
    if (height > 1):
            masked_cost_placement_zerodeg[:, -(1 - height):] = m.inf


    ### TODO: add for loop to check which rotation has lowest cost function and assign to masked_cost_placement variable
    masked_cost_placement = masked_cost_placement_zerodeg

    best_points_locs = np.where(masked_cost_placement == masked_cost_placement.min())
    best_points = [(x,y) for x,y in zip(best_points_locs[0], best_points_locs[1])]
    best_points.sort()
    best_point = best_points[int(len(best_points) / 2)]

    top_left_corner = srfpts_matrix[best_point]
    move_vector = h.vector3d_2pts(top_left_corner, point2)
    success = module_geometry.Translate(move_vector)
    print('moved packages module?', success)

    return module_geometry

#def place_modules(srfpts_tree, cost_function_tree, module_use_tree, module_geometry, module_mask_tree):
   # return 1