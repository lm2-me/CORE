import ghhops_server as hs
import rhino3dm as r3d
import prog_place.helpers as h
import math as m
import numpy as np

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

    #print(rowpts_vals)
    return h.list_to_tree(srfpts), h.list_to_tree(values)

def place_packages(srfpts_tree, cost_function_tree, module_use, module_geometry, module_mask):
    #print("place packages")
    # print(cost_function_tree)
    
    srfpts_matrix = h.tree_to_matrix(srfpts_tree)
    cost_function_matrix = h.matrix_str2floats(h.tree_to_matrix(cost_function_tree))
    # print(cost_function_matrix)

    module_mask_np = np.array([int(v) for v in module_mask.split(',')])
    module_mask_matrix = h.matrix_mask(cost_function_matrix, module_mask)
    #print(module_mask_matrix)

    masked_cost = np.dot(np.array(cost_function_matrix), module_mask_np)
    print(masked_cost)

    masked_cost_placement = np.copy(masked_cost)
    width = 3
    height = 7
    if (width > 1):
        masked_cost_placement[-(1 - width):, :] = 0
    if (height > 1):
        masked_cost_placement[:, -(1 - height):] = 0
    print(masked_cost_placement)
    #return 1

def place_modules(srfpts_tree, cost_function_tree, module_use_tree, module_geometry, module_mask_tree):
    return 1