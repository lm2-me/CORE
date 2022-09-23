from math import sqrt
from typing import Dict, List
import rhino3dm as r3d
import math as m
import copy
import numpy as np

def list_to_tree(l):
    out = {}
    for i,r in enumerate(l):
        out['{}'.format(i)] = r
    return out

def line_closest_point(line: r3d.Line, point):
    ep1 = line.From
    ep2 = line.To

    distance = abs((ep2.X - ep1.X) * (ep1.Y - point.Y) - (ep1.X - point.X)*(ep2.Y - ep1.Y)) / (m.sqrt(m.pow((ep2.X - ep1.X),2) + m.pow((ep2.Y - ep1.Y),2)))
    distance_rounded = round(distance,2)

    return distance_rounded

def tree_to_matrix(tree: Dict[str, List[str]]):
    out = []
    tree_to_list = [(int(k[1:-1]),v) for k,v in tree.items()]
    tree_to_list.sort()
    for _,b in tree_to_list:
        out.append(b)
    return out
    
def matrix_str2floats(matrix: List[List[str]]):
    out = copy.deepcopy(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            out[i][j] = [float(v) for v in matrix[i][j].split(',')]
    return out

def matrix_floats2str(matrix: List[List[float]]):
    out = []
    for i in range(len(matrix)):
        rowpts = []
        for j in range(len(matrix[i])):
            rowpts.append('{0},{1}'.format(matrix[i][j][0], matrix[i][j][1]))
        out.append(rowpts)

    return out

def matrix_mask(matrix, mask_values):
    out = copy.deepcopy(matrix)
    mask_values_str = mask_values.split(',')
    mask_values_int = []

    for v in mask_values_str:
        mask_values_int.append(int(v))

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            out[i][j] = mask_values_int
    
    return out

def vector3d_2pts(pt1, pt2):

    out = r3d.Vector3d(pt2.X - pt1.X, pt2.Y - pt1.Y, pt2.Z - pt1.Z)

    return out

def corners(geometry):
    btm_left = geometry.PointAt(int(geometry.Domain(0).T0), int(geometry.Domain(1).T0))
    tp_left = geometry.PointAt(int(geometry.Domain(0).T0), int(geometry.Domain(1).T1))
    tp_right = geometry.PointAt(int(geometry.Domain(0).T1), int(geometry.Domain(1).T1))
    btm_right = geometry.PointAt(int(geometry.Domain(0).T1), int(geometry.Domain(1).T0))
    return btm_left, tp_left, tp_right, btm_right

def divide_surface_inset(surface, grid_size):
    srf_u_start = int(surface.Domain(0).T0)
    srf_u_end = int(surface.Domain(0).T1)
    srf_v_start = int(surface.Domain(1).T0)
    srf_v_end = int(surface.Domain(1).T1)

    srfpts = []

    for y in range(srf_v_end-int(grid_size), srf_v_start, -int(grid_size)):
        rowpts = []
        for x in range(srf_u_start+int(grid_size), srf_u_end, int(grid_size)):
            point = r3d.Point3d(surface.PointAt(x,y).X,surface.PointAt(x,y).Y,0)
            rowpts.append(point)
                        
        srfpts.append(rowpts)

    return srfpts

def divide_surface(surface, grid_size):
    srf_u_start = int(surface.Domain(0).T0)
    srf_u_end = int(surface.Domain(0).T1)
    srf_v_start = int(surface.Domain(1).T0)
    srf_v_end = int(surface.Domain(1).T1)

    srfpts = []

    for y in range(srf_v_end, srf_v_start-int(grid_size), -int(grid_size)):
        rowpts = []
        for x in range(srf_u_start, srf_u_end+int(grid_size), int(grid_size)):
            point = r3d.Point3d(surface.PointAt(x,y).X,surface.PointAt(x,y).Y,0)
            rowpts.append(point)
                        
        srfpts.append(rowpts)
    
    srfpts_np = np.array(srfpts)
    midpoints = srfpts_np[1:-1, 1:-1].tolist()
    
    edge =  np.concatenate((srfpts_np[0,:], srfpts_np[1:,-1], srfpts_np[-1,:-1], srfpts_np[1:-1,0])).tolist()
    
    return midpoints, edge