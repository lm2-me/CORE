from math import sqrt
from types import NoneType
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

def tree_to_list(tree: Dict[str, List[str]]):
    tree_to_list = [(int(k[1:-1]),v) for k,v in tree.items()]
    tree_to_list.sort()
    return [x[1] for x in tree_to_list]
    
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

def corners(geometry: r3d.Surface):
    bbox = geometry.GetBoundingBox()
    btm_left = bbox.Min
    tp_right = bbox.Max
    tp_left = r3d.Point3d(btm_left.X, tp_right.Y, 0)
    btm_right = r3d.Point3d(tp_right.X, btm_left.Y, 0)

    return btm_left, tp_left, tp_right, btm_right, [btm_left, tp_left, tp_right, btm_right]

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

def location_closest_grid_point(srfpts_matrix, points):
    out_location = []
    for row, l in enumerate(srfpts_matrix):
        for column, pt1 in enumerate(l):
            if isinstance(points, list):
                for pt2 in points:
                    distance = abs(m.dist((pt1.X, pt1.Y, pt1.Z), (pt2.X, pt2.Y, pt2.Z)))
                    if distance < 0.1:
                        out_location.append([row, column])
                    if len(out_location) == len(points):
                        break
                    else: continue
            else:
                distance = abs(m.dist((pt1.X, pt1.Y, pt1.Z), (points.X, points.Y, points.Z)))
                if distance < 0.1:
                    out_location.append(row)
                    out_location.append(column)
                    break
                else: continue
    while len(out_location) < len(points) and len(out_location) < 10:
        out_location.append(None)

    return out_location

def update_label(label_array_np, srfpts_matrix, pts_to_update, new_label):
    for row, l in enumerate(srfpts_matrix):
        for column1, pt1 in enumerate(l):
            for pt2 in pts_to_update:
                if isinstance(pt2, list):
                    for pts in pt2:
                        distance = abs(m.dist((pt1.X, pt1.Y, pt1.Z), (pts.X, pts.Y, pts.Z)))
                        if distance < 0.05:
                            label_array_np[row][column1] = new_label
                else: 
                    distance = abs(m.dist((pt1.X, pt1.Y, pt1.Z), (pt2.X, pt2.Y, pt2.Z)))
                    if distance < 0.05:
                        label_array_np[row][column1] = new_label
                    else:
                        continue
    
    return label_array_np


def convert_interior_boundaries(arr_base: np.ndarray):
    arr_bottom_right = np.vstack((
        np.hstack((
            arr_base[1:,1:],
            np.full((arr_base.shape[0] - 1, 1), 'x')
        )),
        np.full((1, arr_base.shape[1]), 'x')
    ))

    arr_bottom_left = np.vstack((
        np.hstack((
            np.full((arr_base.shape[0] - 1, 1), 'x'),
            arr_base[1:,:-1]
        )),
        np.full((1, arr_base.shape[1]), 'x')
    ))

    arr_top_right = np.vstack((
        np.full((1, arr_base.shape[1]), 'x'),
        np.hstack((
            arr_base[:-1,1:],
            np.full((arr_base.shape[0] - 1, 1), 'x')
        ))
    ))

    arr_top_left = np.vstack((
        np.full((1, arr_base.shape[1]), 'x'),
        np.hstack((
            np.full((arr_base.shape[0] - 1, 1), 'x'),
            arr_base[:-1,:-1]
        ))
    ))

    arr_combined = np.dstack((arr_base, arr_top_left, arr_top_right, arr_bottom_left, arr_bottom_right))

    for y in range(arr_combined.shape[0]):
        for x in range(arr_combined.shape[1]):
            if (
                arr_combined[y,x][0] == 'b' and
                not (
                    np.isin('x', arr_combined[y,x]) or
                    np.isin('e', arr_combined[y,x]) or
                    np.isin('s', arr_combined[y,x])
                )
            ):
                arr_base[y,x] = 'i'

    return arr_base