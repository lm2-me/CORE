from math import sqrt
from typing import Dict, List
import rhino3dm as r3d
import math as m
import copy

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

    out = r3d.Vector3d(abs(pt2.X - pt1.X), abs(pt2.Y - pt1.Y), abs(pt2.Z - pt1.Z))

    return out