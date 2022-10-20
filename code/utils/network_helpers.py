"""
functions related to creating and optimizing hub locations

By:
Lisa-Marie Mueller, TU Delft

Classes:
    - None

Functions:
    - coordinates_to_tupple: convert city coordinates into a tupple
    - flip_lat_long: flip the order of latitude and longitude in coordinates
    - get_euclidean_distance: calculate euclidean distance between two points
    - get_manhattan_distance: calculate manhattan distance between two points
    - nearest_edges_buildings: find the nearest network edges for all building locations
    - nearest_edges_hubs: find the nearest network edges for all hub locations
    - get_yx_transf_from_dict: get the yx transform values from the dictionary containing hub locations
    - get_sublists: from a combined list of distance to all hubs, get the sublists for each hub
"""

import numpy as np

from re import X
from utils.multicore_nearest_edges import multicore_nearest_edge

### convert city coordinates into a tupple
def coordinates_to_tupple(coordinates):
    #[N, S, E, W]
    w_n_corner = (coordinates[3], coordinates[0]) 
    e_n_corner = (coordinates[2], coordinates[0]) 
    w_s_corner = (coordinates[3], coordinates[1]) 
    e_s_corner = (coordinates[2], coordinates[1]) 

    return ([w_n_corner, e_n_corner, w_s_corner, e_s_corner])

### flip the order of latitude and longitude in coordinates
def flip_lat_long(coordinates_list):
    return [(coord[1], coord[0]) for coord in coordinates_list]

### calculate euclidean distance between two points
def get_euclidean_distance(hub_yx_value, house_yx_value):
    point2 = np.array(hub_yx_value)
    point1 = np.array(house_yx_value)
    euclid_dist = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
    
    return euclid_dist

### calculate manhattan distance between two points
def get_manhattan_distance(hub_yx_value, house_yx_value):
    point2 = np.array(hub_yx_value)
    point1 = np.array(house_yx_value)
    #Abs [a − x] + Abs [b − y] + Abs [c − z]
    manhattan_dist = (abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]))
    
    return manhattan_dist 

### find the nearest network edges for all building locations
def nearest_edges_buildings(City, orig_yx_transf, name, data_folder, cpu_count):
    x = []
    y = []

    for orig_yx in orig_yx_transf:
        x.append(orig_yx[1])
        y.append(orig_yx[0])
    
    edges = City.nearest_edges(x, y, 5, cpus=cpu_count)
    City.save_graph(name, data_folder)
    orig_edge = edges
    #print('orig_edge', orig_edge)
    return orig_edge

### find the nearest network edges for all hub locations
def nearest_edges_hubs(City, hub_yx_transf, cpu_count):
    x = []
    y = []

    for hub in hub_yx_transf:
        x.append(hub[1])
        y.append(hub[0])
    
    edges, _ = multicore_nearest_edge(City.graph, x, y, City.interpolation, cpus=cpu_count)
    #print(f"edge: {edges}")
    return edges

### get the yx transform values from the dictionary containing hub locations
def get_yx_transf_from_dict(hub_dictionary):

    label_list = []
    value_list = []
    for (hub_name, hub_loc) in hub_dictionary.items():
        
        label_list.append(hub_name)
        value_list.append(
            (hub_loc['y'], hub_loc['x'])
        )

    return label_list, value_list

### from a combined list of distance to all hubs, get the sublists for each hub
def get_sublists(list, sublist_number):
    sub_list_length = len(list) // sublist_number
    list_of_sublists = []
    for i in range(0, len(list), sub_list_length):
        list_of_sublists.append(list[i:i+sub_list_length])
    return list_of_sublists