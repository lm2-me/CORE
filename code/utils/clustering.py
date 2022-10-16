"""
functions related to creating and optimizing hub locations

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - None

Functions:
    - reset_hub_df_values: Reset the data stored in the dataframe so that previous runs won't impact the current run
    - generate_random_points: generate random points within the boundary of the loaded city at which to place hub locations
    - get_sublists: from a combined list of distance to all hubs, get the sublists for each hub
    - hub_clusters: cluster houses to each hub based on the travel distance to each hub
    - hub_clusters_euclidean: cluster houses to each hub based on the euclidean distance to each hub
    - new_hub_location: move the hub based on travel distance of all houses in the hub cluster
    - add_points: add new hub point location, currently adding random points
    - hub_fitness: calculate the hub fitness value
"""


import numpy as np
import math as m
import random

import utils.network_helpers as h
from re import X

### Reset the data stored in the dataframe so that previous runs won't impact the current run
def reset_hub_df_values(City):
    if 'shortest_path_result' not in City.building_addr_df:
        shortest_path = [(m.inf)] * len(City.building_addr_df)
        City.building_addr_df['shortest_path_result'] = shortest_path
    else:
        City.building_addr_df['shortest_path_result'].values[:] = np.inf

    if 'nearesthub' not in City.building_addr_df:
        hub_num = [('None')] * len(City.building_addr_df)
        City.building_addr_df['nearesthub'] = hub_num
    else:
        City.building_addr_df['nearesthub'].values[:] = 'None'
    
    if 'hubdistance' not in City.building_addr_df:
        hub_dist = [(m.inf)] * len(City.building_addr_df)
        City.building_addr_df['hubdistance'] = hub_dist
    else:
        City.building_addr_df['hubdistance'].values[:] = np.inf
    
    if 'hub_x' not in City.building_addr_df:
        hub_x = [(0)] * len(City.building_addr_df)
        City.building_addr_df['hub_x'] = hub_x
    else:
        City.building_addr_df['hub_x'].values[:] = 0
    
    if 'hub_y' not in City.building_addr_df:
        hub_y = [(0)] * len(City.building_addr_df)
        City.building_addr_df['hub_y'] = hub_y
    else:
        City.building_addr_df['hub_y'].values[:] = 0
    
    if 'path_not_found' not in City.building_addr_df:
        City.building_addr_df['path_not_found'] = False
    else:
        City.building_addr_df['path_not_found'].values[:] = False
    
    if 'euclid_nearesthub' not in City.building_addr_df:
        hub_num = [('')] * len(City.building_addr_df)
        City.building_addr_df['euclid_nearesthub'] = hub_num
    else:
        City.building_addr_df['euclid_nearesthub'].values[:] = ''
    
    if 'euclid_hubdistance' not in City.building_addr_df:
        euclid_hub_dist = [(m.inf)] * len(City.building_addr_df)
        City.building_addr_df['euclid_hubdistance'] = euclid_hub_dist
    else:
        City.building_addr_df['euclid_hubdistance'].values[:] = np.inf

### generate random points within the boundary of the loaded city at which to place hub locations
def generate_random_points(coordinates_transformed_xy,start_pt_ct, hub_dictionary=None):
    print('adding points to hub dictionary', hub_dictionary)
    if hub_dictionary == None:
        hub_dictionary = {}

    #[N, S, E, W]
    #w_n_corner, e_n_corner, w_s_corner, e_s_corner
    coordinateX_min = coordinates_transformed_xy[0][0]
    coordinateX_max = coordinates_transformed_xy[1][0]
    coordinateY_min = coordinates_transformed_xy[2][1]
    coordinateY_max = coordinates_transformed_xy[0][1]

    #print(coordinateX_min, coordinateX_max, coordinateY_min, coordinateY_max)

    index = len(hub_dictionary)+1

    for i in range(start_pt_ct):
        x = random.uniform(coordinateX_min, coordinateX_max)
        y = random.uniform(coordinateY_min, coordinateY_max)

        index_name = 'hub ' + str(index)

        if index_name not in hub_dictionary:
            hub_dictionary[index_name] = {
                "x": x,
                "y": y,
                "avg_time": 0,
                "max_time": 0,
                "people_served": 0,
            } 
        index += 1

    return hub_dictionary

### from a combined list of distance to all hubs, get the sublists for each hub
def get_sublists(list, sublist_number):
    sub_list_length = len(list) // sublist_number
    list_of_sublists = []
    for i in range(0, len(list), sub_list_length):
        list_of_sublists.append(list[i:i+sub_list_length])
    return list_of_sublists

### cluster houses to each hub based on the travel distance to each hub
def hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count):
    ### randomly generated hub locations is hub_dictionary
    ### cluster houses around closest hub locations

    hub_names, hub_yx_transf = h.get_yx_transf_from_dict(hub_dictionary)
    dest_edges = h.nearest_edges_hubs(City, hub_yx_transf, cpu_count)
    
    hub_yx_dict = dict(zip(hub_names, hub_yx_transf))
    hub_edge_dict = dict(zip(hub_names, dest_edges))
    
    hub_yx_transf_long = []
    dest_edges_long = []

    ### first_paths returns [(route_weight, nx_route, orig_partial_edge, dest_partial_edge)]
    for hub in hub_names:
        print('calculating paths to ', hub)
        #get distance of all current hub to all houses
        hub_yx_transf_current = [hub_yx_dict[hub]] * len(orig_yx_transf)
        dest_edges_current = [hub_edge_dict[hub]] * len(orig_yx_transf)

        hub_yx_transf_long = hub_yx_transf_long + hub_yx_transf_current
        dest_edges_long = dest_edges_long + dest_edges_current

    orig_yx_transf_long = orig_yx_transf * len(hub_dictionary)
    orig_edges_long = orig_edges * len(hub_dictionary)
    
    # weight options: travel_time, length
    # Returning route_weight, nx_route, orig_partial_edge, dest_partial_edge, orig_yx, dest_yx
    # [(route_weight, nx_route, orig_partial_edge, dest_partial_edge, orig_yx, dest_yx)] 
    hub_dist = City.shortest_paths(orig_yx_transf_long, hub_yx_transf_long, orig_edges_long, dest_edges_long, weight='travel_time', method='dijkstra', return_path=True, cpus=cpu_count)

    list_of_sublists = get_sublists(hub_dist, len(hub_dictionary))

    not_found_count = 0
    for i, sublist in enumerate(list_of_sublists):
        hub = hub_names[i]
        for j, row in enumerate(sublist):
            if row[1] == []:
                not_found_count += 1
                City.building_addr_df.at[j,'path_not_found']=True 
                manhattan = h.get_manhattan_distance(hub_yx_dict[hub], orig_yx_transf[i])
                row_aslist = list(row)
                row_aslist[0] = manhattan
                row = tuple(row_aslist)
  
            dist = row[0]
            if dist < City.building_addr_df.loc[j,'hubdistance']:
                City.building_addr_df.at[j,'shortest_path_result']=hub_dist
                City.building_addr_df.at[j,'hubdistance']=dist
                City.building_addr_df.at[j,'nearesthub']=hub
                City.building_addr_df.at[j,'hub_x']=hub_dictionary[hub]['x']
                City.building_addr_df.at[j,'hub_y']=hub_dictionary[hub]['y']
    
    print(not_found_count, ' path(s) not found. Used mannhattan distance')
              
    City.save_graph(name, data_folder)
    #df_print1 = City.building_addr_df[['shortest_path_result', 'nearesthub']]
    #print(df_print1)

    #df_print2 = City.building_addr_df[['x', 'y', 'hubdistance', 'nearesthub', 'path_not_found']]
    #print(df_print2)
    return 0

### cluster houses to each hub based on the euclidean distance to each hub
def hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder):
    ### randomly generated hub locations is hub_dictionary
    hub_names, hub_yx_transf = h.get_yx_transf_from_dict(hub_dictionary)
    
    hub_yx_dict = dict(zip(hub_names, hub_yx_transf))

    ### first_paths returns [(route_weight, nx_route, orig_partial_edge, dest_partial_edge)]
    for hub in hub_names:
        print(hub)
        #get distance of all current hub to all houses        
        point2 = np.array(hub_yx_dict[hub])
        ## print(point2)

        for i, house in enumerate(orig_yx_transf):
            point1 = np.array(house)
            euclid_dist = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
          
            if euclid_dist < City.building_addr_df.loc[i,'euclid_hubdistance']:
                City.building_addr_df.at[i,'euclid_hubdistance']=euclid_dist
                City.building_addr_df.at[i,'euclid_nearesthub']=hub
    
    City.save_graph(name, data_folder)
    
    return 0

### move the hub based on travel distance of all houses in the hub cluster
def new_hub_location(City, hub_dictionary):
    for (hub_name, hub_data) in hub_dictionary.items():
        dist_moved = 0
        x = []
        y = []
        for _, row in City.building_addr_df.iterrows():
            if row['nearesthub'] == hub_name:
                #x
                house_x = row['x']
                x.append(house_x)
                #y
                house_y = row['y']
                y.append(house_y)   
        all_x = np.array(x)
        all_y = np.array(y)
        average_x = np.sum(all_x) / len(all_x)
        average_y = np.sum(all_y) / len(all_y)
        previous_location = (hub_dictionary[hub_name]['x'], hub_dictionary[hub_name]['y'])
        hub_dictionary[hub_name]['x'] = average_x
        hub_dictionary[hub_name]['y'] = average_y
        new_location = (hub_dictionary[hub_name]['x'], hub_dictionary[hub_name]['y'])

        point1 = previous_location
        point2 = new_location
        move_distance = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5 


    return move_distance

### add new random hub points
def add_points(point_count, hub_dictionary, coordinates_transformed_xy):
    print('adding ', point_count, ' point(s)')
    generate_random_points(coordinates_transformed_xy, point_count, hub_dictionary=hub_dictionary)
    return hub_dictionary

### calculate the hub fitness
def hub_fitness(City, hub_dictionary, max_travel_time):
    # dictionary: x, y, avg_time, people_served
    # find and save average time and people served to hub_dictionary
    
    #print(City.building_addr_df[['x', 'y', 'hubdistance', 'nearesthub']])
    time_requirement = False
    max_time_list = []
    
    for (hub_name, _) in hub_dictionary.items():
        all_times = []
        all_people = []
        for _, row in City.building_addr_df.iterrows():
            if row['nearesthub'] == hub_name:
                #travel time
                time = row['hubdistance']
                all_times.append(time)
                #people served
                people = row['addr']
                all_people.append(people)

        all_times_np = np.array(all_times)
        average = np.sum(all_times_np) / len(all_times)
        all_people_np = np.array(all_people)
        total_people = np.sum(all_people_np)
        hub_dictionary[hub_name]['avg_time'] = average

        max_time_list.append(np.max(all_times_np))
        hub_dictionary[hub_name]['max_time'] = np.max(all_times_np)

        hub_dictionary[hub_name]['people_served'] = total_people
    
    max_time_list_np = np.array(max_time_list)
    time_check = all(i <= max_travel_time for i in max_time_list_np)

    return time_check, max_time_list