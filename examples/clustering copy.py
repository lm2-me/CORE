"""
functions related to creating and optimizing hub locations

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - NetworkClustering

Methods:
    - reset_hub_df_values: Reset the data stored in the dataframe so that previous runs won't impact the current run
    - generate_random_points: generate random points within the boundary of the loaded city at which to place hub locations
    - hub_clusters: cluster houses to each hub based on the travel distance to each hub
    - hub_clusters_euclidean: cluster houses to each hub based on the euclidean distance to each hub
    - new_hub_location: move the hub based on travel distance of all houses in the hub cluster
    - add_points: add new hub point location, currently adding random points
    - hub_fitness: calculate the hub fitness value
"""


import numpy as np
import math as m
import random
import pickle
import os.path

import pandas
from .utils.network_helpers import *
from re import X

class NetworkClustering():
    # Plot settings
    
    def __init__(self, name: str):
        self.name = name
        self.iteration = 1
        self.hub_list_dictionary = None
        self.hub_assignments_df = pandas.DataFrame()
        
    def __repr__(self):
        return "<Clustering object of {}>".format(self.name)

    def save_iteration(self, name: str, folder: str):        
            object_name = name
            path = folder + str(object_name) + '.pkl'
            print('Saving {} to {}'.format(object_name, path))

            with open(path, 'wb') as file:
                pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
    
    def load_clustering(self, name: str, folder:str):
        object_name = name
        path = folder + str(object_name) + '.pkl'
        if os.path.isfile(path):
            # Load a graph from drive
            print('Loading {} from {}'.format(object_name, path))

            with open(path, 'rb') as file:
                last_state = pickle.load(file)
                self.name = last_state.name
                self.iteration = last_state.iteration
                self.hub_list_dictionary = last_state.hub_list_dictionary
                self.hub_assignments_df = last_state.hub_assignments_df
        else:
            print('Could not locate {} at {}'.format(object_name, path))

    
    ### Initialize DF, reset the data stored in the dataframe so that previous runs won't impact the current run
    def reset_hub_df(self, City):
        hub_assignments_df = self.hub_assignments_df

        print("Loading hubs...")
        if 'shortest_path_result' not in hub_assignments_df:
            shortest_path = [(m.inf)] * len(City.building_addr_df)
            hub_assignments_df['shortest_path_result'] = shortest_path
        else:
            hub_assignments_df['shortest_path_result'].values[:] = np.inf

        if 'nearesthub' not in hub_assignments_df:
            hub_num = [('None')] * len(City.building_addr_df)
            hub_assignments_df['nearesthub'] = hub_num
        else:
            hub_assignments_df['nearesthub'].values[:] = 'None'
        
        if 'hubdistance' not in hub_assignments_df:
            hub_dist = [(m.inf)] * len(City.building_addr_df)
            hub_assignments_df['hubdistance'] = hub_dist
        else:
            hub_assignments_df['hubdistance'].values[:] = np.inf
        
        if 'hub_x' not in hub_assignments_df:
            hub_x = [(0)] * len(City.building_addr_df)
            hub_assignments_df['hub_x'] = hub_x
        else:
            hub_assignments_df['hub_x'].values[:] = 0
        
        if 'hub_y' not in hub_assignments_df:
            hub_y = [(0)] * len(City.building_addr_df)
            hub_assignments_df['hub_y'] = hub_y
        else:
            hub_assignments_df['hub_y'].values[:] = 0
        
        if 'path_not_found' not in hub_assignments_df:
            value = [(False)] * len(City.building_addr_df)
            hub_assignments_df['path_not_found'] = value
        else:
            hub_assignments_df['path_not_found'].values[:] = False
        
        if 'euclid_nearesthub' not in hub_assignments_df:
            hub_num = [('')] * len(City.building_addr_df)
            hub_assignments_df['euclid_nearesthub'] = hub_num
        else:
            hub_assignments_df['euclid_nearesthub'].values[:] = ''
        
        if 'euclid_hubdistance' not in hub_assignments_df:
            euclid_hub_dist = [(m.inf)] * len(City.building_addr_df)
            hub_assignments_df['euclid_hubdistance'] = euclid_hub_dist
        else:
            hub_assignments_df['euclid_hubdistance'].values[:] = np.inf

        # Save to Clustering object
        self.hub_assignments_df = hub_assignments_df

    ### generate random points within the boundary of the loaded city at which to place hub locations
    def generate_random_points(self, coordinates_transformed_xy, start_pt_ct):
        print('adding points to hub dictionary', self.hub_list_dictionary)
        if self.hub_list_dictionary == None:
            hub_dictionary = {}
            self.hub_list_dictionary = hub_dictionary

        #[N, S, E, W]
        #w_n_corner, e_n_corner, w_s_corner, e_s_corner
        coordinateX_min = coordinates_transformed_xy[0][0]
        coordinateX_max = coordinates_transformed_xy[1][0]
        coordinateY_min = coordinates_transformed_xy[2][1]
        coordinateY_max = coordinates_transformed_xy[0][1]

        #print(coordinateX_min, coordinateX_max, coordinateY_min, coordinateY_max)

        index = len(self.hub_list_dictionary)+1
        
        #k-means ++
        # for i in range(start_pt_ct):
        #     step_avg.append([list(ci[0])])
        #     init_list.pop(sel_int[0])
        #     dist_list_dim = ds.cdist(ci, dataset[init_list])**2
        #     dist_list = dist_list_dim[0] / np.sum(dist_list_dim[0])
        #     sel_int = np.random.choice(init_list, 1, p=dist_list)
        #     ci = [dataset[sel_int[0]]]

        for i in range(start_pt_ct):
            x = random.uniform(coordinateX_min, coordinateX_max)
            y = random.uniform(coordinateY_min, coordinateY_max)

            index_name = 'hub ' + str(index)

            if index_name not in self.hub_list_dictionary:
                self.hub_list_dictionary[index_name] = {
                    "x": x,
                    "y": y,
                    "avg_time": 0,
                    "max_time": 0,
                    "people_served": 0,
                    "dest_edge": 0,
                    "yx_transform": 0,
                } 
            index += 1

    ### cluster houses to each hub based on the travel distance to each hub
    def hub_clusters(self, City, orig_yx_transf, orig_edges, name, data_folder, cpu_count):
        ### randomly generated hub locations in hub_dictionary
        ### cluster houses around closest hub locations
        hub_dictionary = self.hub_list_dictionary

        #!get yx transform no longer needed
        hub_names, hub_yx_transf = get_yx_transf_from_dict(hub_dictionary)

        hub_yx_tranf_to_calc = []
        hub_name_to_calc = []

        for hub_name, hub_value in self.hub_list_dictionary.items():
            if hub_value['dest_edge'] == 0:
                hub_yx_tranf_to_calc.append(hub_yx_transf[hub_names.index(hub_name)])
                hub_name_to_calc.append(hub_name)

        if len(hub_yx_tranf_to_calc) > 0:
            #! no longer needed
            dest_edges = h.nearest_edges_hubs(City, hub_yx_tranf_to_calc, cpu_count)
            for i, edge in enumerate(dest_edges):
                self.hub_list_dictionary[hub_name_to_calc[i]]['dest_edge'] = edge
                self.hub_list_dictionary[hub_name_to_calc[i]]['yx_transform'] = hub_yx_tranf_to_calc[i]
        
        #print('after getting additional paths ', self.hub_list_dictionary)
        #dest_edges_all = h.nearest_edges_hubs(City, hub_yx_transf, cpu_count) ### remove for testing only
        
        dest_edges_all = []
        for hub_name, hub_value in self.hub_list_dictionary.items():
            dest_edges_all.append(hub_value['dest_edge'])
        
        hub_yx_dict = dict(zip(hub_names, hub_yx_transf))
        hub_edge_dict = dict(zip(hub_names, dest_edges_all))
        
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
        ### CAN THIS BE CALLED HERE?
        hub_dist = City.shortest_paths(orig_yx_transf_long, hub_yx_transf_long, orig_edges_long, dest_edges_long, weight='travel_time', method='dijkstra', return_path=True, cpus=cpu_count)

        #! not needed
        list_of_sublists = get_sublists(hub_dist, len(hub_dictionary))

        not_found_count = 0
        for i, sublist in enumerate(list_of_sublists):
            hub = hub_names[i]
            for j, row in enumerate(sublist):
                if row[1] == []:
                    not_found_count += 1
                    self.hub_assignments_df.at[j,'path_not_found']=True 
                    manhattan = get_manhattan_distance(hub_yx_dict[hub], orig_yx_transf[i])
                    row_aslist = list(row)
                    row_aslist[0] = manhattan
                    row = tuple(row_aslist)
                
                dist = row[0]
                if dist < self.hub_assignments_df.loc[j,'hubdistance']:
                    self.hub_assignments_df.at[j,'shortest_path_result']=hub_dist
                    self.hub_assignments_df.at[j,'hubdistance']=dist
                    self.hub_assignments_df.at[j,'nearesthub']=hub
                    self.hub_assignments_df.at[j,'hub_x']=hub_dictionary[hub]['x']
                    self.hub_assignments_df.at[j,'hub_y']=hub_dictionary[hub]['y']
        
        print(not_found_count, ' path(s) not found. Used mannhattan distance for these instances.')
                
        self.save_iteration(name, data_folder)
        #df_print1 = City.building_addr_df[['shortest_path_result', 'nearesthub']]
        #print(df_print1)

        #df_print2 = City.building_addr_df[['x', 'y', 'hubdistance', 'nearesthub', 'path_not_found']]
        #print(df_print2)

    ### cluster houses to each hub based on the euclidean distance to each hub
    def hub_clusters_euclidean(self, orig_yx_transf, name, data_folder):
        ### randomly generated hub locations is hub_dictionary
        #! not needed anymore
        hub_names, hub_yx_transf = h.get_yx_transf_from_dict(self.hub_list_dictionary)
        
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
            
                if euclid_dist < self.hub_assignments_df.loc[i,'euclid_hubdistance']:
                    self.hub_assignments_df.at[i,'euclid_hubdistance']=euclid_dist
                    self.hub_assignments_df.at[i,'euclid_nearesthub']=hub
        
        self.save_iteration(name, data_folder)

    ### move the hub based on travel distance of all houses in the hub cluster
    def new_hub_location(self, City):
        hub_dictionary = self.hub_list_dictionary
        for (hub_name, hub_data) in hub_dictionary.items():
            dist_moved = 0
            x = []
            y = []
            for i, row in self.hub_assignments_df.iterrows():
                if row['nearesthub'] == hub_name:
                    #x
                    house_x = City.building_addr_df.iloc[i]['x']
                    x.append(house_x)
                    #y
                    house_y = City.building_addr_df.iloc[i]['y']
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

        self.hub_list_dictionary = hub_dictionary

        return move_distance

    ### add new random hub points
    def add_points(self, point_count, coordinates_transformed_xy):
        print('adding ', point_count, ' point(s)')
        self.generate_random_points(coordinates_transformed_xy, point_count)

    ### calculate the hub fitness
    def hub_fitness(self, City, max_travel_time):
        # dictionary: x, y, avg_time, people_served
        # find and save average time and people served to hub_dictionary
        hub_dictionary = self.hub_list_dictionary
        
        #print(City.building_addr_df[['x', 'y', 'hubdistance', 'nearesthub']])
        time_requirement = False
        max_time_list = []
        
        for (hub_name, _) in hub_dictionary.items():
            all_times = []
            all_people = []
            for i, row in self.hub_assignments_df.iterrows():
                if row['nearesthub'] == hub_name:
                    #travel time
                    time = row['hubdistance']
                    all_times.append(time)
                    #people served
                    people = City.building_addr_df.iloc[i]['addr']
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

        self.hub_list_dictionary = hub_dictionary

        return time_check, max_time_list