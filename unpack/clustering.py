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

from collections import OrderedDict
from .utils.network_helpers import *
from re import X

class NetworkClustering():
    # Plot settings
    
    def __init__(self, name: str):
        self.name = name
        self.iteration = 1
        self.hub_list_dictionary = None
        self.hub_assignments_df = pandas.DataFrame()
        self.max_cores = None
        self.cluster_number = 0
        
    def __repr__(self):
        return "<Clustering object of {}>".format(self.name)

    def save_iteration(self, name: str, folder: str, session_name: str, iteration: str, step:str): 
        folder_path = folder + session_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)   

        object_name = str(name) + '_iteration ' + str(iteration) + '_' + step
        path = folder_path + object_name + '.pkl'
        print('Saving {} to {}'.format(object_name, path))

        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
    
    def continue_clustering(self, path: str):
        if os.path.isfile(path):
            # Load a graph from drive
            print('Loading {}'.format(path))

            with open(path, 'rb') as file:
                last_state = pickle.load(file)
                self.name = last_state.name
                self.iteration = last_state.iteration
                self.hub_list_dictionary = last_state.hub_list_dictionary
                self.hub_assignments_df = last_state.hub_assignments_df
        else:
            print('Could not locate {}'.format(path))
   
    ### Initialize DF, reset the data stored in the dataframe so that previous runs won't impact the current run
    def reset_hub_df(self, City):
        hub_assignments_df = self.hub_assignments_df

        print("Loading hubs...")
        if 'Nearest_hub_idx' not in hub_assignments_df:
            nearest_hub_indx = [0] * len(City.building_addr_df)
            hub_assignments_df['Nearest_hub_idx'] = nearest_hub_indx
        else:
            hub_assignments_df['Nearest_hub_idx'].values[:] = 0

        if 'Nearest_hub_name' not in hub_assignments_df:
            hub_num = [('None')] * len(City.building_addr_df)
            hub_assignments_df['Nearest_hub_name'] = hub_num
        else:
            hub_assignments_df['Nearest_hub_name'].values[:] = 'None'
        
        if 'Weight' not in hub_assignments_df:
            hub_dist = [(m.inf)] * len(City.building_addr_df)
            hub_assignments_df['Weight'] = hub_dist
        else:
            hub_assignments_df['Weight'].values[:] = np.inf
        
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
        
        if 'Path_not_found' not in hub_assignments_df:
            value = [(False)] * len(City.building_addr_df)
            hub_assignments_df['Path_not_found'] = value
        else:
            hub_assignments_df['Path_not_found'].values[:] = False
        
        if 'Path' not in hub_assignments_df:
            value = [None] * len(City.building_addr_df)
            hub_assignments_df['Path'] = value
        else:
            hub_assignments_df['Path'].values[:] = None
        
        if 'Euclid_nearesthub' not in hub_assignments_df:
            hub_num = [('')] * len(City.building_addr_df)
            hub_assignments_df['Euclid_nearesthub'] = hub_num
        else:
            hub_assignments_df['Euclid_nearesthub'].values[:] = ''
        
        if 'Euclid_hubdistance' not in hub_assignments_df:
            euclid_hub_dist = [(m.inf)] * len(City.building_addr_df)
            hub_assignments_df['Euclid_hubdistance'] = euclid_hub_dist
        else:
            hub_assignments_df['Euclid_hubdistance'].values[:] = np.inf

        # Save to Clustering object
        self.hub_assignments_df = hub_assignments_df

    ### generate random points within the boundary of the loaded city at which to place hub locations
    def generate_random_points(self, coordinates_transformed_xy, start_pt_ct):
        print('adding points to hub dictionary', self.hub_list_dictionary)

        coordinates_as_tupple = []

        if self.hub_list_dictionary == None:
            hub_dictionary = OrderedDict()
            self.hub_list_dictionary = hub_dictionary

        #[N, S, E, W]
        #w_n_corner, e_n_corner, w_s_corner, e_s_corner
        coordinateX_min = coordinates_transformed_xy[0][0]
        coordinateX_max = coordinates_transformed_xy[1][0]
        coordinateY_min = coordinates_transformed_xy[2][1]
        coordinateY_max = coordinates_transformed_xy[0][1]

        index = len(self.hub_list_dictionary)
        
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

            coordinates_as_tupple.append((y,x))

            if index_name not in self.hub_list_dictionary:
                self.hub_list_dictionary[index_name] = {
                    "index": index,
                    "x": x,
                    "y": y,
                    "avg_time": 0,
                    "max_time": 0,
                    "people_served": 0,
                    "dest_edge": 0,
                    "yx_transform": 0,
                } 
            index += 1

        return coordinates_as_tupple
        

    ### cluster houses to each hub based on the euclidean distance to each hub
    def hub_clusters_euclidean(self, orig_yx_transf, name, data_folder):
        ### get hub locations from hub list dictionary
        hub_names, hub_yx_transf = get_yx_transf_from_dict(self.hub_list_dictionary)
        hub_yx_dict = dict(zip(hub_names, hub_yx_transf))

        for hub in hub_names:
            #get distance of all current hub to all houses        
            point2 = np.array(hub_yx_dict[hub])

            for i, house in enumerate(orig_yx_transf):
                point1 = np.array(house)
                euclid_dist = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
            
                if euclid_dist < self.hub_assignments_df.loc[i,'Euclid_hubdistance']:
                    self.hub_assignments_df.at[i,'Euclid_hubdistance']=euclid_dist
                    self.hub_assignments_df.at[i,'Euclid_nearesthub']=hub
        
        self.save_iteration(name, data_folder)

    ### move the hub based on travel distance of all houses in the hub cluster
    def new_hub_location(self, City):
        #! ensure this is weighted based on number of addresses at each location
        hub_dictionary = self.hub_list_dictionary
        for (hub_name, hub_data) in hub_dictionary.items():
            dist_moved = 0
            x = []
            y = []

            for i, row in self.hub_assignments_df.iterrows():
                if row['Nearest_hub_name'] == hub_name:
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
        #!add code that identifies how many are not yet assigned (if hubs need to be added)
        # dictionary: x, y, avg_time, people_served
        # find and save average time and people served to hub_dictionary
        hub_dictionary = self.hub_list_dictionary
        
        time_requirement = False
        max_time_list = []
        unassigned = 0
        
        for (hub_name, _) in hub_dictionary.items():
            all_times = []
            all_people = []
            for i, row in self.hub_assignments_df.iterrows():
                if row['Nearest_hub_name'] == hub_name:
                    #travel time
                    time = row['Weight']
                    all_times.append(time)
                    #people served
                    people = City.building_addr_df.iloc[i]['addr']
                    all_people.append(people)
                if row['Nearest_hub_name'] == None:
                    unassigned += 1

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
    
    def load_files_for_plot(path):
        cluster_iterations = []
        destinations = []
        closest_hub = []

        allfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        print(allfiles)

        return cluster_iterations, destinations, closest_hub