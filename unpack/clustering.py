"""
functions related to creating and optimizing hub locations

By: 
Lisa-Marie Mueller, TU Delft

Classes:
    - NetworkClustering

Methods:
    >>> save_iteration: Saves current object state to pickle file
    >>> save_dataframe: Saves current dataframe for plotting
    >>> load_dataframe: Loads dataframe for plotting
    >>> continue_clustering: Load previous clustering session and continue
    >>> reset_hub_df_values: Reset the data stored in the dataframe so that previous runs won't impact the current run
    >>> generate_points: Generate points within the boundary of the loaded city using k++ weighted locations at which to place hub locations
    >>> hub_clusters_euclidean: cluster houses to each hub based on the euclidean distance to each hub
    >>> new_hub_location: move the hub based on travel distance of all houses in the hub cluster
    >>> add_points: add new hub points using k++ weighted locations
    >>> hub_fitness: calculate the hub fitness value
    >>> optimize_location: adaptive, weighted k-means clustering implementation incorproating fitness and k++ location intialization
"""

import numpy as np
import math as m
import random
import pickle
import os.path
import pandas
import datetime

import multiprocessing as mp

import unpack.utils.network_helpers as h
from collections import OrderedDict
from .utils.network_helpers import *
from re import X
from .utils.multicore_shortest_path import *

class NetworkClustering():
    """A NewtworkClustering class that implements adaptive weighted K-means 
    clustering for the facility location problem. It can be used in 
    combination with the CityNetwork class and shortest path computations
    
    Developed by Lisa-Marie Mueller
    """
    
    def __init__(self, name: str):
        self.name = name
        self.session_name = ''
        self.iteration = 1
        self.kmeansstep = 0
        self.hub_list_dictionary = None
        self.hub_assignments_df = pandas.DataFrame()
        self.max_cores = None
        self.cluster_number = 0
        
    def __repr__(self):
        return "<Clustering object of {}>".format(self.name)

    def save_iteration(self, folder: str, step:str):
        """Save the object to a pickle file
        
        Developed by Lisa-Marie Mueller

        Parameters
        ----------
        folder : string
            Path to where pickle file should be stored.

        step : string
            Name for the step for legibility of saved files.
        """  
        session_name = self.session_name
        iteration = self.iteration 
        iteration_num = '{:03}'.format(iteration)
        folder_path = folder + session_name + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)   

        object_name = str(self.name) + '_iteration_' + str(iteration_num) + '_' + step
        path = folder_path + object_name + '.pkl'
        print('Saving {} to {}'.format(object_name + str('.pkl'), path))

        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        
        print('Iteration state saved as .pkl file')
    
    def save_dataframe(self, dataframe, hubs, k_means_iteration_i, state: str, folder: str, session_name: str):
        """Save dataframe with hub information for plotting
        
        Developed by Job de Vogel

        Parameters
        ----------
        dataframe : Pandas Dataframe
            Data to be stored.

        hubs : Dictionary
            Dictionary containing all hub information.
        
        k_means_iteration_i : String
            Current k-means iteration step.
        
        state : String
            Name for the step for legibility of saved files.

        folder : String
            Path to where pickle file should be stored.

        session_name : String
            Name of the session.
        """ 
        hub_locations = []
        for hub in hubs.values():
            locations = (hub['y'], hub['x'])
            hub_locations.append(locations)
        
        hubs = hub_locations
        
        path = folder + session_name + '/Dataframe/'
        name = 'iteration_' + str(self.iteration).zfill(2) + '_step_' + str(k_means_iteration_i) + '_' + state
        file_path = path + name + '.pkl'
        
        df_to_save = pd.DataFrame()
        
        df_to_save['Path'] = dataframe.loc[:, 'Path']
        df_to_save['Color_mask'] = dataframe.loc[:, 'Color_mask']
        
        data = (df_to_save, hubs, name)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)        
    
    @staticmethod
    def load_dataframe(path: str, session_name: str):
        """Load dataframe for printing
        
        Developed by Lisa-Marie Mueller

        Parameters
        ----------
        Path : String
            Path from which to load pickle file.

        session_name : String
            Name of the session.
        """ 

        cluster_iterations = []
        hubs_data = []
        file_names = []
        route_color_masks = []
        dest_color_masks = []
        dataframes = []
        
        allfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        
        for i, file in enumerate(allfiles):
            with open(path + file, 'rb') as f:
                dataframe, hubs, name = pickle.load(f)
            
            name = str(i).zfill(4) + '_' + session_name + '_' + name + '_' + str(i).zfill(4)         

            paths = dataframe.loc[:, 'Path']
            paths = paths.tolist()
    
            no_Nones = [i for (i, path) in enumerate(paths) if path != None]

            paths = dataframe.loc[no_Nones, 'Path'].tolist()
            
            if len(paths) > 0:
                cluster_iterations.append(paths)
                hubs_data.append(hubs)
                file_names.append(name)
                route_color_masks.append(dataframe.loc[no_Nones, 'Color_mask'].tolist())
                dest_color_masks.append(dataframe.loc[:, 'Color_mask'].tolist())
                dataframes.append(dataframe)
            else:
                cluster_iterations.append(None)
                hubs_data.append(hubs)
                file_names.append(name)
                route_color_masks.append(None)
                dest_color_masks.append(None)
                dataframes.append(None)
        
        return cluster_iterations, file_names, hubs_data, route_color_masks, dest_color_masks, dataframes
    
    def continue_clustering(self, path: str):
        """Load previous clustering session and continue
        
        Developed Lisa-Marie Mueller

        Parameters
        ----------

        Path : String
            Path to loaded pickle file.

        """ 
        if os.path.isfile(path):
            # Load a graph from drive
            print('Loading {}'.format(path))

            with open(path, 'rb') as file:
                last_state = pickle.load(file)
                self.name = last_state.name
                self.session_name = last_state.session_name
                self.iteration = last_state.iteration
                self.hub_list_dictionary = last_state.hub_list_dictionary
                self.hub_assignments_df = last_state.hub_assignments_df
        else:
            print('Could not locate {}'.format(path))
   
    ### Initialize DF, reset the data stored in the dataframe so that previous runs won't impact the current run
    def reset_hub_df(self, City):
        """Reset the data stored in the dataframe so that previous runs won't impact the current run
        
        Developed Lisa-Marie Mueller

        Parameters
        ----------

        City : Object
            CityNetwork object of the city being analyzed.

        """ 
        hub_assignments_df = self.hub_assignments_df

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

        if 'Color_mask' not in hub_assignments_df:
            value = [None] * len(City.building_addr_df)
            hub_assignments_df['Color_mask'] = value
        else:
            hub_assignments_df['Path'].values[:] = None

        # Save to Clustering object
        self.hub_assignments_df = hub_assignments_df

    ### generate random points within the boundary of the loaded city at which to place hub locations
    def generate_points(self, City, boundary_coordinates, start_pt_ct, orig_yx_transf, zero_people_hubs):
        """Generate random points within the boundary of the loaded city at which to place hub locations
        
        Developed Lisa-Marie Mueller

        Parameters
        ----------

        City : Object
            CityNetwork object of the city being analyzed.

        boundary_coordinates : List
            [N, S, E, W] coordinates denoting the boundaries of the city.

        start_pt_ct : Integer
            The number of hubs used to start clustering.
        
        orig_yx_transf : List of Tupples
            The hub coordinates listed as (y,x).

        zero_people_hubs : List
            A list of hub dictionary keys that denote which hubs have no people assigned.

        """ 
        # Print current hub_dictionary state
        if self.hub_list_dictionary is not None:
            print('Adding points to hub dictionary:')
            print(pd.DataFrame(dict(self.hub_list_dictionary)).T.astype({'index' : int}))
            print('')

        coordinates_as_tupple = []
        changed_points = []

        #[N, S, E, W]
        #w_n_corner, e_n_corner, w_s_corner, e_s_corner
        coordinateX_min = boundary_coordinates[0][0]
        coordinateX_max = boundary_coordinates[1][0]
        coordinateY_min = boundary_coordinates[2][1]
        coordinateY_max = boundary_coordinates[0][1]

        if self.hub_list_dictionary == None:
            print('No hub dictionary assigned yet, initializing hub dictionary')
            hub_dictionary = OrderedDict()
            self.hub_list_dictionary = hub_dictionary

            index = len(self.hub_list_dictionary)

            ### initialize first hub randomly
            x = random.uniform(coordinateX_min, coordinateX_max)
            y = random.uniform(coordinateY_min, coordinateY_max)

            index_name = 'hub ' + str(index)

            coordinates_as_tupple.append((y,x))

            if index_name not in self.hub_list_dictionary:
                self.hub_list_dictionary[index_name] = {
                    "index": index,
                    "x": x,
                    "y": y,
                    "avg_weight": 0,
                    "max_weight": 0,
                    "people_served": 0
                } 
            index += 1
            start_pt_ct -= 1

        else:
            index = len(self.hub_list_dictionary)
        
        if len(zero_people_hubs) > 0:
            for hub in zero_people_hubs:
                x = random.uniform(coordinateX_min, coordinateX_max)
                y = random.uniform(coordinateY_min, coordinateY_max)
                self.hub_list_dictionary[hub]['x'] = x
                self.hub_list_dictionary[hub]['x'] = y
                index = self.hub_list_dictionary[hub]['index']
                changed_points.append([index, (y,x)])
                print(changed_points)

        ### k-means ++ to initialize better starting hub locations
        for i in range(start_pt_ct):
            self.reset_hub_df(City)
            self.hub_clusters_euclidean(orig_yx_transf)
            max_dist = 0

            for i, row in self.hub_assignments_df.iterrows():
                if row['Euclid_hubdistance'] > max_dist:
                    x_max = City.building_addr_df.iloc[i]['x']
                    y_max = City.building_addr_df.iloc[i]['y']
                    max_dist = row['Euclid_hubdistance']
            
            for i, row in self.hub_assignments_df.iterrows():
                if row['Euclid_hubdistance'] > 0.7 * max_dist and row['Euclid_hubdistance'] < 0.8 * max_dist:
                    x = City.building_addr_df.iloc[i]['x']
                    y = City.building_addr_df.iloc[i]['y']
                    break
                else:
                    x = x_max
                    y = y_max

            index_name = 'hub ' + str(index)

            coordinates_as_tupple.append((y,x))

            if index_name not in self.hub_list_dictionary:
                self.hub_list_dictionary[index_name] = {
                    "index": index,
                    "x": x,
                    "y": y,
                    "avg_weight": 0,
                    "max_weight": 0,
                    "people_served": 0,
                } 
            index += 1

        return coordinates_as_tupple, changed_points

    ### cluster houses to each hub based on the euclidean distance to each hub
    def hub_clusters_euclidean(self, orig_yx_transf):
        """cluster houses to each hub based on the euclidean distance to each hub
        
        Developed Lisa-Marie Mueller

        Parameters
        ----------

        orig_yx_transf : List of Tupples
            The hub coordinates listed as (y,x).

        """ 
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

    ### move the hub based on travel distance of all houses in the hub cluster
    def new_hub_location(self, City, boundary_coordinates):
        """move the hub based on travel distance of all houses in the hub cluster
        
        Developed Lisa-Marie Mueller

        Parameters
        ----------

        City : Object
            CityNetwork object of the city being analyzed.

        boundary_coordinates : List
            [N, S, E, W] coordinates denoting the boundaries of the city.

        """ 
        hub_dictionary = self.hub_list_dictionary
        for (hub_name, _) in hub_dictionary.items():
            x = []
            y = []

            for i, row in self.hub_assignments_df.iterrows():
                if row['Nearest_hub_name'] == hub_name:
                    people_num = City.building_addr_df.iloc[i]['people']

                    house_x = City.building_addr_df.iloc[i]['x']
                    for j in range(m.ceil(people_num)):
                        x.append(house_x)

                    house_y = City.building_addr_df.iloc[i]['y']
                    for j in range(m.ceil(people_num)):
                        y.append(house_y)
   
            # If points are assigned:
            if len(x) > 0 and len(y) > 0:
                all_x = np.array(x)
                all_y = np.array(y)

                # Computer average x and y for this hub
                average_x = np.sum(all_x) / len(all_x)
                average_y = np.sum(all_y) / len(all_y)
            
                # Extract previous location
                previous_location = (hub_dictionary[hub_name]['x'], hub_dictionary[hub_name]['y'])
            
                # Set new x and y for this hub
                hub_dictionary[hub_name]['x'] = average_x
                hub_dictionary[hub_name]['y'] = average_y
                
                # Create new location coordinate
                new_location = (hub_dictionary[hub_name]['x'], hub_dictionary[hub_name]['y'])
            
            # The hubs changed in such a way that no houses are assigned to this hub
            # anymore, therefore generate a new random location.
            else:
                #[N, S, E, W]
                #w_n_corner, e_n_corner, w_s_corner, e_s_corner
                coordinateX_min = boundary_coordinates[0][0]
                coordinateX_max = boundary_coordinates[1][0]
                coordinateY_min = boundary_coordinates[2][1]
                coordinateY_max = boundary_coordinates[0][1]
                
                # Create a new random location
                x = random.uniform(coordinateX_min, coordinateX_max)
                y = random.uniform(coordinateY_min, coordinateY_max)
                
                # Set new x and y for this hub
                hub_dictionary[hub_name]['x'] = x
                hub_dictionary[hub_name]['y'] = y

                # Create new location coordinate
                new_location = (hub_dictionary[hub_name]['x'], hub_dictionary[hub_name]['y'])

            point1 = previous_location
            point2 = new_location
            
            # Compute distance between previous and new
            move_distance = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5 

        self.hub_list_dictionary = hub_dictionary

        return move_distance

    ### add new hub points using k++ weighted locations
    def add_points(self, City, boundary_coordinates, point_count, orig_yx_transf, zero_people_hubs):
        """add new hub points using k++ weighted locations
        
        Developed Lisa-Marie Mueller

        Parameters
        ----------

        City : Object
            CityNetwork object of the city being analyzed.

        boundary_coordinates : List
            [N, S, E, W] coordinates denoting the boundaries of the city.

        point_count : Integer
            The number of hubs to add.
        
        orig_yx_transf : List of Tupples
            The hub coordinates listed as (y,x).

        zero_people_hubs : List
            A list of hub dictionary keys that denote which hubs have no people assigned.

        """ 
        print('Adding ', point_count, ' point(s)')

        add_points, changed_points = self.generate_points(City, boundary_coordinates, point_count, orig_yx_transf, zero_people_hubs)

        return add_points, changed_points

    ### calculate the hub fitness
    def hub_fitness(self, City, max_travel_time, max_people_served, capacity_factor, max_unassigned_percent, max_long_walk_percent, session_name):
        """ Calculate the hub fitness
        
        Developed Lisa-Marie Mueller, Job de Vogel

        Parameters
        ----------

        City : Object
            CityNetwork object of the city being analyzed.

        max_travel_time : Float
            Maximum travel time in seconds.

        max_people_served : Integer
            Maximum number of people that can be served by one hub.
        
        capacity_factor : Float
            Percentage of people assigned to a hub that can exceed the capacity.

        max_unassigned_percent : Float
            Percentage of people that can remain unassigned.
            
        max_long_walk_percent : Float
            Percentage of people assigned to a hub that can have longer travel times.

        session_name : String
            Name of the session.
        """ 
        # find and save average time and people served to hub_dictionary
        hub_dictionary = self.hub_list_dictionary
        fitness_check = False
        k_check = False

        max_time_list = []
        capacity_list = []
        average_list = []
        std_list = []
        city_wide_people_served = sum(City.building_addr_df.loc[:, 'people'].tolist())
        unassigned = 0

        zero_people_hubs = []

        long_travel = True

        # Calculate how many people are not assigned
        for i, row in self.hub_assignments_df.iterrows():
            if row['Nearest_hub_name'] == None:
                unassigned += City.building_addr_df.iloc[i]['people']

        # For each hub
        for (hub_name, _) in hub_dictionary.items():
            all_times = []
            all_people = []
            weights = []
            people_at_hub = 0
            num_people_per_hub = []
            
            # Iterate over the houses connected to the hub
            for i, row in self.hub_assignments_df.iterrows():
                if row['Nearest_hub_name'] == hub_name:
                    #travel time
                    time = row['Weight']
                    all_times.append(time)
                    people = City.building_addr_df.iloc[i]['people']
                    all_people.append(people)
                    num_people_per_hub.append(people)
                    
                    weights.append(time)
                    people_at_hub += people                     

            if len(all_times) == 0:
                zero_people_hubs.append(hub_name)
                
            long_people = 0.
            for weight, people in zip(weights, num_people_per_hub):
                if weight > max_travel_time:
                    long_people += people
            
            if long_people > (people_at_hub * max_long_walk_percent):
                long_travel = False

            # All times for one hub, excluding None
            all_times_np = np.array([time for time in all_times if time != None])
            
            # Average time for one hub
            if len(all_times) > 0:
                average = np.sum(all_times_np) / len(all_times)
                average_list.append(average)
            else:
                average = 0
                average_list.append(0)
            
            # People served in this hub
            if len(all_people) > 0:
                all_people_np = np.array(all_people)
                total_people = np.sum(all_people_np)
            else:
                total_people = 0
                
            capacity_list.append(total_people)

            if len(all_times_np) > 0:
                # Maximum time for one hub
                max_time_list.append(np.max(all_times_np))
                
                # Standard deviation time for one hub
                std = np.std(all_times_np)
                std_list.append(std)
            else:
                max_time_list.append(0)
                std_list.append(0)
            
            hub_dictionary[hub_name]['avg_weight'] = average
            hub_dictionary[hub_name]['std_weight'] = std
            
            if len(all_times_np) > 0:
                hub_dictionary[hub_name]['max_weight'] = np.max(all_times_np)
            else:
                hub_dictionary[hub_name]['max_weight'] = 0
                
            hub_dictionary[hub_name]['people_served'] = total_people             

        max_time_list_np = np.array(max_time_list)
        
        # If all the max times are smaller than allowed, then time_check is True
        time_check = all(i <= max_travel_time for i in max_time_list_np)

        # If all capacities are lower than max_people_served per hub
        capacity_np = np.array(capacity_list)
        if len(capacity_np) > 0: 
            capacity_check = all(i <= max_people_served for i in capacity_np)
            k_check = all(i <= capacity_factor * max_people_served for i in capacity_np)
        else:
            capacity_check = False
            k_check = False

        # If unassigned higer than max allowed
        unassigned_check = False
        if unassigned < (max_unassigned_percent * city_wide_people_served):
            unassigned_check = True

        # Check which fitness checks are True
        if (time_check or long_travel) and capacity_check and unassigned_check:
            fitness_check = True
        
        # Print the results for all hubs
        for i in range(len(average_list)):
            print('Hub {}: avg. weight: {}, std. weight: {}, max weight {}, people served per hub: {}'.format(i, round(average_list[i], 2), round(std_list[i], 2), round(max_time_list[i], 2), m.ceil(capacity_list[i])))

        # Print the fitness checks
        print('')
        print(f'Time_check: {time_check}, Long_travel_check: {long_travel}, Capacity_check: {capacity_check}, Unassigned_check: {unassigned_check}')
        print(f"Unassigned people out of total: {m.ceil(unassigned)}/{m.ceil(city_wide_people_served)}")
        print(f"Avg. weight: {round(np.sum(np.array(average_list)) / len(average_list), 2)}, avg. std: {round(np.sum(np.array(std_list)) / len(std_list), 2)}, avg. people served: {m.ceil(np.sum(np.array(capacity_list)) / len(capacity_list))}")
        print('')

        # If hubs have zero people assigned, print
        if len(zero_people_hubs)> 0: 
            print('The following hubs have no users assigned: {}. This/these hub location(s) will be replaced with new location(s) in next itteration.'.format(zero_people_hubs))

        # Log iteration
        run_name = City.data_folder + 'log_' + session_name + '.txt'
        with open(run_name, 'a') as file:
            file.writelines(f'Logtime: {datetime.datetime.now()}\n')
            for i in range(len(average_list)):
                info = 'Hub {}: avg. weight: {}, std. weight: {}, max weight {}, people served per hub: {}'.format(i, round(average_list[i], 2), round(std_list[i], 2), round(max_time_list[i], 2), m.ceil(capacity_list[i]))
            
                file.writelines(info + '\n')
            
            file.writelines('\n')
            file.writelines(f'Time_check: {time_check}, Long_travel_check: {long_travel}, Capacity_check: {capacity_check}, Unassigned_check: {unassigned_check} \n')
            file.writelines(f"Unassigned people out of total: {m.ceil(unassigned)}/{m.ceil(city_wide_people_served)}\n")
            file.writelines(f"Avg. weight: {round(np.sum(np.array(average_list)) / len(average_list), 2)}, avg. std: {round(np.sum(np.array(std_list)) / len(std_list), 2)}, avg. people served: {m.ceil(np.sum(np.array(capacity_list)) / len(capacity_list))}\n")
            file.writelines('----------------\n')
            file.writelines('\n')
            
        return fitness_check, time_check, k_check, zero_people_hubs

    def optimize_locations(self, City, session_name, data_folder, boundary_coordinates, destinations, weight_input, cutoff_input, skip_non_shortest_input, skip_treshold_input, start_pt_ct, point_count, max_weight, max_cpu_count, hub_colors, 
        calc_euclid=False, 
        max_distance=100, 
        max_additional_clusters=3, 
        max_iterations=4, 
        max_people_served=6570, 
        capacity_factor=1.2, 
        distance_decrease_factor=0.95, 
        max_unassigned_percent=0.1, 
        max_long_walk_percent=0.1):

        """ Calculate the hub fitness
        
        Developed Lisa-Marie Mueller, Job de Vogel

        Parameters
        ----------

        City : Object
            CityNetwork object of the city being analyzed.

        session_name : String
            Name of the session.
        
        data_folder : String
            Path to where pickle file should be stored.

        boundary_coordinates : List
            [N, S, E, W] coordinates denoting the boundaries of the city.

        destinations : List
            List of house locations from CityNetwork object

        weight_input : String
            Type of calculation to use for calculating shortest paths

        cutoff_input : Integer
            Distance at which to stop calcuating shortest paths

        skip_non_shortest_input : Boolean
            Option to stop calculating when a path is longer than the shortest path, less accurate but saves on computation

        skip_treshold_input : Boolean
            Option to skip inputing a threshold
        
        start_pt_ct : Integer
            The number of hubs used to start clustering

        point_count : Integer
            The number of hubs to add.

        max_weight : Float
            Maximum travel time in seconds.
        
        max_cpu_count : Integer or None
            Maximum number of cpus to use.
        
        hub_colors : List
            List of colours to use for plotting.

        calc_euclid : Boolean
            Calculate euclidean distance during first iteration.

        max_distance : Integer
            Maximum distance that the hub assignement algorithm extends out from hubs.
        
        max_additional_clusters : Integer
            Maximum number of hubs that are added after the initialization.
        
        max_iterations : Integer
            Maximum number of k-means clustering itterations happen each round.

        max_people_served : Integer
            Maximum number of people that can be served by one hub.
        
        capacity_factor : Float
            Percentage of people assigned to a hub that can exceed the capacity.
        
        distance_decrease_factor : Float
            Decrease maximum walk distance by factor

        max_unassigned_percent : Float
            Percentage of people that can remain unassigned.
            
        max_long_walk_percent : Float
            Percentage of people assigned to a hub that can have longer travel times.
        """ 

        # Extract data
        dest_edges = City.ne

        #initialize variables
        iteration = self.iteration
        time_check = False
        fitness_check = False
        self.session_name = session_name
        max_iterations_active = max_iterations
        zero_people_hubs = []

        # Catch for small number of clusters
        if max_additional_clusters < 2: max_additional_clusters = 2

        Dataframe_path = data_folder + session_name + '/Dataframe/'
        if not os.path.exists(Dataframe_path):
            os.makedirs(Dataframe_path)

        while not fitness_check and iteration < max_additional_clusters:
            print(f'Running k iteration {start_pt_ct - 1 + iteration} step 0...')
            
            ### reset all DF values to default for iteration and update class properties
            if max_additional_clusters > 50: max_additional_clusters = 50
            self.reset_hub_df(City)
            self.iteration = iteration
            if max_cpu_count is None: max_cpu_count = mp.cpu_count()
            self.max_cores = max_cpu_count
            ### save class object to restart clustering for an iteration
            self.save_iteration(data_folder, 'initialize')
            
            if iteration == 1:
                # update number of CPUs to use based on number of clusters
                if self.max_cores > self.hub_assignments_df.shape[0]: 
                    cpu_count = self.hub_assignments_df.shape[0]
                else: 
                    cpu_count = self.max_cores

                ### only on first iteration generate random points for hubs and cluster houses based on closest hub
                hubs,_ = self.generate_points(City, boundary_coordinates, start_pt_ct, destinations, zero_people_hubs)
                self.cluster_number = start_pt_ct

                self.save_dataframe(self.hub_assignments_df, self.hub_list_dictionary, 0, 'locations', data_folder, session_name)
                
                # Print results
                print('')
                print(pd.DataFrame(dict(self.hub_list_dictionary)).T.astype({'index' : int}))
                print('')
                
                paths = multicore_single_source_shortest_path(City.graph, self.hub_list_dictionary, destinations, dest_edges,
                    skip_non_shortest=skip_non_shortest_input, 
                    skip_treshold=skip_treshold_input,
                    weight=weight_input, 
                    cutoff=cutoff_input, 
                    cpus=cpu_count
                    )

                self.hub_assignments_df = paths_to_dataframe(paths, hub_colors, hubs=self.hub_list_dictionary)

                if calc_euclid:
                    self.hub_clusters_euclidean(destinations)
                    self.save_dataframe(self.hub_assignments_df, self.hub_list_dictionary, 1, 'cluster_euclid', data_folder, session_name)
                
                self.save_dataframe(self.hub_assignments_df, self.hub_list_dictionary, 1, 'cluster', data_folder, session_name)
                self.max_cores = cpu_count
                
            else:        
                ### on all other iterations, add a new hub each time the while loop runs 
                additional_points, changed_points = self.add_points(City, boundary_coordinates, point_count, destinations, zero_people_hubs)

                self.cluster_number += point_count
                print('Added hub(s):')
                print(pd.DataFrame(dict(self.hub_list_dictionary)).T.astype({'index' : int}))
                print('')
                
                self.save_dataframe(self.hub_assignments_df, self.hub_list_dictionary, 0, 'locations', data_folder, session_name)

                # update number of CPUs to use based on number of clusters
                if self.max_cores > self.hub_assignments_df.shape[0]: 
                    cpu_count = self.hub_assignments_df.shape[0]
                else: 
                    cpu_count = self.max_cores

                paths = multicore_single_source_shortest_path(City.graph, self.hub_list_dictionary, destinations, dest_edges,
                    skip_non_shortest=skip_non_shortest_input, 
                    skip_treshold=skip_treshold_input,
                    weight=weight_input, 
                    cutoff=cutoff_input, 
                    cpus=cpu_count
                    )

                self.hub_assignments_df = paths_to_dataframe(paths, hub_colors, hubs=self.hub_list_dictionary)

                if calc_euclid:
                    self.hub_clusters_euclidean(destinations)
                    self.save_dataframe(self.hub_assignments_df, self.hub_list_dictionary, 1, 'clusters_euclid', data_folder, session_name)
                
                self.save_dataframe(self.hub_assignments_df, self.hub_list_dictionary, 1, 'clusters', data_folder, session_name)
                self.max_cores = cpu_count

            print('------------------------------------\n')

            ###optimize hub locations
            move_distance = self.new_hub_location(City, boundary_coordinates)
            i = 0
            self.kmeansstep = i
            while move_distance > max_distance and i < max_iterations_active:
                print(f'Running k iteration {start_pt_ct - 1 + iteration} optimization step {i + 1}...')
                
                paths = multicore_single_source_shortest_path(City.graph, self.hub_list_dictionary, destinations, dest_edges,
                    skip_non_shortest=skip_non_shortest_input, 
                    skip_treshold=skip_treshold_input,
                    weight=weight_input, 
                    cutoff=cutoff_input, 
                    cpus=cpu_count
                    )

                self.hub_assignments_df = paths_to_dataframe(paths, hub_colors, hubs=self.hub_list_dictionary)        
                
                if calc_euclid: self.hub_clusters_euclidean(destinations)
                move_distance = self.new_hub_location(City, boundary_coordinates)

                self.save_dataframe(self.hub_assignments_df, self.hub_list_dictionary, i+2, 'kmeans', data_folder, session_name) 
                
                print(f'Moved hubs on average {round(move_distance, 2)} meters')
                _, _, k_check, _ = self.hub_fitness(City, max_weight, max_people_served, capacity_factor, max_unassigned_percent, max_long_walk_percent, session_name)

                if not k_check: max_iterations_active = 2

                i += 1
                self.kmeansstep = i
                print('------------------------------------\n')
            
            self.save_dataframe(self.hub_assignments_df, self.hub_list_dictionary, i+2, 'clusters_final', data_folder, session_name) 
            
            ###check fitness function
            fitness_check, _, _, zero_people_hubs = self.hub_fitness(City, max_weight, max_people_served, capacity_factor, max_unassigned_percent, max_long_walk_percent, session_name)

            iteration += 1
            self.iteration = iteration
            
            if max_distance*distance_decrease_factor > 0: 
                max_distance = max_distance*distance_decrease_factor
            else: 
                max_distance = 1
            
            max_iterations_active = max_iterations
            max_iterations_active += 10
                
            print(f'Max distance updated to {round(max_distance, 2)}')
            print('------------------------------------\n')