"""
functions related to creating and optimizing hub locations

By: 
Job de Vogel, TU Delft (generate and load network)
Lisa-Marie Mueller, TU Delft (main, hub clustering and cluster optimization)

Classes:
    - None

Functions:
    - generate_network: generates the network based on the inputs provided in the main function including the location
    - load_network: loads a previously generated network


Other functions:
    - main: optimizes hub locations
"""

import matplotlib.pyplot as plt
import numpy as np
import math as m
import osmnx as ox
import random
import time

import utils.clustering as c
import utils.network_helpers as h
import utils.visualizations as v
from re import X
from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import transform_coordinates
from utils.multicore_nearest_edges import multicore_nearest_edge


def generate_network(name, data_folder, vehicle_type, coordinates):
    ### Job's Code: generate network function
    ''' --- GENERATE NETWORK ---
    Generate a new network using the functions in the CityNetwork class. If a network already has been generated and stored in the data folder, comment this part and continue with PREPARE NETWORK. '''
    # Initialize CityNetwork object [N, S, E, W]
    City = CityNetwork(name, coordinates, vehicle_type)
    
    # Load osm from local or online file
    City.load_osm_graph(data_folder + name + '.osm')
    City.load_building_addr(data_folder + name + '_building_addresses.csv', 
        data_folder + name + '_buildings.csv', 
        data_folder + name + '_addresses.csv',
        data_folder +'runtime/'+ name + '_cbs.xml')

    # Add speeds, lengths and distances to graph
    # Overwrite speed by using overwrite_bike=16
    # Further types available: overwrite_walk and overwrite_epv
    City.add_rel_attributes(overwrite_bike=16)

    # Project graph
    City.project_graph()

    # Plot the CityNetwork
    # City.plot()

    # Calculate dataframes of nodes and edges
    City.convert_graph_edges_to_df()
    City.convert_graph_nodes_to_df()

    # Save Pickle file
    City.save_graph(name, data_folder)
    print('------------------------------------')


def load_network(name, data_folder):
    ###Job's Code load network function
    ''' --- PREPARE NETWORK ---
    Load the network from .pkl file. Transform the origin and destination to the unprojected space.'''

    # Load the CityNetwork class object
    City = CityNetwork.load_graph(name, data_folder)

    # Specify the coordinates for origin and destination
    # Get the coordinates from building dataframe   
    coordinates = City.building_addr_df.loc[:, ['latitude', 'longitude']]

    # Convert the coordinates to tuples, destinations are one goal, will be hubs
    origins = list(coordinates.itertuples(index=False, name=None))

    # Extract the graph from the City CityNetwork
    graph = City.graph
    #print(f"Processing {graph}")

    orig_yx_transf = []

    # Transform the start and origin to epsg:3857 (surface)
    # for index, row in City.building_addr_df.T.items():
    for index, row in City.building_addr_df.iterrows():
        coordinate = row['y'], row['x']
        orig_yx_transf.append(coordinate)

    return City, orig_yx_transf

@timer_decorator
def main():
    #initialization variables
    cpu_count = None
    name = 'delft_walk'
    data_folder = 'data/'
    vehicle_type = 'walk' # walk, bike, drive, all (See osmnx documentation)
    max_travel_time = 150 #time in seconds
    random_init = 100

    ### Delft City Center #coordinates = [52.018347, 52.005217, 4.369142, 4.350504]
    ### Delft #coordinates = [52.03, 51.96, 4.4, 4.3]
    #[N, S, E, W] coordiantes of location to analyse
    coordinates = [52.018347, 52.005217, 4.369142, 4.350504]

    start_pt_ct = 3 # number of starting hubs
    calc_euclid = False #also calculate euclidean distances for first iteration?

    hub_colors = ['#FFE54F', '#82C5DA', '#C90808', '#FAA12A', '#498591', '#E64C4C', '#E4FFFF', '#CA5808', '#3F4854']
    
    ###### ------- ######
    
    random.seed(random_init)

    coordinates_list = h.coordinates_to_tupple(coordinates)
    coordinates_list_flip = h.flip_lat_long(coordinates_list)
    coordinates_transformed_yx = transform_coordinates(coordinates_list_flip)
    coordinates_transformed_xy = h.flip_lat_long(coordinates_transformed_yx)

    ### generate new network, only run at beginning
    generate_network(name, data_folder, vehicle_type, coordinates)

    ### load network from file, run after network is generated
    City, orig_yx_transf = load_network(name, data_folder)
    c.reset_hub_df_values(City)
    
    #City.ne = None

    orig_edges = h.nearest_edges_buildings(City, orig_yx_transf, name, data_folder, cpu_count)

    #initialized variables - do not change
    iteration = 1
    time_check = False
    max_time_list = m.inf 

    #option to change/adjust these as clusering settings
    point_count = 1 #the number of new hubs to add when the fitness is not reached
    max_distance = 50 #when distance the hub moves during k-means clustering, is less than max_distance, the hub will stop moving
    max_iterations = 30 #maximum iterations in case the max_distance moves a lot for too long

    while not time_check and iteration < 10:
        ### reset all DF values to default for iteration
        print('iteration: ', iteration)
        c.reset_hub_df_values(City)

        if iteration == 1:
            ### only on first iteration generate random points for hubs and cluster houses based on closest hub
            hub_dictionary = c.generate_random_points(coordinates_transformed_xy, start_pt_ct)
            #v.visualize_clusters(City, hub_dictionary, 'hub_locations_1', hub_colors, save=True)
            print(hub_dictionary) 
            c.hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count)
            if calc_euclid:
                c.hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder)
                v.euclid_visualize_clusters(City, hub_dictionary, hub_colors, save=True)
            #v.visualize_clusters(City, hub_dictionary, 'hub_clusters_1', hub_colors, save=True)

        else:
            ### on all other iterations, add a new hub each time the while loop runs 
            c.add_points(point_count, hub_dictionary, coordinates_transformed_xy)
            print('added hub', hub_dictionary)
            image_title1 = 'hub_locations_' + str(iteration)
            #v.visualize_clusters(City, hub_dictionary, image_title1, hub_colors, save=True)
            c.hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count)
            if calc_euclid:
                c.hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder)
            image_title2 = 'hub_locations_' + str(iteration)
            #v.visualize_clusters(City, hub_dictionary, image_title2, hub_colors, save=True)

        ###optimize hub locations
        move_distance = c.new_hub_location(City, hub_dictionary)
        i = 0
        while move_distance > max_distance and i < max_iterations:
            c.hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count)
            c.hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder)
            move_distance = c.new_hub_location(City, hub_dictionary)
            print('moved hubs on average ', move_distance)

            i += 1
        image_title = 'optimized_hub_locations_' + str(iteration)
        v.visualize_clusters(City, hub_dictionary, image_title, hub_colors, save=True)
        
        ###check fitness function
        time_check, max_time_list = c.hub_fitness(City, hub_dictionary, max_travel_time)
        iteration += 1
        max_distance -= 3
        max_iterations += 10
    
if __name__ == '__main__':
    main()