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

# Add parent directiory to path
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

import math as m
import random
import pandas as pd

import unpack.utils.network_helpers as h
from re import X
import unpack
from unpack.city_network import CityNetwork
from unpack.utils.utils.timer import timer_decorator
from unpack.clustering import NetworkClustering
from unpack.utils.multicore_shortest_path import transform_coordinates

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
    max_cpu_count = None #input the maximum CPU cores, use None to automatically set to maximum available
    network_name = 'delft_center_walk'
    cluster_name = 'delft_center_cluster'
    data_folder = 'data/'
    vehicle_type = 'walk' # walk, bike, drive, all (See osmnx documentation)
    network_scale = 'medium' #set to 'large', 'medium', or 'small' based on the city size
    max_travel_time = 150 #time in seconds
    #! per our call update these to adjust hub capacity and to restart clustering faster when k isn't close to the right amount
    max_people_served = 6570 #one hub has the capacity for 2190 packages, 6570 capacity assumes each person receives a package once every 3 days
    capacity_factor = 1.2 #add a factor that checks if the number of clusters is close to the required number, if it is not, then it will add more hubs with minimal hub location optimization
    #capacity_factor is multiplication based so entering 2 will means if the hub capacity is greater than 2 * max_people_served, then k will increase
    distance_decrease_factor = 0.9
    random_init = 101
    #shortest path settings
    skip_non_shortest = False
    skip_treshold = 60
    weight = 'travel_time'
    cutoff = None
    max_unassigned_percent = 0.1 #percentage of people that can be unassigned to a hub
    max_long_walk_percent = 0.1 #percentage of people that can have long walks to hub

    # Give a name, to avoid overwriting your plots
    session_name = input("Please insert a unique name for this clustering session: ")

    ### Delft City Center #coordinates = [52.018347, 52.005217, 4.369142, 4.350504]
    ### Delft #coordinates = [52.03, 51.96, 4.4, 4.3]
    #[N, S, E, W] coordiantes of location to analyse
    coordinates = [52.018347, 52.005217, 4.369142, 4.350504]
    #coordinates = [52.03, 51.96, 4.4, 4.3]

    #! Increase the start_pt_ct to increase the number of hubs to initialize with, for larger networks set to a larger number
    start_pt_ct = 3 # number of starting hubs
    calc_euclid = False #also calculate euclidean distances for first iteration
    generate_new_network = False # if true, new network will be generated, otherwise set to false to only load a network
    continue_clustering = False # if clustering crashed or finished, set to true to load the last iteration
    visualize_clustering = True # set to true to generate clustering images

    #enter color pallet to use for plotting
    hub_colors = ['red', 'tomato', 'orangered', 'lightsalmon', 'indianred', 'firebrick', 'gold', 'darkorange', 'yellow', 'yellowgreen', 'greenyellow', 'darkolivegreen', 'lawngreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'seagreen', 'darkslategray', 'blue', 'deepskyblue', 'steelblue', 'dodgerblue', 'skyblue', 'cornflowerblue', 'royalblue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'darkviolet', 'mediumorchid' 'purple', 'darkmagenta', 'mediumvioletred', 'deeppink', 'palevioletred', 'crimson']

    ###### ------- ######
    
    random.seed(random_init)

    coordinates_list = h.coordinates_to_tupple(coordinates)
    coordinates_list_flip = h.flip_lat_long(coordinates_list)
    coordinates_transformed_yx = transform_coordinates(coordinates_list_flip)
    coordinates = h.flip_lat_long(coordinates_transformed_yx)

    ### generate new network, only runs if variable is set to true
    if generate_new_network: generate_network(network_name, data_folder, vehicle_type, coordinates)

    ### load network from file, run after network is generated
    City, orig_yx_transf = load_network(network_name, data_folder)
    
    destinations = City.get_yx_destinations()
    # dest_edges = City.nearest_edges(5, cpus=12)
    # City.save_graph(network_name, data_folder)
    dest_edges = City.ne

    print('------------------------------------\n')
    ###initiate clustering
    Clusters = NetworkClustering(cluster_name)
    Clusters.reset_hub_df(City)

    ### if clustering crashed or finishes, load the latest iteration
    if continue_clustering:
        path = input("Please insert the file name that you would like to load to continue clustering: ")
        Clusters.continue_clustering(path)
        print('Starting with iteration ', Clusters.iteration)

    #option to change/adjust these as clusering settings
    point_count = 1 #the number of new hubs to add when the fitness is not reached

    Clusters.optimize_locations(City, session_name, data_folder, coordinates, destinations, weight, cutoff, skip_non_shortest, skip_treshold, start_pt_ct, point_count, max_travel_time, max_cpu_count, hub_colors, max_additional_clusters=7, max_iterations=6)

    dataframes_path = data_folder + session_name + '/Dataframe/'
    cluster_iterations, file_names, hubs, route_color_masks, dest_color_masks, dataframes = Clusters.load_dataframe(dataframes_path, session_name)
    
    print('Final hub locations and information:')
    
    destinations = []
    for i, row in City.building_addr_df.iterrows():
        destinations.append((row['y'], row['x']))
    
    if visualize_clustering: unpack.multiplot_save(City, cluster_iterations, hubs, destinations, file_names, route_color_masks, hub_colors, dest_color_masks=dest_color_masks, cpus=12)
    
if __name__ == '__main__':
    main()