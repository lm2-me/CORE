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

import math as m
import random
import pandas as pd

import unpack.utils.network_helpers as h
from re import X
import unpack
from unpack.network_delft import CityNetwork
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
    random_init = 100
    #shortest path settings
    skip_non_shortest_input = False
    skip_treshold_input = 60
    weight_input = 'travel_time'
    cutoff_input = None

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
    hub_colors = ['#FFE54F', '#82C5DA', '#C90808', '#FAA12A', '#498591', '#E64C4C', '#E4FFFF', '#CA5808', '#3F4854']

    ###### ------- ######
    
    random.seed(random_init)

    coordinates_list = h.coordinates_to_tupple(coordinates)
    coordinates_list_flip = h.flip_lat_long(coordinates_list)
    coordinates_transformed_yx = transform_coordinates(coordinates_list_flip)
    coordinates_transformed_xy = h.flip_lat_long(coordinates_transformed_yx)

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
    #! Decrease max_distance to make the hub locations more accurate earlier (this variable is automatically decreased each itteration so as the hubs are more accurately placed as they get closer to the optimila number)
    max_distance = 50 #when distance the hub moves during k-means clustering, is less than max_distance, the hub will stop moving
    #! Increase max_iterations if the k-means clustering stops moving the hubs too fast
    max_iterations = 50 #maximum iterations in case the max_distance moves a lot for too long
    #! Increase the max_additional_clusters to allow the algorithm to place more hubs after the initialization
    #! Based on the network_scale input, this variable will be reset to the following maximums (if the value set is less than the maximum, the entered value will be used): 50 if small network scale, 100 if medium network scale, 200 if large network scale
    max_additional_clusters = 20 #after adding this many new locations, the algorithm will stop use this to cut the algorithm before the optimial solution if time is a concern, if set above 50 it will be reset to 50

    Clusters.optimize_locations(City, session_name, data_folder, start_pt_ct, coordinates_transformed_xy, destinations, 
        dest_edges, skip_non_shortest_input, skip_treshold_input, weight_input, cutoff_input, max_additional_clusters, 
        calc_euclid, orig_yx_transf, point_count, max_travel_time, max_distance, max_iterations, max_cpu_count, hub_colors, 
        network_scale, max_people_served, capacity_factor, distance_decrease_factor)

    print_df_files_path = data_folder + session_name + '/Dataframe/'
    cluster_iterations, file_name, hubs, title, colors = Clusters.load_files_for_plot(print_df_files_path)
    
    print('Final hub locations and information:')
    print(pd.DataFrame(dict(Clusters.hub_list_dictionary)).T.astype({'index' : int}))
    
    destinations = []
    for i, row in City.building_addr_df.iterrows():
        destinations.append((row['y'], row['x']))

    if visualize_clustering: unpack.multiplot_save(City, cluster_iterations, hubs, destinations, file_name, colors, hub_colors, cpus=None)
    
if __name__ == '__main__':
    main()