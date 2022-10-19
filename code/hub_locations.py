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

import utils.network_helpers as h
import utils.visualizations as v
from re import X
from network_delft import CityNetwork, timer_decorator
from clustering import NetworkClustering
from utils.multicore_shortest_path import transform_coordinates


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
    network_name = 'delft_walk'
    cluster_name = 'delft_cluster'
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
    generate_new_network = True # if true, new network will be generated, otherwise set to false to only load a network
    load_clustering = True # if clustering crashed or finished, set to true to load the last iteration
    visualize_clustering = True # if clusering images are generated

    hub_colors = ['#FFE54F', '#82C5DA', '#C90808', '#FAA12A', '#498591', '#E64C4C', '#E4FFFF', '#CA5808', '#3F4854']
    
    ###### ------- ######
    
    random.seed(random_init)

    coordinates_list = h.coordinates_to_tupple(coordinates)
    coordinates_list_flip = h.flip_lat_long(coordinates_list)
    coordinates_transformed_yx = transform_coordinates(coordinates_list_flip)
    coordinates_transformed_xy = h.flip_lat_long(coordinates_transformed_yx)

    ### generate new network, only run at beginning
    if generate_new_network: generate_network(network_name, data_folder, vehicle_type, coordinates)

    ### load network from file, run after network is generated
    City, orig_yx_transf = load_network(network_name, data_folder)

    ###initiate clustering
    Clusters = NetworkClustering('delftclustering')
    Clusters.reset_hub_df(City)

    ### if clustering crashed or finishes, load the latest iteration
    if load_clustering: Clusters.load_clustering(cluster_name, data_folder)
    print('Starting with iteration ', Clusters.iteration)
    
    #City.ne = None

    orig_edges = h.nearest_edges_buildings(City, orig_yx_transf, network_name, data_folder, cpu_count)

    #initialized variables - do not change
    iteration = Clusters.iteration
    time_check = False
    max_time_list = m.inf 

    #option to change/adjust these as clusering settings
    point_count = 1 #the number of new hubs to add when the fitness is not reached
    max_distance = 50 #when distance the hub moves during k-means clustering, is less than max_distance, the hub will stop moving
    max_iterations = 30 #maximum iterations in case the max_distance moves a lot for too long

    while not time_check and iteration < 10:
        ### reset all DF values to default for iteration
        print('iteration: ', iteration)
        Clusters.reset_hub_df(City)
        Clusters.iteration = iteration
        Clusters.save_iteration(cluster_name, data_folder)

        if iteration == 1:
            ### only on first iteration generate random points for hubs and cluster houses based on closest hub
            Clusters.generate_random_points(coordinates_transformed_xy, start_pt_ct)
            if visualize_clustering: v.visualize_clusters(City, Clusters, 'locations_1', hub_colors, save=True)
            print(Clusters.hub_list_dictionary) 
            Clusters.hub_clusters(City, orig_yx_transf, orig_edges, cluster_name, data_folder, cpu_count)
            if calc_euclid:
                Clusters.hub_clusters_euclidean(orig_yx_transf, cluster_name, data_folder)
                if visualize_clustering: v.euclid_visualize_clusters(City, Clusters, 'clusters_1_euclid', hub_colors, save=True)
            if visualize_clustering: v.visualize_clusters(City, Clusters, 'clusters_1', hub_colors, save=True)

        else:
            ### on all other iterations, add a new hub each time the while loop runs 
            Clusters.add_points(point_count, coordinates_transformed_xy)
            print('added hub', Clusters.hub_list_dictionary)
            image_title1 = 'locations_' + str(iteration)
            if visualize_clustering: v.visualize_clusters(City, Clusters, image_title1, hub_colors, save=True)
            Clusters.hub_clusters(City, orig_yx_transf, orig_edges, cluster_name, data_folder, cpu_count)
            if calc_euclid:
                Clusters.hub_clusters_euclidean(orig_yx_transf, cluster_name, data_folder)
                image_title_euclid = 'clusters_' + str(iteration) + '_euclid'
                if visualize_clustering: v.euclid_visualize_clusters(City, Clusters, image_title_euclid, hub_colors, save=True)
            image_title2 = 'clusters_' + str(iteration)
            if visualize_clustering: v.visualize_clusters(City, Clusters, image_title2, hub_colors, save=True)

        ###optimize hub locations
        move_distance = Clusters.new_hub_location(City)
        i = 0
        while move_distance > max_distance and i < max_iterations:
            Clusters.hub_clusters(City, orig_yx_transf, orig_edges, cluster_name, data_folder, cpu_count)
            if calc_euclid: Clusters.hub_clusters_euclidean(orig_yx_transf, cluster_name, data_folder)
            move_distance = Clusters.new_hub_location(City)
            print('moved hubs on average ', move_distance)

            i += 1

        ### visualized optimized hub locations
        image_title = 'clusters_' + str(iteration) + '_optimized'
        image_title_euclid = 'clusters_' + str(iteration) + '_euclid_optimized'
        if visualize_clustering: v.visualize_clusters(City, Clusters, image_title, hub_colors, save=True)
        if calc_euclid and visualize_clustering: v.euclid_visualize_clusters(City, Clusters, image_title_euclid, hub_colors, save=True) 
        
        ###check fitness function
        time_check, max_time_list = Clusters.hub_fitness(City, max_travel_time)
        iteration += 1
        max_distance -= 3
        max_iterations += 10
    
if __name__ == '__main__':
    main()