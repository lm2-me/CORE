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
import unpack
import multiprocessing as mp

import unpack.utils.network_helpers as h
import unpack.utils.visualizations as v
import matplotlib.pyplot as plt
from re import X
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
    max_cpu_count = 12 #input the maximum CPU cores, use None to automatically set to maximum available
    network_name = 'delft_center_walk'
    cluster_name = 'delft_cluster'
    data_folder = 'data/'
    vehicle_type = 'walk' # walk, bike, drive, all (See osmnx documentation)
    max_travel_time = 150 #time in seconds
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
    #coordinates = [52.018347, 52.005217, 4.369142, 4.350504]
    coordinates = [52.03, 51.96, 4.4, 4.3]

    start_pt_ct = 3 # number of starting hubs
    calc_euclid = False #also calculate euclidean distances for first iteration?
    generate_new_network = False # if true, new network will be generated, otherwise set to false to only load a network
    continue_clustering = False # if clustering crashed or finished, set to true to load the last iteration
    visualize_clustering = True # if clusering images are generated

    #enter color pallet to use for plotting
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
    destinations = City.get_yx_destinations()
    dest_edges = City.nearest_edges(5, cpus=12)
    City.save_graph(network_name, data_folder)
    dest_edges = City.ne

    ###initiate clustering
    Clusters = NetworkClustering('delftclustering')
    Clusters.reset_hub_df(City)

    ### if clustering crashed or finishes, load the latest iteration
    if continue_clustering:
        path = input("Please insert the file name that you would like to load to continue clustering: ")
        Clusters.continue_clustering(path)
    print('Starting with iteration ', Clusters.iteration)

    #initialized variables - do not change
    iteration = Clusters.iteration
    time_check = False
    max_time_list = m.inf 

    #option to change/adjust these as clusering settings
    point_count = 1 #the number of new hubs to add when the fitness is not reached
    max_distance = 50 #when distance the hub moves during k-means clustering, is less than max_distance, the hub will stop moving
    max_iterations = 30 #maximum iterations in case the max_distance moves a lot for too long

    while not time_check and iteration < 10:
        ### reset all DF values to default for iteration and update class properties
        print('iteration: ', iteration)
        Clusters.reset_hub_df(City)
        Clusters.iteration = iteration
        if max_cpu_count is None: cpus = mp.cpu_count()
        Clusters.max_cores = max_cpu_count
        Clusters.save_iteration(cluster_name, data_folder, session_name, Clusters.iteration, '01_initialize')
        print('saved iteration ' + str(Clusters.iteration) + ' initialize 01')

        if iteration == 1:           
            # update number of CPUs to use based on number of clusters
            if Clusters.max_cores > start_pt_ct: cpu_count = start_pt_ct
            else: cpu_count = Clusters.max_cores

            ### only on first iteration generate random points for hubs and cluster houses based on closest hub
            hubs = Clusters.generate_random_points(coordinates_transformed_xy, start_pt_ct)
            Clusters.cluster_number = start_pt_ct
            Clusters.save_iteration(cluster_name, data_folder, session_name, Clusters.iteration, '02_locations')
            print('saved iteration ' + str(Clusters.iteration) + ' locations 02')
            print(Clusters.hub_list_dictionary) 
            paths = unpack.multicore_single_source_shortest_path(City.graph, Clusters.hub_list_dictionary, destinations, dest_edges,
                skip_non_shortest=skip_non_shortest_input, 
                skip_treshold=skip_treshold_input,
                weight=weight_input, 
                cutoff=cutoff_input, 
                cpus=cpu_count
                )
            Clusters.hub_assignments_df = unpack.paths_to_dataframe(paths, hubs=hubs)

            if calc_euclid:
                Clusters.hub_clusters_euclidean(orig_yx_transf, cluster_name, data_folder)
                Clusters.save_iteration(cluster_name, data_folder, session_name, Clusters.iteration, '03_cluster_euclid')
                print('saved iteration ' + str(Clusters.iteration) + ' cluster euclid 03')
            Clusters.save_iteration(cluster_name, data_folder, session_name, Clusters.iteration, '03_cluster')
            print('saved iteration ' + str(Clusters.iteration) + ' cluster 03')
            Clusters.max_cores = cpu_count
            
        else:           
            ### on all other iterations, add a new hub each time the while loop runs 
            Clusters.add_points(point_count, coordinates_transformed_xy)
            Clusters.cluster_number += point_count
            print('added hub', Clusters.hub_list_dictionary)
            Clusters.save_iteration(cluster_name, data_folder, session_name, Clusters.iteration, '02_locations')
            print('saved iteration ' + str(Clusters.iteration) + ' locations 02')

            # update number of CPUs to use based on number of clusters
            if Clusters.max_cores > Clusters.cluster_number: cpu_count = Clusters.cluster_number
            else: cpu_count = Clusters.max_cores

            paths = unpack.multicore_single_source_shortest_path(City.graph, Clusters.hub_list_dictionary, destinations, dest_edges,
                skip_non_shortest=skip_non_shortest_input, 
                skip_treshold=skip_treshold_input,
                weight=weight_input, 
                cutoff=cutoff_input, 
                cpus=cpu_count
                )
            Clusters.hub_assignments_df = unpack.paths_to_dataframe(paths, hubs=hubs)

            if calc_euclid:
                Clusters.hub_clusters_euclidean(orig_yx_transf, cluster_name, data_folder)
                Clusters.save_iteration(cluster_name, data_folder, session_name, Clusters.iteration, '03_clusters_euclid')
                print('saved iteration ' + str(Clusters.iteration) + ' cluster euclid 03')
            Clusters.save_iteration(cluster_name, data_folder, session_name, Clusters.iteration, '03_clusters')
            print('saved iteration ' + str(Clusters.iteration) + ' cluster 03')
            Clusters.max_cores = cpu_count

        ###optimize hub locations
        move_distance = Clusters.new_hub_location(City)
        i = 0
        while move_distance > max_distance and i < max_iterations:
            paths = unpack.multicore_single_source_shortest_path(City.graph, Clusters.hub_list_dictionary, destinations, dest_edges,
                skip_non_shortest=skip_non_shortest_input, 
                skip_treshold=skip_treshold_input,
                weight=weight_input, 
                cutoff=cutoff_input, 
                cpus=cpu_count
                )
            Clusters.hub_assignments_df = unpack.paths_to_dataframe(paths, hubs=hubs)
            if calc_euclid: Clusters.hub_clusters_euclidean(orig_yx_transf, cluster_name, data_folder)
            move_distance = Clusters.new_hub_location(City)
            print('moved hubs on average ', move_distance)

            i += 1

        Clusters.save_iteration(cluster_name, data_folder, session_name, Clusters.iteration, '04_clusters')
        print('saved iteration ' + str(Clusters.iteration) + ' cluster 04')
        
        ###check fitness function
        time_check, max_time_list = Clusters.hub_fitness(City, max_travel_time)
        iteration += 1
        max_distance -= 3
        max_iterations += 10

    # ! @LM something goes wrong here, I get a TypeError
    file_path_to_read = data_folder + session_name + '/'
    cluster_iterations, destinations, closest_hub = Clusters.load_files_for_plot(file_path_to_read)
    #unpack.multiplot_save(cluster_iterations, City, destinations, closest_hub, hub_colors, session_name, dpi=300, cpus=None)
    ##!at end call multiplot with 1 core and add variable as option to show the image
    
if __name__ == '__main__':
    main()