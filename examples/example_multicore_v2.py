# Import unpack from parent directory
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

import unpack
from unpack import CityNetwork

import random
import time

@unpack.timer_decorator
def main():
    # LOAD THE NETWORK (SIMILAR AS BEFORE)
    name = 'Delft_center_walk'
    data_folder = 'data/'
    vehicle_type = 'walk' # walk, bike, drive, all (See osmnx documentation)

    # Give a name, to avoid overwriting your plots
    session_name = input("Please insert a name for this multiplot session: ")

    ''' --- GENERATE NETWORK ---
    Generate a new network using the functions in the CityNetwork class. If a network already has been generated and stored in the data folder, comment this part and continue with PREPARE NETWORK. '''
    coordinates = [52.018347, 52.005217, 4.369142, 4.350504]
    
    # Initialize CityNetwork object [N, S, E, W]
    City = CityNetwork(name, coordinates, vehicle_type)
    
    # Load osm from local or online file
    City.load_osm_graph(data_folder + name + '.osm')
    City.load_building_addr(data_folder + name + '_building_addresses.csv', 
        data_folder + name + '_buildings.csv', 
        data_folder + name + '_addresses.csv',
        data_folder +'runtime/'+ name + '_cbs.xml')
    
    print(City.building_addr_df)

    # Add speeds, lengths and distances to graph
    # Overwrite speed by using overwrite_bike=16
    # Further types available: overwrite_walk and overwrite_epv
    City.add_rel_attributes(overwrite_bike=16, overwrite_walk=5)

    # Add an experience attribute to the graph, inputs are
    # edges: list with edges to overwrite
    # factors: list of factors between 0 and float(inf), lower is better

    # Project graph
    City.project_graph()

    ''' EXAMPLE HIGHLIGHTS WITH EXPERIENCE
    Assign bonus or penalty to the Oude Delft 
    > 1 is bonus
    < 1 is penalty

    Function takes as input: name/coordinate_tuple, factor
    City.add_street_experience(['Oude Delft'], [10])
    OR
    City.add_coord_experience([(latitude, longitude)], [10])
    '''

    # Plot the CityNetwork
    # City.plot()

    # Calculate dataframes of nodes and edges
    City.convert_graph_edges_to_df()
    City.convert_graph_nodes_to_df()

    # Save graph edges to file
    #  City.graph_edges_df.to_csv('data/test.csv')

    # Save Pickle file
    City.save_graph(name, data_folder)
    print('------------------------------------') 

    # Load the CityNetwork
    City = CityNetwork.load_graph(name, data_folder)

    # City.plot(show=True)

    # CALCULATE NEAREST EDGES IF NOT AVAILABLE IN City.ne    
    # City.ne = None
    dest_edges = City.nearest_edges(5, cpus=12)
    City.save_graph(name, data_folder)


    # REMOVE OUTLIERS FROM A CERTAIN DISTANCE
    City.drop_outliers(30)
    dest_edges = City.ne

    # Extract the new destinations skipping the outliers
    destinations = City.get_yx_destinations()

    # MULTICORE V2 STARTS HERE:
    # Compute shortest paths by hub for n clustering iterations
    # Hubs are randomly placed each iteration, just as example
    cluster_iterations = []
    num_hubs = 48
    num_iterations = 1

    for i in range(num_iterations):
        # Random seed, to compare results for different parameters
        random.seed(2)

        # Delft Center
        # Should be defined by k++ version of k-means in final algorithm
        # hubs = [(random.randint(6801030, 6803490), random.randint(484261, 486397)) for _ in range(num_hubs)]
        
        # Full Delft
        hubs = [(random.randint(6792760, 6805860), random.randint(478510, 490000)) for _ in range(num_hubs)]

        print(f"Cluster iteration {i} started...")
        start=time.time()

        # Calculate shortest paths by hub
        # Check the code for description of inputs.
        start = time.time()
        paths = unpack.multicore_single_source_shortest_path(City.graph, hubs, destinations, dest_edges,
            skip_non_shortest=False, 
            skip_treshold=60,
            weight='travel_time', 
            cutoff=None, 
            cpus=12
            )
        end = time.time()
        print(end-start)

        colors = ['red', 'orange', 'yellow', 'pink', 'purple', 'peru']

        # Show the results
        paths_df = unpack.paths_to_dataframe(paths, colors, hubs=hubs)
        print(paths_df)

        '''

        CALCULATE FITNESS HERE
        REPOSITION THE HUBS BASED ON K-MEANS CLUSTERING      
        
        '''

        # Add to cluster_iterations results
        cluster_iterations.append(paths)

        end=time.time()

        print(f"Cluster iteration {i} finished in {end-start}s...")
        
if __name__ == '__main__':
    main()