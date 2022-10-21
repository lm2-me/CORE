from utils.multicore_shortest_path import paths_to_dataframe
from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import transform_coordinates
from utils.multicore_shortest_path import multicore_single_source_shortest_path, closest_hub
from utils.multiplot import multiplot_save, format_paths_for_plot

import random

import time

@timer_decorator
def main():
    # LOAD THE NETWORK (SIMILAR AS BEFORE)
    name = 'Delft_center_walk'
    data_folder = 'data/'
    vehicle_type = 'walk' # walk, bike, drive, all (See osmnx documentation)

    session_name = input("Please insert a name for this multiplot session: ")

    City = CityNetwork.load_graph(name, data_folder)

    # CALCULATE NEAREST EDGES IF NOT AVAILABLE IN City.ne    
    # City.ne = None
    # dest_edges = City.nearest_edges(x, y, 5, cpus=12)
    # City.save_graph(name, data_folder)

    # REMOVE OUTLIERS FROM A CERTAIN DISTANCE
    City.drop_outliers(30)
    dest_edges = City.ne

    # Extract the new destinations skipping the outliers
    destinations = list(City.building_addr_df.loc[:, ['y', 'x']].itertuples(index=False, name=None))
    """--------------------------------------------------"""



    # MULTICORE V2 STARTS HERE:

    # Compute shortest paths by hub for n clustering iterations
    # Hubs are randomly placed each iteration, just as example
    cluster_iterations = []
    num_hubs = 50
    num_iterations = 1

    for i in range(num_iterations):
        # Random positions of hubs
        random.seed(2)

        # Delft Center
        hubs = [(random.randint(6801030, 6803490), random.randint(484261, 486397)) for _ in range(num_hubs)]
        
        # Full Delft
        # hubs = [(random.randint(6792760, 6805860), random.randint(478510, 490000)) for _ in range(num_hubs)]

        print(f"Cluster iteration {i} started...")
        start=time.time()

        # Calculate shortest paths by hub
        # Check the code for description of inputs.
        paths = multicore_single_source_shortest_path(City.graph, hubs, destinations, dest_edges, skip_non_shortest=False, weight='travel_time', cutoff=600, cpus=12)

        # Show the results
        paths_df = paths_to_dataframe(paths, hubs=hubs)
        print(paths_df)

        '''
        CALCULATE FITNESS HERE
        REPOSITION THE HUBS BASED ON K-MEANS CLUSTERING        
        '''

        # Add to cluster_iterations results
        cluster_iterations.append(paths)

        end=time.time()

        print(f"Cluster iteration {i} finished in {end-start}s...")
        print("--------------------------------")

    # PLOTTING CAN BE DONE THROUGH:
    # - EASY: The CityNetwork class using CityNetwork.plot(kwargs)
    # - MULTICORE: The multiplot package, to save many images

    # SAVING MULTIPLE CLUSTERING ITERATION PLOTS, MULTICORE
    # Show is always set to False in this case!

    # WARNING: THIS PROCESS MEMORY BOUND, USES UP TO 40 GB COMMITED MEMORY FOR FULL DELFT NETWORK dpi=300
    start = time.time()
    
    colors = ['red', 'orange', 'yellow', 'pink', 'purple', 'peru']  
    multiplot_save(cluster_iterations, City, destinations, closest_hub, colors, session_name, dpi=300, cpus=None)
    
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    main()