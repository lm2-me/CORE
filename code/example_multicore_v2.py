from utils.multicore_shortest_path import paths_to_dataframe
from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import multicore_single_source_shortest_path, closest_hub
from utils.multiplot import multiplot_save

import random
import time

@timer_decorator
def main():
    # LOAD THE NETWORK (SIMILAR AS BEFORE)
    name = 'Delft_center_walk'
    data_folder = 'data/'
    vehicle_type = 'walk' # walk, bike, drive, all (See osmnx documentation)

    # Give a name, to avoid overwriting your plots
    session_name = input("Please insert a name for this multiplot session: ")

    # Load the CityNetwork
    City = CityNetwork.load_graph(name, data_folder)


    # CALCULATE NEAREST EDGES IF NOT AVAILABLE IN City.ne    
    # City.ne = None
    # dest_edges = City.nearest_edges(5, cpus=12)
    # City.save_graph(name, data_folder)


    # REMOVE OUTLIERS FROM A CERTAIN DISTANCE
    City.drop_outliers(30)
    dest_edges = City.ne

    # Extract the new destinations skipping the outliers
    destinations = City.get_xy_destinations()


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
        hubs = [(random.randint(6801030, 6803490), random.randint(484261, 486397)) for _ in range(num_hubs)]
        
        # Full Delft
        # hubs = [(random.randint(6792760, 6805860), random.randint(478510, 490000)) for _ in range(num_hubs)]

        print(f"Cluster iteration {i} started...")
        start=time.time()

        # Calculate shortest paths by hub
        # Check the code for description of inputs.
        paths = multicore_single_source_shortest_path(City.graph, hubs, destinations, dest_edges, 
            skip_non_shortest=False, 
            weight='travel_time', 
            cutoff=None, 
            cpus=12
            )

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

    # SAVING MULTIPLE CLUSTERING ITERATION PLOTS, MULTICORE
    # WARNING: THIS PROCESS IS MEMORY BOUND, USES UP TO 40 GB COMMITED MEMORY FOR FULL DELFT NETWORK dpi=300, num_hubs=48
    # This solution solves the waiting time for generating all the pngs.
    # It can also be done after the algorithm has finished, and be executed
    # on multiple computers. For just checking the result, it is recommended
    # to only use City.plot(**kwargs) once for the final hub setup.
    start = time.time()
    
    colors = ['red', 'orange', 'yellow', 'pink', 'purple', 'peru']
    
    # To do: remove closest hub function from the multiplot_save inputs
    multiplot_save(cluster_iterations, City, destinations, closest_hub, colors, session_name, dpi=300, cpus=None)
    
    end = time.time()
    print(f"Finished multiplot in {end-start}s")

if __name__ == '__main__':
    main()