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

    coordinates = City.building_addr_df.loc[:, ['latitude', 'longitude']]
    destinations = list(coordinates.itertuples(index=False, name=None))

    dest_yx_transf = transform_coordinates(destinations)

    # CALCULATE NEAREST EDGES IF NOT AVAILABLE IN City.ne
    # y, x = list(map(list, zip(*dest_yx_transf)))

    # City.ne = None
    # dest_edges = City.nearest_edges(x, y, 5, cpus=12)
    # City.save_graph(name, data_folder)
    dest_edges = City.ne
    """--------------------------------------------------"""




    # MULTICORE V2 STARTS HERE:

    # Compute shortest paths by hub for n clustering iterations
    # Hubs are randomly placed each iteration, just as example
    cluster_iterations = []
    num_hubs = 10
    num_iterations = 1

    for i in range(num_iterations):
        # Random positions of hubs
        random.seed(2)
        hubs = [(random.randint(6801030, 6803490), random.randint(484261, 486397)) for _ in range(num_hubs)]

        print(f"Cluster iteration {i} started...")

        # Calculate shortest paths by hub
        # Requires the following inputs: graph, orig(hubs), dest(houses), dest_edges(pre-calculated), method='dijkstra', weight='travel_time', cutoff=None, cpus=None
        paths = multicore_single_source_shortest_path(City.graph, hubs, dest_yx_transf, dest_edges, weight='travel_time', cutoff=120, cpus=1)

        paths_df = paths_to_dataframe(paths, hubs=hubs)
        print(paths_df)

        '''
        CALCULATE FITNESS HERE
        REPOSITION THE HUBS BASED ON K-MEANS CLUSTERING        
        '''

        # Add to cluster_iterations results
        cluster_iterations.append(paths)

        print(f"Cluster iteration {i} finished...")
        print("--------------------------------")
    


    # PLOTTING CAN BE DONE THROUGH:
    # - EASY: The CityNetwork class using CityNetwork.plot(kwargs)
    # - MULTICORE: The multiplot package, to save many images

    colors = ['red', 'orange', 'yellow', 'blue', 'green', 'black', 'gray', 'pink', 'purple', 'peru']



    # PLOTTING ONE CLUSTERING RESULT, SINGLE CORE
    # closest_hubs, assigned_houses = closest_hub(paths)

    # Format the paths and destinations
    # The format_paths_for_plot functions returns:
    # - Cleaned list of paths that should be plotted based on closest hubs
    # - The destinations that should be plotted
    # - A color mask that can be used for the color_mask of the route and destinations
    # - A color mask for the origins
    # Colors are repeated if number of hubs > number of colors.

    # cleaned_paths, destinations, color_mask, orig_color_mask = format_paths_for_plot(paths, hubs, destinations, closest_hubs, assigned_houses, colors)
    # fig, ax = City.plot(routes=cleaned_paths, origins=hubs, destinations=destinations, route_color_mask=color_mask, orig_color_mask=orig_color_mask, dest_color_mask=color_mask, save=True, show=True)
    


    # SAVING MULTIPLE CLUSTERING ITERATION PLOTS, MULTICORE
    # Show is always set to False in this case!

    # WARNING: THIS PROCESS MEMORY BOUND, USES UP TO 40 GB COMMITED MEMORY FOR FULL DELFT NETWORK dpi=300
    start = time.time()
    
    multiplot_save(cluster_iterations, City, dest_yx_transf, closest_hub, colors, session_name, dpi=50, cpus=None)
    
    end = time.time()
    print(end-start)

if __name__ == '__main__':
    main()