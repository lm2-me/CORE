import matplotlib.pyplot as plt

from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import transform_coordinates

import time

'''
ImportError? Move this file to code folder TEMPORATILY
'''

''' --- CityNetwork EXAMPLE ---
Developed by Job de Vogel

This file is an example of how to use the 
Delft_network script and the shortest_path.

It includes references to:
    - CityNetwork (Job de Vogel)
    - multicore_shortest_path (Nathan A. Rooy, Job de Vogel)
    - multicore_nearest_edges (gboeing, Job de Vogel)
    - load_building_addr (Jirri van den Bos)
'''

@timer_decorator
def main():
    ''' --- INITIALIZE --- '''
    name = 'Rotterdam_bike'
    data_folder = 'data/'
    vehicle_type = 'bike' # walk, bike, drive, all (See osmnx documentation)

    # Initialize CityNetwork object [N, S, E, W]
    # Delft City center: [52.018347, 52.005217, 4.369142, 4.350504]
    # Delft: [52.03, 51.96, 4.4, 4.3]
    # Rotterdam center (control): [51.926366, 51.909002, 4.48460, 4.455496]
    coordinates = [51.926366, 51.909002, 4.48460, 4.455496]


    ''' --- GENERATE NETWORK ---
    Generate a new network using the functions in the CityNetwork class. If a network already has been generated and stored in the data folder, skip this part and continue with PREPARE NETWORK. '''

    # Initialize CityNetwork object [N, S, E, W]
    Delft = CityNetwork(name, coordinates, vehicle_type)
    
    # Load osm from local or online file
    Delft.load_osm_graph(data_folder + name + '.osm')
    Delft.load_building_addr(data_folder + name + '_building_addresses.csv', 
        data_folder + name + '_buildings.csv', 
        data_folder + name + '_addresses.csv')
    
    print(Delft.building_addr_df)

    # Add speeds, lengths and distances to graph
    # Overwrite speed by using overwrite_bike=16
    # Further types available: overwrite_walk and overwrite_epv
    Delft.add_rel_attributes(overwrite_bike=16)

    # Project graph
    Delft.project_graph()
    # Delft.plot()

    # Calculate dataframes of nodes and edges
    Delft.convert_graph_edges_to_df()
    Delft.convert_graph_nodes_to_df()

    # # Save Pickle file
    Delft.save_graph(name, data_folder)
    print('------------------------------------')


    ''' --- PREPARE NETWORK ---
    Load the network from .pkl file. Transform the origin and destination to the unprojected space.'''
    # Load the Delft CityNetwork class object
    Delft = CityNetwork.load_graph(name, data_folder)

    # Specify the coordinates for origin and destination
    # Get the coordinates from building dataframe   
    coordinates = Delft.building_addr_df.loc[:, ['latitude', 'longitude']]

    # Convert the coordinates to tuples, destinations are one goal
    origins = list(coordinates.itertuples(index=False, name=None))
    destinations = [((51.916328, 4.473386))] * len(origins)

    # Extract the graph from the Delft CityNetwork
    graph = Delft.graph
    print(f"Processing {graph}")

    # Transform the start and origin to epsg:3857 (surface)
    orig_yx_transf = transform_coordinates(origins)
    dest_yx_transf = transform_coordinates(destinations)   

    ''' --- MULTICORE NEAREST EDGE COMPUTATION ---
    Find the nearest edges between the graph of Delft and origin and destination. This edge is used to figure out the shortest route in the beginning and end of the path. The origin and destination coordinates are combined so that the time consuming interpolation of the graph only takes place once. Afterwards, the found edges are split again in origin and destination.

    It may be usefull to save the CityNetwork again after computing, using Delft.save_graph('...'), so that you can skip calculating the nearest edges. Remember however that the algorithm does not check if the coordinate inputs are the same as in the saved pickle file. ONLY do this if you are planning to reuse the nearest edges.
    The nearest edges from an earlier saved CityNetwork object can be removed using:
    
    Delft.ne = None
    Delft.save_graph('...')
    '''
    print('------------------------------------')

    # Remove existing nearest_edges in CityNetwork object, proceed carefully:
    # Delft.ne = None

    # Split the origins and destinations in x coordinate and y coordinate
    # Combine the origins and destinations values so they can be computed in one calculation
    x = []
    y = []

    for orig in orig_yx_transf:
        x.append(orig[1])
        y.append(orig[0])
    
    for dest in dest_yx_transf:
        x.append(dest[1])
        y.append(dest[0])

    # Find the nearest edges using multicore processing. If the CityNetwork class
    # already has nearest edges stored, it skips the computation.
    
    print('Finding origin and destination edges...')
    edges = Delft.nearest_edges(x, y, 5, cpus=None)

    # Save graph with edges, proceed carefully:
    Delft.save_graph(name, data_folder)

    # Split the found edges in origins and destinations (in half)
    orig_edge = edges[:len(edges)//2]
    dest_edge = edges[len(edges)//2:]
    

    ''' --- MULTICORE SHORTEST PATH COMPUTATION ---
    Compute the shortest path between origin and destination, taking in account the position of the start and end in relation to the nearest edges. Input of orig, dest, orig_edge and dest_edge will interally be converted to a list, if inputed as tuples. '''    
    print('------------------------------------')
    start = time.time()
    paths = Delft.shortest_paths(orig_yx_transf, dest_yx_transf, orig_edge, dest_edge, weight='travel_time', method='dijkstra', return_path=True, cpus=None)
    end = time.time()
    print(f"Finished solving {len(paths)} paths in {round(end-start)}s.")
    # print(paths)
    

    '''' --- PRINT RESULTS DATAFRAME---
    Although the paths have been found, you may want to see the results in a dataframe.
    Make sure the shortest path algorithm is using input return_path=True'''
    print('------------------------------------')

    # Select a specific path:
    path = paths[0]

    # Get information about the speeds, travel_time, road_names etc.
    edges = []
    for i, node in enumerate(path[1][:-1]):
        edges.append((node, path[1][i+1], 0))

    # Specify which attributes to show
    # Not sure which attributes are available? Use Delft.get_edge_attribute_types()
    df = Delft.graph_edges_df.loc[edges, ['name', 'length', 'speed_kph', 'travel_time']]
    print(df)
    
    ''' --- PLOT RESULTS ---
    Plot one or multiple paths using the CityNetwork plot functions. Advantage of this function is the ability to print multiple routes instead of one, including the linestrings calculated using the taxicab method.
    
    If you need different color and size settings for the plot, change them in the CityNetwork class (on top of class code)'''
    fig, ax = Delft.plot(paths, orig_yx_transf, dest_yx_transf, save=True)
    plt.show()
    
    return

if __name__ == '__main__':
    main()