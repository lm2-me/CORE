import taxicab as tc
import matplotlib.pyplot as plt

from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import transform_coordinates, multicore_shortest_path
from utils.multicore_nearest_edges import multicore_nearest_edge

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
'''

@timer_decorator
def main(): 
    ''' --- GENERATE NETWORK ---
    Generate a new network using the functions in the CityNetwork class. If a network already has been generated and stored in the data folder, skip this part and continue with PREPARE NETWORK. '''

    # # Initialize CityNetwork object
    # Delft = CityNetwork('Delft_bike', [52.03, 51.96, 4.4, 4.3], 'bike')
    
    # # Load osm from local or online file
    # Delft.load_osm_graph('data/Delft_bike.osm')
    
    # # Add speeds, lengths and distances to graph
    # Delft.add_rel_attributes()

    # # Project graph
    # Delft.project_graph()
    # # Delft.plot()

    # # Calculate dataframes of nodes and edges
    # Delft.convert_graph_edges_to_df()
    # Delft.convert_graph_nodes_to_df()

    # '''
    # Before continuing, you will have to decide what speeds
    # you are actually using on your paths. OSM assumes the
    # fastest allowed speed, e.g. 50 km/h on a bike...
    # '''

    # # # Save Pickle file
    # Delft.save_graph('Delft_bike')
    # print('------------------------------------')


    ''' --- PREPARE NETWORK ---
    Load the network from .pkl file. Transform the origin and destination to the unprojected space.'''
    _multiply = 100

    # Load the Delft CityNetwork class object
    Delft = CityNetwork.load_graph('Delft_bike')

    # Specify the coordinates for origin and destination
    orig = (51.99274, 4.35108)
    dest = (52.02184, 4.37890)

    # Extract the graph from the Delft CityNetwork
    graph = Delft.graph
    print(f"Processing {graph}")

    # Transform the start and origin to epsg:3857 (surface)
    orig_yx_transf = transform_coordinates(orig)
    dest_yx_transf = transform_coordinates(dest)

    # Temporary overwrite, remove if you want to use coordinates
    # It is easier to plot a graph and select manual points with 
    # the unprojected coordinates.
    orig_yx_transf = [(6803701, 484081), (6804701, 484281)]
    dest_yx_transf = [(6800000, 480000), (6794000, 488700)]
    

    ''' --- MULTICORE NEAREST EDGE COMPUTATION ---
    Find the nearest edges between the graph of Delft and origin and destination. This edge is used to figure out the shortest route in the beginning and end of the path. The origin and destination coordinates are combined so that the time consuming interpolation of the graph only takes place once. Afterwards, the found edges are split again in origin and destination.

    It may be usefull to save the CityNetwork again after computing, using Delft.save_graph('...'), so that you can skip calculating the nearest edges. Remember however that the algorithm does not check if the coordinate inputs are the same as in the saved pickle file. ONLY do this if you are planning to reuse the nearest edges.
    The nearest edges from an earlier saved CityNetwork object can be removed using:
    
    Delft.ne = None
    Delft.save_graph('...')
    '''
    print('------------------------------------')

    # Remove existing nearest_edges in graph, proceed carefully:
    Delft.ne = None

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
    edges = Delft.nearest_edges(x, y, 10, cpus=1)

    # Save graph with edges, proceed carefully:
    Delft.save_graph('Delft_bike')

    # Split the found edges in origins and destinations (in half)
    orig_edge = edges[:len(edges)//2]
    dest_edge = edges[len(edges)//2:]

    ''' --- MULTICORE SHORTEST PATH COMPUTATION ---
    Compute the shortest path between origin and destination, taking in account the position of the start and end in relation to the nearest edges. Input of orig, dest, orig_edge and dest_edge will interally be converted to a list, if inputed as tuples. '''
    print('------------------------------------')
    paths = Delft.shortest_paths(orig_yx_transf, dest_yx_transf, orig_edge, dest_edge, weight='travel_time', method='dijkstra', return_path=True, cpus=1, _multiply=1)

    '''' --- PRINT RESULTS DATAFRAME---
    Although the paths have been found, you may want to see the results in a dataframe.'''
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

if __name__ == '__main__':
    main()