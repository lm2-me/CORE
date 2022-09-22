import osmnx as ox
import taxicab as tc
import matplotlib.pyplot as plt

from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import transform_coordinates, find_nearest_edge, multicore_shortest_path, _single_shortest_path
from utils.multicore_nearest_edges import multicore_nearest_edge
import utils.taxicab_source as tcs

'''
ImportError? Move this file to code folder TEMPORATILY
'''

'''
This file is an example of how to use the 
Delft_network script and the shortest_path.
'''

def plot(graph, route, orig, dest, route_color='grey'):
    fig, ax = tc.plot.plot_graph_route(
    graph, route, figsize=(10,10), show=False, close=False, route_color=route_color)

    ax.scatter(orig[1], orig[0],
    color='lime', marker='x', s=100, label='orig-point')

    ax.scatter(dest[1], dest[0],
        color='red', marker='x', s=100, label='dest-point')

@timer_decorator
def main():   
    # Initialize CityNetwork object
    Delft = CityNetwork('Delft_bike', [52.03, 51.96, 4.4, 4.3], 'walk')
    
    # # # Load osm from local or online file
    Delft.load_osm_graph('data/Delft_walk.osm')
    
    # # # Add speeds, lengths and distances to graph
    Delft.add_rel_attributes()

    # # # Project graph
    Delft.project_graph()
    # Delft.plot()

    # # # Calculate dataframes of nodes and edges
    Delft.convert_graph_edges_to_df()
    Delft.convert_graph_nodes_to_df()

    '''
    Before continuing, you will have to decide what speeds
    you are actually using on your paths. OSM assumes the
    fastest allowed speed, e.g. 50 km/h on a bike...
    '''

    # # Save Pickle file
    Delft.save_graph('Delft_walk')
    print('------------------------------------')

    """
    Load network from pickle file
    """
    # Load the Delft CityNetwork class object
    #Delft = CityNetwork.load_graph('Delft')

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
    orig_yx_transf = (6803701, 484081)
    dest_yx_transf = (6794547, 483281)
    
    # Find the nearest edges between the graph of Delft and origin and destination
    # Recommended to combine all orig_yx_transf and dest_yx_transf into one list
    # and then only run multicore_nearest_edge once.
    # The script is currently optimized for a great amount of orig_edges and dest_edges
    # that need to be calculated.
    print('Finding origin and destination edges...')
    orig_edge = multicore_nearest_edge(Delft.graph, orig_yx_transf[1], orig_yx_transf[0], 10, cpus=1)[0]
    dest_edge = multicore_nearest_edge(Delft.graph, dest_yx_transf[1], dest_yx_transf[0], 10, cpus=1)[0]
    
    # EXAMPLE
    path = multicore_shortest_path(graph, [orig_yx_transf], [dest_yx_transf], [orig_edge], [dest_edge], weight='travel_time', method = 'dijkstra', return_path=True, cpus=1)
    print(path)

    # Print the route from the Delft dataframe
    # Get information about the speeds, travel_time, road_names etc.
    edges = []
    for i, node in enumerate(path[0][1][:-1]):
        edges.append((node, path[0][1][i+1], 0))

    print(Delft.graph_edges_df.loc[edges, ['name', 'length', 'speed_kph']])
    #print(Delft.graph_edges_df.loc[edges, :])
    
    # PLOT RESULTS
    plot(graph, path[0], orig_yx_transf, dest_yx_transf, route_color='red')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()