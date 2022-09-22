from network_delft import CityNetwork, timer_decorator
from utils.shortest_path import transform_coordinates, find_nearest_edge, multicore_shortest_path, _single_shortest_path
import utils.taxicab_source as tcs
import osmnx as ox
import taxicab as tc
import matplotlib.pyplot as plt

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
    """
    # Initialize network if not already saved as pickle file:
    
    # Initialize CityNetwork object
    Delft = CityNetwork('Delft', [52.03, 51.96, 4.4, 4.3])
    
    # # # Load osm from local or online file
    Delft.load_osm_graph('data/Delft.osm')
    
    # # # Add speeds, lengths and distances to graph
    Delft.add_rel_attributes()

    # # # Project graph
    Delft.project_graph()
    # Delft.plot()

    # # # Calculate dataframes of nodes and edges
    Delft.convert_graph_edges_to_df()
    Delft.convert_graph_nodes_to_df()

    # # Save Pickle file
    Delft.save_graph('Delft')
    """

    """
    Load network from pickle file
    """
    # Load the Delft CityNetwork class object
    Delft = CityNetwork.load_graph('Delft')

    # Specify the coordinates for origin and destination
    orig = (51.99274, 4.35108)
    dest = (52.02184, 4.37890)

    # Extract the graph from the Delft CityNetwork
    graph = Delft.graph
    print(f"Processing {graph}")

    # Transform the start and origin to epsg:3857 (surface)
    orig_yx_transf = transform_coordinates(orig)
    dest_yx_transf = transform_coordinates(dest)

    orig_yx_transf = (6803701, 484081)
    dest_yx_transf = (6794547, 483281)
    
    # Find the nearest edges between the graph of Delft and origin and destination
    print('Finding origin and destination edges...')
    #orig_edge, dest_edge = find_nearest_edge(graph, orig_yx_transf, dest_yx_transf)
    orig_edge, dest_edge = ((1448535927, 1371031575, 0), (6883754839, 6883754848, 0))

    # COMPARISON OF DIFFERENT SHORTEST PATH METHODS
    #path_tc = tcs.shortest_path(graph, orig_yx_transf, dest_yx_transf, orig_edge, dest_edge)
    #path_ssp_dist = _single_shortest_path(graph, orig_yx_transf, dest_yx_transf, orig_edge, dest_edge, weight='length', method = 'dijkstra', return_path=True)
    #path_ssp_time = _single_shortest_path(graph, orig_yx_transf, dest_yx_transf, orig_edge, dest_edge, weight='travel_time', method = 'dijkstra', return_path=True)
    path_msp = multicore_shortest_path(graph, [orig_yx_transf] * 5000, [dest_yx_transf] * 5000, [orig_edge] * 5000, [dest_edge] * 5000, weight='length', method = 'dijkstra', return_path=False, cpus=10)
    # 5000 shortest paths in 108.51s, with path return

    # PLOT RESULTS
    #plot(graph, path_tc, orig_yx_transf, dest_yx_transf, route_color='yellow')
    #plot(graph, path_ssp_dist, orig_yx_transf, dest_yx_transf, route_color='purple')
    # #plot(graph, path_ssp_time, orig_yx_transf, dest_yx_transf, route_color='red')
    #plt.legend()
    #plt.show()
   
    # Print the amount of paths calculated
    print(f"Computed {len(path_msp)} paths in xs")

if __name__ == '__main__':
    main()