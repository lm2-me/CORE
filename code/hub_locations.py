from cgi import test
from pickletools import read_unicodestring1
import matplotlib.pyplot as plt
import numpy as np
import math as m
import osmnx as ox

from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import transform_coordinates
from utils.multicore_nearest_edges import multicore_nearest_edge

import random
import time


### Job's Code: generate network
''' --- GENERATE NETWORK ---
Generate a new network using the functions in the CityNetwork class. If a network already has been generated and stored in the data folder, comment this part and continue with PREPARE NETWORK. '''

def generate_network(name, data_folder, vehicle_type, coordinates):
    ''' --- INITIALIZE --- '''

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

###Job's Code load network
''' --- PREPARE NETWORK ---
Load the network from .pkl file. Transform the origin and destination to the unprojected space.'''
def load_network(name, data_folder):

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

    for index, row in City.building_addr_df.T.items():
        coordinate = row['x'], row['y']
        orig_yx_transf.append(coordinate)

    return City, orig_yx_transf

def nearest_edges_buildings(City, orig_yx_transf, name, data_folder):
    x = []
    y = []

    for orig in orig_yx_transf:
        x.append(orig[1])
        y.append(orig[0])
    
    edges = City.nearest_edges(x, y, 5, cpus=None)
    City.save_graph(name, data_folder)
    orig_edge = edges
    return orig_edge

def nearest_edges_hubs(City, hub_yx_transf):
    x = []
    y = []

    for hub in hub_yx_transf:
        x.append(hub[1])
        y.append(hub[0])
    
    edges, _ = multicore_nearest_edge(City.graph, x, y, City.interpolation, cpus=1)
    #print(f"edge: {edges}")
    return edges

def coordinates_to_tupple(coordinates):
    #[N, S, E, W]
    w_n_corner = (coordinates[0], coordinates[3]) 
    e_n_corner = (coordinates[0], coordinates[2]) 
    w_s_corner = (coordinates[1], coordinates[3]) 
    e_s_corner = (coordinates[1], coordinates[2]) 

    return ([w_n_corner, e_n_corner, w_s_corner, e_s_corner])


def generate_random_points(coordinates_transformed,start_pt_ct,hub_dictionary=None):
    hub_yx_transf = []
    
    if hub_dictionary == None:
        hub_dictionary = {
            "hub_id": {
                "x": 0,
                "y": 0
            }
        }

    #[N, S, E, W]
    coordinateX_min = coordinates_transformed[1][1]
    coordinateX_max = coordinates_transformed[0][1]
    coordinateY_min = coordinates_transformed[0][0]
    coordinateY_max = coordinates_transformed[2][0]

    index = len(hub_dictionary)

    for i in range(start_pt_ct):
        x = random.uniform(coordinateX_min, coordinateX_max)
        y = random.uniform(coordinateY_min, coordinateY_max)

        if index not in hub_dictionary:
            hub_dictionary[index] = {
                "x": x,
                "y": y
            } 
        index += 1

    return hub_dictionary

def get_yx_transf_from_dict(hub_dictionary):
    hub_yx_transf = []
    for index in range(1, len(hub_dictionary)):
        coordinate = (hub_dictionary[index]['y'], hub_dictionary[index]['x'])
        hub_yx_transf.append(coordinate)

    return hub_yx_transf

def hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges):
    ### randomly generated hub locations is hub_dictionary
    ### use k-means to cluster houses around hub locations
    hub_num = len(hub_dictionary)
    hub_yx_transf = get_yx_transf_from_dict(hub_dictionary)

    if 'nearesthub' not in City.building_addr_df:
        hub_num = [(0)] * len(City.building_addr_df)
        City.building_addr_df['nearesthub'] = hub_num
    
    if 'hubdistance' not in City.building_addr_df:
        hub_dist = [(m.inf)] * len(City.building_addr_df)
        City.building_addr_df['hubdistance'] = hub_dist
    
    if 'hubx' not in City.building_addr_df:
        hub_dist = [(0)] * len(City.building_addr_df)
        City.building_addr_df['hubx'] = hub_dist
    
    if 'huby' not in City.building_addr_df:
        hub_dist = [(0)] * len(City.building_addr_df)
        City.building_addr_df['huby'] = hub_dist

    dest_edge = nearest_edges_hubs(City, hub_yx_transf)
    dest_edges = dest_edge * len(orig_yx_transf)

    ### first_paths returns [(route_weight, nx_route, orig_partial_edge, dest_partial_edge)]
    for hub_index in range(1, len(hub_dictionary)):
        
        #get distance of all current hub to all houses
        hub_yx_transf_current = [hub_yx_transf[hub_index-1]] * len(orig_yx_transf)
        dest_edges_current = [dest_edges[hub_index-1]] * len(orig_yx_transf)
        
        # Returning route_weight, nx_route, orig_partial_edge, dest_partial_edge
        # [(route_weight, nx_route, orig_partial_edge, dest_partial_edge)]      
        hub_dist = City.shortest_paths(orig_yx_transf, hub_yx_transf_current, orig_edges, dest_edges_current, weight='travel_time', method='dijkstra', return_path=True, cpus=None)
        
        for i, row in enumerate(hub_dist):
            if row[0] < City.building_addr_df.at[i,'hubdistance']:
                City.building_addr_df.at[i,'hubdistance']=row[0]
                City.building_addr_df.at[i,'nearesthub']=hub_index
                City.building_addr_df.at[i,'hubx']=hub_dictionary[hub_index]['x']
                City.building_addr_df.at[i,'huby']=hub_dictionary[hub_index]['y']
        
        print(City.building_addr_df)

    ### 2.5 min is 150 seconds 
    return 1

def hub_fitness():
    return 1

def add_points():
    return 1

def visualize_clusters(City, hub_dictionary):

    print('Plotting figure...')

    fig, ax = ox.plot_graph(City.graph, show=False, save=False, close=False,
        figsize = City.figsize,
        bgcolor = City.bgcolor,
        edge_color = City.edge_color,
        node_color = City.node_color,
        edge_linewidth = City.edge_linewidth,
        node_size = City.node_size)
            
    for hub in hub_dictionary:
        ax.scatter(orig[1], orig[0],
            color=City.origin_color, marker='x', s=100, label='orig-point')
        
        ax.scatter(dest[1], dest[0],
            color=self.destination_color, marker='x', s=100, label='orig-point')

        # Add an x to specific points
        for mark in marks:
            ax.scatter(mark[1], mark[0],
                color=self.marker_color, marker='x', s=100, label='Mark')

        # Add annotations to the edges, can be names, travel_times etc.
        if annotations:
            graph = ox.get_undirected(self.graph)
            for _, edge in ox.graph_to_gdfs(graph, nodes=False).fillna('').iterrows():
                c = edge['geometry'].centroid
                text = edge[annotations]
                ax.annotate(text, (c.x, c.y), c=self.font_color, fontsize=self.font_size)

        fig, ax = ox.plot._save_and_show(fig, ax)

        if save:
            fig.savefig(f'data/plot_pngs/plot_{time.time()}.png')

        return fig, ax
    else:
        # Only print the map
        fig, ax = ox.plot_graph(self.graph, show=False, save=False, close=False,
        figsize = self.figsize,
        bgcolor = self.bgcolor,
        edge_color = self.edge_color,
        node_color = self.node_color,
        edge_linewidth = self.edge_linewidth,
        node_size = self.node_size)
        
        # Add annotations to the edges, can be names, travel_times etc.
        if annotations:
            graph = ox.get_undirected(self.graph)
            for _, edge in ox.graph_to_gdfs(graph, nodes=False).fillna('').iterrows():
                c = edge['geometry'].centroid
                text = edge[annotations]
                ax.annotate(text, (c.x, c.y), c=self.font_color, fontsize=self.font_size)

        # Add an x to specific points
        for mark in marks:
            ax.scatter(mark[1], mark[0],
                color=self.marker_color, marker='x', s=100, label='Mark')

        ox.plot._save_and_show(fig, ax)

    # Loading the graph as a pickle file avoids having to recalculate
    # attributes such as speed, length etc. and is way faster
    # Call this function through var = CityNetwork.load_graph('var')

    return 1
    #City.plot()


def main():
    ### what is the right info to import for Delft
    name = 'delft_walk'
    data_folder = 'data/'
    #vehicle_type = 'bike' # walk, bike, drive, all (See osmnx documentation)
    vehicle_type = 'walk'
    ### Delft
    #coordinates = [52.03, 51.96, 4.4, 4.3]
    ### Delft City Center
    coordinates = [52.018347, 52.005217, 4.369142, 4.350504]
    coordinates_list = coordinates_to_tupple(coordinates)
    coordinates_transformed = transform_coordinates(coordinates_list)

    start_pt_ct = 3

    # ### generate new network, only run at beginning
    # generate_network(name, data_folder, vehicle_type, coordinates)

    ### load network from file, run after network is generated
    City, orig_yx_transf = load_network(name, data_folder)
    #City.ne = None

    orig_edges = nearest_edges_buildings(City, orig_yx_transf, name, data_folder)

    ### generate random points for hubs and cluster houses based on closest hub
    hub_dictionary = generate_random_points(coordinates_transformed, start_pt_ct)       
    hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges)

    #visualize_clusters(City)

    '''
        (done) 1. Initilize clusters - generate_random_starting_points
        (done), job 2. Find distances - city.shortest_paths
        (done) 3. Find dest with shortest weight for each orig
            - these are the "clusters"
        4. Look up the geometric location of each orig
        5. Compute new "centroids"
        6. Find new distances (... repeat 2-5)
        7. Stop when nothing moves (or moves less than X amount)
        8. If any distance > 2.5min. Add more clusters and start from 1.
    '''

if __name__ == '__main__':
    main()