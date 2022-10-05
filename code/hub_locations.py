from cgi import test
from pickletools import read_unicodestring1
from re import X
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

def reset_hub_df_values(City):
    if 'nearesthub' not in City.building_addr_df:
        hub_num = [('')] * len(City.building_addr_df)
        City.building_addr_df['nearesthub'] = hub_num
    else:
        City.building_addr_df['nearesthub'].values[:] = ''
    
    if 'hubdistance' not in City.building_addr_df:
        hub_dist = [(m.inf)] * len(City.building_addr_df)
        City.building_addr_df['hubdistance'] = hub_dist
    else:
        City.building_addr_df['hubdistance'].values[:] = np.inf
    
    if 'hub_x' not in City.building_addr_df:
        hub_x = [(0)] * len(City.building_addr_df)
        City.building_addr_df['hub_x'] = hub_x
    else:
        City.building_addr_df['hub_x'].values[:] = 0
    
    if 'hub_y' not in City.building_addr_df:
        hub_y = [(0)] * len(City.building_addr_df)
        City.building_addr_df['hub_y'] = hub_y
    else:
        City.building_addr_df['hub_y'].values[:] = 0
    
    if 'euclid_nearesthub' not in City.building_addr_df:
        hub_num = [('')] * len(City.building_addr_df)
        City.building_addr_df['euclid_nearesthub'] = hub_num
    else:
        City.building_addr_df['euclid_nearesthub'].values[:] = ''
    
    if 'euclid_hubdistance' not in City.building_addr_df:
        euclid_hub_dist = [(m.inf)] * len(City.building_addr_df)
        City.building_addr_df['euclid_hubdistance'] = euclid_hub_dist
    else:
        City.building_addr_df['euclid_hubdistance'].values[:] = np.inf

def nearest_edges_buildings(City, orig_yx_transf, name, data_folder, cpu_count):
    x = []
    y = []

    for orig in orig_yx_transf:
        x.append(orig[1])
        y.append(orig[0])
    
    edges = City.nearest_edges(x, y, 5, cpus=cpu_count)
    City.save_graph(name, data_folder)
    orig_edge = edges
    return orig_edge

def nearest_edges_hubs(City, hub_yx_transf, cpu_count):
    x = []
    y = []

    for hub in hub_yx_transf:
        x.append(hub[1])
        y.append(hub[0])
    
    edges, _ = multicore_nearest_edge(City.graph, x, y, City.interpolation, cpus=cpu_count)
    #print(f"edge: {edges}")
    return edges

def coordinates_to_tupple(coordinates):
    #[N, S, E, W]
    w_n_corner = (coordinates[3], coordinates[0]) 
    e_n_corner = (coordinates[2], coordinates[0]) 
    w_s_corner = (coordinates[3], coordinates[1]) 
    e_s_corner = (coordinates[2], coordinates[1]) 

    return ([w_n_corner, e_n_corner, w_s_corner, e_s_corner])

def flip_lat_long(coordinates_list):
    return [(coord[1], coord[0]) for coord in coordinates_list]

def generate_random_points(coordinates_transformed_xy,start_pt_ct, random_init, hub_dictionary=None):
    random.seed(random_init)

    if hub_dictionary == None:
        hub_dictionary = {}

    #[N, S, E, W]
    print('coordinates transformed', coordinates_transformed_xy)
    #w_n_corner, e_n_corner, w_s_corner, e_s_corner
    coordinateX_min = coordinates_transformed_xy[0][0]
    coordinateX_max = coordinates_transformed_xy[1][0]
    coordinateY_min = coordinates_transformed_xy[2][1]
    coordinateY_max = coordinates_transformed_xy[0][1]

    index = len(hub_dictionary)+1

    for i in range(start_pt_ct):
        x = random.uniform(coordinateX_min, coordinateX_max)
        y = random.uniform(coordinateY_min, coordinateY_max)

        index_name = 'hub ' + str(index)

        if index_name not in hub_dictionary:
            hub_dictionary[index_name] = {
                "x": x,
                "y": y
            } 
        index += 1

    return hub_dictionary

def get_xy_transf_from_dict(hub_dictionary):
    label_list = []
    value_list = []
    for (hub_name, hub_loc) in hub_dictionary.items():
        label_list.append(hub_name)
        value_list.append(
            (hub_loc['x'], hub_loc['y'])
        )

    return label_list, value_list
    # hub_yx_transf = []
    # for index in range(1, len(hub_dictionary)+1):
    #     coordinate = (hub_dictionary[index]['y'], hub_dictionary[index]['x'])
    #     hub_yx_transf.append(coordinate)

    # return hub_yx_transf

def hub_clusters_euclidean(City, hub_dictionary, orig_xy_transf, orig_edges,name, data_folder, cpu_count):
    ### randomly generated hub locations is hub_dictionary
    ### use k-means to cluster houses around hub locations
    hub_num = len(hub_dictionary)
    hub_names, hub_xy_transf = get_xy_transf_from_dict(hub_dictionary)
    dest_edges = nearest_edges_hubs(City, hub_xy_transf, cpu_count)
    
    hub_xy_dict = dict(zip(hub_names, hub_xy_transf))
    hub_edge_dict = dict(zip(hub_names, dest_edges))

    ### first_paths returns [(route_weight, nx_route, orig_partial_edge, dest_partial_edge)]
    for hub in hub_names:
        print(hub)
        #get distance of all current hub to all houses
        hub_xy_transf_current = [hub_xy_dict[hub]] * len(orig_xy_transf)
        dest_edges_current = [hub_edge_dict[hub]] * len(orig_xy_transf)
        
        # Returning route_weight, nx_route, orig_partial_edge, dest_partial_edge, orig_yx, dest_yx
        # [(route_weight, nx_route, orig_partial_edge, dest_partial_edge, orig_yx, dest_yx)]      
        
        point2 = np.array(hub_xy_dict[hub])
        ## print(point2)

        #print('hub distance ', hub_dist[0])
        #print('location x ', hub_dist[0][4][0])

        for i, house in enumerate(orig_xy_transf):
            point1 = np.array(house)
            euclid_dist = ((point2[1] - point1[0]) ** 2 + (point2[0] - point1[1]) ** 2) ** 0.5
          
            if euclid_dist < City.building_addr_df.loc[i,'euclid_hubdistance']:
                City.building_addr_df.at[i,'euclid_hubdistance']=euclid_dist
                City.building_addr_df.at[i,'euclid_nearesthub']=hub

        #print(City.building_addr_df.at[0,'x'],City.building_addr_df.at[0,'y'])
    
    City.save_graph(name, data_folder)
    print(City.building_addr_df)
    ### 2.5 min is 150 seconds 
    return 0

def hub_clusters(City, hub_dictionary, orig_xy_transf, orig_edges,name, data_folder, cpu_count):
    ### randomly generated hub locations is hub_dictionary
    ### use k-means to cluster houses around hub locations
    hub_num = len(hub_dictionary)
    hub_names, hub_xy_transf = get_xy_transf_from_dict(hub_dictionary)
    dest_edges = nearest_edges_hubs(City, hub_xy_transf, cpu_count)
    
    hub_xy_dict = dict(zip(hub_names, hub_xy_transf))
    hub_edge_dict = dict(zip(hub_names, dest_edges))

    ### first_paths returns [(route_weight, nx_route, orig_partial_edge, dest_partial_edge)]
    for hub in hub_names:
        print(hub)
        #get distance of all current hub to all houses
        hub_xy_transf_current = [hub_xy_dict[hub]] * len(orig_xy_transf)
        dest_edges_current = [hub_edge_dict[hub]] * len(orig_xy_transf)
        
        # Returning route_weight, nx_route, orig_partial_edge, dest_partial_edge, orig_yx, dest_yx
        # [(route_weight, nx_route, orig_partial_edge, dest_partial_edge, orig_yx, dest_yx)]    
  
        #@Job orig_xy_transf coming from load_network was labeled as orig_yx_tranf but it returns xy (in that order)
        #shortes_paths works when both orig and hub are entered as xy coordinates (x = longitude, y = latitude like you would do when plotting a point on a graph)
        hub_dist = City.shortest_paths(orig_xy_transf, hub_xy_transf_current, orig_edges, dest_edges_current, weight='travel_time', method='dijkstra', return_path=True, cpus=cpu_count)

        ## point2 = np.array(hub_xy_dict[hub])
        ## print(point2)

        #print('hub distance ', hub_dist[0])
        #print('location x ', hub_dist[0][4][0])
        for i, row in enumerate(hub_dist):
        # #for i, house in enumerate(orig_xy_transf):
        #     point1 = np.array(house)
        #     euclid_dist = ((point2[1] - point1[0]) ** 2 + (point2[0] - point1[1]) ** 2) ** 0.5

            dist = row[0]

            # location_x = row[4][0]
            # location_y = row[4][1]

            # house_index = City.building_addr_df.index[(City.building_addr_df['x'] == location_x) & (City.building_addr_df['y'] == location_y)].tolist()
            if dist < City.building_addr_df.loc[i,'hubdistance']:
                City.building_addr_df.at[i,'hubdistance']=dist
                City.building_addr_df.at[i,'nearesthub']=hub
                City.building_addr_df.at[i,'hub_x']=hub_dictionary[hub]['x']
                City.building_addr_df.at[i,'hub_y']=hub_dictionary[hub]['y']
            
            # #if euclid_dist < City.building_addr_df.loc[i,'hubdistance']:
            #     City.building_addr_df.at[i,'hubdistance']=euclid_dist
            #     City.building_addr_df.at[i,'nearesthub']=hub
            #     City.building_addr_df.at[i,'hub_x']=hub_dictionary[hub]['x']
            #     City.building_addr_df.at[i,'hub_y']=hub_dictionary[hub]['y']

        #print(City.building_addr_df.at[0,'x'],City.building_addr_df.at[0,'y'])
    
    City.save_graph(name, data_folder)
    print(City.building_addr_df)
    ### 2.5 min is 150 seconds 
    return 0

def hub_fitness(City, hub_dictionary):
    return 1

def add_points():
    return 1

def visualize_clusters(City, hub_dictionary, save=False):
    hub_colors = ['blue', 'green', 'purple', 'pink', 'yellow']
    hub_colors_dict = dict(zip(hub_dictionary.keys(), hub_colors))

    print('Plotting figure...')

    fig, ax = ox.plot_graph(City.graph, show=False, save=False, close=False,
        figsize = City.figsize,
        bgcolor = City.bgcolor,
        edge_color = City.edge_color,
        node_color = City.node_color,
        edge_linewidth = City.edge_linewidth,
        node_size = City.node_size)
    
    # add spots for the hubs
    for hub_name, hub_value in hub_dictionary.items():
        color_to_use = hub_colors_dict[hub_name]
        current_label = hub_name
        #print(current_label, point, color_to_use)
        ax.scatter(hub_value['x'], hub_value['y'],
            color=color_to_use, marker='o', s=100, label=current_label)

    for index, row in City.building_addr_df.T.items():
        color_to_use = hub_colors_dict[row['nearesthub']]
        current_label = hub_name
        ax.scatter(row['x'], row['y'],
                    color=color_to_use, marker='o', s=5, label=current_label) 
    
    plt.show()

    if save:
        fig.savefig(f'data/plot_pngs/plot_{time.time()}.png')

    return fig, ax


@timer_decorator
def main():
    #initialization variables self.

    cpu_count = None
    ### what is the right info to import for Delft
    name = 'delft_walk'
    data_folder = 'data/'
    #vehicle_type = 'bike' # walk, bike, drive, all (See osmnx documentation)
    vehicle_type = 'walk'
    ### Delft
    #coordinates = [52.03, 51.96, 4.4, 4.3]

    ### Delft City Center
    #[N, S, E, W]
    coordinates = [52.018347, 52.005217, 4.369142, 4.350504]
    random_init = 100
    start_pt_ct = 5
    ###### ------- ######
    
    coordinates_list = coordinates_to_tupple(coordinates)
    coordinates_list_flip = flip_lat_long(coordinates_list)
    coordinates_transformed_yx = transform_coordinates(coordinates_list_flip)
    coordinates_transformed_xy = flip_lat_long(coordinates_transformed_yx)
    print(coordinates_transformed_xy)


    # ### generate new network, only run at beginning
    #generate_network(name, data_folder, vehicle_type, coordinates)

    ### load network from file, run after network is generated
    City, orig_xy_transf = load_network(name, data_folder)
    reset_hub_df_values(City)
    
    #City.ne = None

    orig_edges = nearest_edges_buildings(City, orig_xy_transf, name, data_folder, cpu_count)

    ### generate random points for hubs and cluster houses based on closest hub

    ###possible class functions
    ###self. hub_dictionary and self.City
    hub_dictionary = generate_random_points(coordinates_transformed_xy, start_pt_ct, random_init) 
    print(hub_dictionary)      
    
    hub_clusters(City, hub_dictionary, orig_xy_transf, orig_edges, name, data_folder, cpu_count)

    hub_fitness(City, hub_dictionary)

    visualize_clusters(City, hub_dictionary)

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