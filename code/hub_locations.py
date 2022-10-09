"""
functions related to creating and optimizing hub locations

By: 
Job de Vogel, TU Delft (generate and load network)
Lisa-Marie Mueller, TU Delft (hub locations, clustering, hub location optimization)

Classes:
    - None

    Functions:
        - 

Other functions:
    - main: optimizes hub locations
"""

import matplotlib.pyplot as plt
import numpy as np
import math as m
import osmnx as ox
import random
import time

from re import X
from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import transform_coordinates
from utils.multicore_nearest_edges import multicore_nearest_edge

def generate_network(name, data_folder, vehicle_type, coordinates):
    ### Job's Code: generate network function
    ''' --- GENERATE NETWORK ---
    Generate a new network using the functions in the CityNetwork class. If a network already has been generated and stored in the data folder, comment this part and continue with PREPARE NETWORK. '''
    # Initialize CityNetwork object [N, S, E, W]
    City = CityNetwork(name, coordinates, vehicle_type)
    
    # Load osm from local or online file
    City.load_osm_graph(data_folder + name + '.osm')
    City.load_building_addr(data_folder + name + '_building_addresses.csv', 
        data_folder + name + '_buildings.csv', 
        data_folder + name + '_addresses.csv',
        data_folder +'runtime/'+ name + '_cbs.xml')

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


def load_network(name, data_folder):
    ###Job's Code load network function
    ''' --- PREPARE NETWORK ---
    Load the network from .pkl file. Transform the origin and destination to the unprojected space.'''

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
    # for index, row in City.building_addr_df.T.items():
    for index, row in City.building_addr_df.iterrows():
        coordinate = row['y'], row['x']
        orig_yx_transf.append(coordinate)

    return City, orig_yx_transf

### Reset the data stored in the dataframe so that previous runs won't impact the current run
def reset_hub_df_values(City):
    if 'nearesthub' not in City.building_addr_df:
        hub_num = [('None')] * len(City.building_addr_df)
        City.building_addr_df['nearesthub'] = hub_num
    else:
        City.building_addr_df['nearesthub'].values[:] = 'None'
    
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

### find the nearest network edges for all building locations
def nearest_edges_buildings(City, orig_yx_transf, name, data_folder, cpu_count):
    x = []
    y = []

    for orig_yx in orig_yx_transf:
        x.append(orig_yx[1])
        y.append(orig_yx[0])
    
    edges = City.nearest_edges(x, y, 5, cpus=cpu_count)
    City.save_graph(name, data_folder)
    orig_edge = edges
    #print('orig_edge', orig_edge)
    return orig_edge

### find the nearest network edges for all hub locations
def nearest_edges_hubs(City, hub_yx_transf, cpu_count):
    x = []
    y = []

    for hub in hub_yx_transf:
        x.append(hub[1])
        y.append(hub[0])
    
    edges, _ = multicore_nearest_edge(City.graph, x, y, City.interpolation, cpus=cpu_count)
    #print(f"edge: {edges}")
    return edges

### convert city coordinates into a tupple
def coordinates_to_tupple(coordinates):
    #[N, S, E, W]
    w_n_corner = (coordinates[3], coordinates[0]) 
    e_n_corner = (coordinates[2], coordinates[0]) 
    w_s_corner = (coordinates[3], coordinates[1]) 
    e_s_corner = (coordinates[2], coordinates[1]) 

    return ([w_n_corner, e_n_corner, w_s_corner, e_s_corner])

### flip the order of latitude and longitude in coordinates
def flip_lat_long(coordinates_list):
    return [(coord[1], coord[0]) for coord in coordinates_list]

### generate random points within the boundary of the loaded city at which to place hub locations
def generate_random_points(coordinates_transformed_xy,start_pt_ct, hub_dictionary=None):
    print('generate random points', hub_dictionary)
    if hub_dictionary == None:
        hub_dictionary = {}

    #[N, S, E, W]
    #w_n_corner, e_n_corner, w_s_corner, e_s_corner
    coordinateX_min = coordinates_transformed_xy[0][0]
    coordinateX_max = coordinates_transformed_xy[1][0]
    coordinateY_min = coordinates_transformed_xy[2][1]
    coordinateY_max = coordinates_transformed_xy[0][1]

    #print(coordinateX_min, coordinateX_max, coordinateY_min, coordinateY_max)

    index = len(hub_dictionary)+1

    for i in range(start_pt_ct):
        x = random.uniform(coordinateX_min, coordinateX_max)
        y = random.uniform(coordinateY_min, coordinateY_max)

        index_name = 'hub ' + str(index)

        if index_name not in hub_dictionary:
            hub_dictionary[index_name] = {
                "x": x,
                "y": y,
                "avg_time": 0,
                "max_time": 0,
                "people_served": 0,
            } 
        index += 1

    return hub_dictionary

### get the yx transform values from the dictionary containing hub locations
def get_yx_transf_from_dict(hub_dictionary):
    label_list = []
    value_list = []
    for (hub_name, hub_loc) in hub_dictionary.items():
        
        label_list.append(hub_name)
        value_list.append(
            (hub_loc['y'], hub_loc['x'])
        )

    return label_list, value_list

### cluster houses to each hub based on the euclidean distance to each hub
def hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder):
    ### randomly generated hub locations is hub_dictionary
    hub_names, hub_yx_transf = get_yx_transf_from_dict(hub_dictionary)
    
    hub_yx_dict = dict(zip(hub_names, hub_yx_transf))

    ### first_paths returns [(route_weight, nx_route, orig_partial_edge, dest_partial_edge)]
    for hub in hub_names:
        print(hub)
        #get distance of all current hub to all houses        
        point2 = np.array(hub_yx_dict[hub])
        ## print(point2)

        for i, house in enumerate(orig_yx_transf):
            point1 = np.array(house)
            euclid_dist = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5
          
            if euclid_dist < City.building_addr_df.loc[i,'euclid_hubdistance']:
                City.building_addr_df.at[i,'euclid_hubdistance']=euclid_dist
                City.building_addr_df.at[i,'euclid_nearesthub']=hub
    
    City.save_graph(name, data_folder)
    
    return 0

### from a combined list of distance to all hubs, get the sublists for each hub
def get_sublists(list, sublist_number):
    sub_list_length = len(list) // sublist_number
    list_of_sublists = []
    for i in range(0, len(list), sub_list_length):
        list_of_sublists.append(list[i:i+sub_list_length])
    return list_of_sublists

### cluster houses to each hub based on the travel distance to each hub
def hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count):
    ### randomly generated hub locations is hub_dictionary
    ### cluster houses around closest hub locations

    hub_names, hub_yx_transf = get_yx_transf_from_dict(hub_dictionary)
    dest_edges = nearest_edges_hubs(City, hub_yx_transf, cpu_count)
    
    hub_yx_dict = dict(zip(hub_names, hub_yx_transf))
    hub_edge_dict = dict(zip(hub_names, dest_edges))
    
    hub_yx_transf_long = []
    dest_edges_long = []

    ### first_paths returns [(route_weight, nx_route, orig_partial_edge, dest_partial_edge)]
    for hub in hub_names:
        print(hub)
        #get distance of all current hub to all houses
        hub_yx_transf_current = [hub_yx_dict[hub]] * len(orig_yx_transf)
        dest_edges_current = [hub_edge_dict[hub]] * len(orig_yx_transf)

        hub_yx_transf_long = hub_yx_transf_long + hub_yx_transf_current
        dest_edges_long = dest_edges_long + dest_edges_current

    orig_yx_transf_long = orig_yx_transf * len(hub_dictionary)
    orig_edges_long = orig_edges * len(hub_dictionary)
    
    # weight options: travel_time, length
    # Returning route_weight, nx_route, orig_partial_edge, dest_partial_edge, orig_yx, dest_yx
    # [(route_weight, nx_route, orig_partial_edge, dest_partial_edge, orig_yx, dest_yx)] 
    hub_dist = City.shortest_paths(orig_yx_transf_long, hub_yx_transf_long, orig_edges_long, dest_edges_long, weight='travel_time', method='dijkstra', return_path=True, cpus=cpu_count)

    list_of_sublists = get_sublists(hub_dist, len(hub_dictionary))

    for i, sublist in enumerate(list_of_sublists):
        hub = hub_names[i]
        for j, row in enumerate(sublist):
            dist = row[0]
            if dist < City.building_addr_df.loc[j,'hubdistance']:
                City.building_addr_df.at[j,'hubdistance']=dist
                City.building_addr_df.at[j,'nearesthub']=hub
                City.building_addr_df.at[j,'hub_x']=hub_dictionary[hub]['x']
                City.building_addr_df.at[j,'hub_y']=hub_dictionary[hub]['y']
              
    City.save_graph(name, data_folder)
    #df_print = City.building_addr_df[['x', 'y', 'hubdistance', 'nearesthub']]
    #print(df_print)
    ### 2.5 min is 150 seconds 
    return 0

### calculate the hub fitness
def hub_fitness(City, hub_dictionary, max_travel_time):
    # dictionary: x, y, avg_time, people_served
    # find and save average time and people served to hub_dictionary
    
    #print(City.building_addr_df[['x', 'y', 'hubdistance', 'nearesthub']])
    time_requirement = False
    max_time_list = []
    
    for (hub_name, _) in hub_dictionary.items():
        all_times = []
        all_people = []
        for _, row in City.building_addr_df.iterrows():
            if row['nearesthub'] == hub_name:
                #travel time
                time = row['hubdistance']
                all_times.append(time)
                #people served
                people = row['addr']
                all_people.append(people)

        all_times_np = np.array(all_times)
        average = np.sum(all_times_np) / len(all_times)
        all_people_np = np.array(all_people)
        total_people = np.sum(all_people_np)
        hub_dictionary[hub_name]['avg_time'] = average

        max_time_list.append(np.max(all_times_np))
        hub_dictionary[hub_name]['max_time'] = np.max(all_times_np)

        hub_dictionary[hub_name]['people_served'] = total_people
    
    max_time_list_np = np.array(max_time_list)
    time_check = all(i <= max_travel_time for i in max_time_list_np)

    return time_check, max_time_list

### move the hub based on travel distance of all houses in the hub cluster
def new_hub_location(City, hub_dictionary):
    for (hub_name, hub_data) in hub_dictionary.items():
        dist_moved = 0
        x = []
        y = []
        for _, row in City.building_addr_df.iterrows():
            if row['nearesthub'] == hub_name:
                #x
                house_x = row['x']
                x.append(house_x)
                #y
                house_y = row['y']
                y.append(house_y)   
        all_x = np.array(x)
        all_y = np.array(y)
        average_x = np.sum(all_x) / len(all_x)
        average_y = np.sum(all_y) / len(all_y)
        previous_location = (hub_dictionary[hub_name]['x'], hub_dictionary[hub_name]['y'])
        hub_dictionary[hub_name]['x'] = average_x
        hub_dictionary[hub_name]['y'] = average_y
        new_location = (hub_dictionary[hub_name]['x'], hub_dictionary[hub_name]['y'])

        point1 = previous_location
        point2 = new_location
        move_distance = ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5 


    return move_distance

### add new random hub points
def add_points(City, hub_dictionary, max_time_list, max_travel_time, coordinates_transformed_xy):
    sorted_max_time = np.sort(np.array(max_time_list))
    point_count = m.ceil(sorted_max_time[-1] / max_travel_time)
    print('adding ', point_count, ' points')
    generate_random_points(coordinates_transformed_xy, point_count, hub_dictionary=hub_dictionary)
    return hub_dictionary

### visualize the hub clusters based on travel time
def visualize_clusters(City, hub_dictionary, text, hub_colors, save=False):
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
        if row['nearesthub'] == 'None':
            color_to_use = 'white'
        else:
            color_to_use = hub_colors_dict[row['nearesthub']]

        current_label = hub_name
        ax.scatter(row['x'], row['y'],
                    color=color_to_use, marker='o', s=5, label=current_label) 
    
    plt.show()

    if save:
        fig.savefig(f'data/plot_pngs/plot_'+ text +f'_{time.time()}.png')

    return fig, ax

### visualize the hub clusters based on euclidean distance
def euclid_visualize_clusters(City, hub_dictionary, hub_colors, save=False):
    
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
        color_to_use = hub_colors_dict[row['euclid_nearesthub']]
        current_label = hub_name
        ax.scatter(row['x'], row['y'],
                    color=color_to_use, marker='o', s=5, label=current_label) 
    
    plt.show()

    if save:
        fig.savefig(f'data/plot_pngs/plot_euclid_{time.time()}.png')

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
    max_travel_time = 150 #time in seconds

    ### Delft City Center
    #[N, S, E, W]
    coordinates = [52.018347, 52.005217, 4.369142, 4.350504]
    random_init = 100
    start_pt_ct = 3
    random.seed(random_init)

    hub_colors = ['#FFE54F', '#82C5DA', '#C90808', '#FAA12A', '#498591', '#E64C4C', '#E4FFFF', '#CA5B08' '#3F4854']
    ###### ------- ######
    
    coordinates_list = coordinates_to_tupple(coordinates)
    coordinates_list_flip = flip_lat_long(coordinates_list)
    coordinates_transformed_yx = transform_coordinates(coordinates_list_flip)
    coordinates_transformed_xy = flip_lat_long(coordinates_transformed_yx)

    ### generate new network, only run at beginning
    generate_network(name, data_folder, vehicle_type, coordinates)

    ### load network from file, run after network is generated
    City, orig_yx_transf = load_network(name, data_folder)
    reset_hub_df_values(City)
    
    #City.ne = None

    orig_edges = nearest_edges_buildings(City, orig_yx_transf, name, data_folder, cpu_count)

    ### generate random points for hubs and cluster houses based on closest hub

    ###possible class functions
    ###self. hub_dictionary and self.City
    hub_dictionary = generate_random_points(coordinates_transformed_xy, start_pt_ct)
    visualize_clusters(City, hub_dictionary, 'first_hub_locations', hub_colors, save=True)

    print(hub_dictionary) 
    hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count)
    hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder)
    hub_fitness(City, hub_dictionary, max_travel_time)

    visualize_clusters(City, hub_dictionary, 'first_iteration', hub_colors, save=True)

    move_distance = new_hub_location(City, hub_dictionary)
    i = 0

    while move_distance > 50 and i < 10:
        hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count)
        hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder)
        move_distance = new_hub_location(City, hub_dictionary)
        print('moved hubs on average ', move_distance)

        i += 1

    time_check, max_time_list = hub_fitness(City, hub_dictionary, max_travel_time)
    visualize_clusters(City, hub_dictionary, 'optimized', hub_colors, save=True)

    print('time check', time_check)

    if not time_check:
        add_points(City, hub_dictionary, max_time_list, max_travel_time, coordinates_transformed_xy)
    
    print('added points', hub_dictionary)

    visualize_clusters(City, hub_dictionary, 'added_clusters', hub_colors, save=True)
    #euclid_visualize_clusters(City, hub_dictionary, hub_colors, save=True)

    ### FOR MIDTERM VISUALIZAION ONLY ###
    reset_hub_df_values(City)
    hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count)
    hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder)
    hub_fitness(City, hub_dictionary, max_travel_time)

    visualize_clusters(City, hub_dictionary, 'second_iteration', hub_colors, save=True)

    move_distance = new_hub_location(City, hub_dictionary)
    j = 0

    while move_distance > 50 and j < 10:
        hub_clusters(City, hub_dictionary, orig_yx_transf, orig_edges, name, data_folder, cpu_count)
        hub_clusters_euclidean(City, hub_dictionary, orig_yx_transf, name, data_folder)
        move_distance = new_hub_location(City, hub_dictionary)
        print('moved hubs on average ', move_distance)

        j += 1

    time_check, max_time_list = hub_fitness(City, hub_dictionary, max_travel_time)
    visualize_clusters(City, hub_dictionary, 'optimized2', hub_colors, save=True)
    ### FOR MIDTERM VISUALIZAION ONLY ###

if __name__ == '__main__':
    main()