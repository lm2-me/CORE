from cgi import test
from pickletools import read_unicodestring1
import matplotlib.pyplot as plt

from network_delft import CityNetwork, timer_decorator
from utils.multicore_shortest_path import transform_coordinates

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
        data_folder + name + '_addresses.csv')
    
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
    destinations = [((51.916328, 4.473386))] * len(origins)

    # Extract the graph from the City CityNetwork
    graph = City.graph
    print(f"Processing {graph}")

    # Transform the start and origin to epsg:3857 (surface)
    orig_yx_transf = transform_coordinates(origins)
    dest_yx_transf = transform_coordinates(destinations)
    return 1

def generate_random_starting_points():
    return 1

def optimize_point_location():
    return 1

def add_points():
    return 1

def main():
    ### what is the right info to import for Delft
    name = 'Rotterdam_bike'
    data_folder = 'data/'
    #vehicle_type = 'bike' # walk, bike, drive, all (See osmnx documentation)
    vehicle_type = 'walk'
    ### Delft
    coordinates = [52.03, 51.96, 4.4, 4.3]

    ### generate new network, only run at beginning
    generate_network(name, data_folder, vehicle_type, coordinates)

    ### load network from file, run after network is generated
    #load_network(name, data_folder)



if __name__ == '__main__':
    main()