"""
CityNetwork class which builds network from OSM
and calculates shortest paths

Classes:
    - CityNetwork

    Functions:
        - load_osm_graph: load osm data to networkx graph online or local
        - add_rel_attributes: calculate speed, length and travel_time
        - project_graph: project graph on different coordinate system
        - convert_graph_edges_to_df: build dataframe for all edges
        - convert_graph_nodes_to_df: build dataframe for all nodes
        - get_edge_attribute_types: get all attribute types from edge dataframe
        - get_node_attribute_types: get all attribute types from node dataframe
        - get_node_osmids: get all osm id's from nodes
        - get_edge_osmids: get all osm id's from edges
        - save_graph: save graph to pickle file
        - load_graph: load graph from pickle file
        - shortest_paths: calculate shortest paths between nodes or coordinates
        - plot: plot city graph without routes

Decorators:
    - timer_decorator: time a function

Other functions:
    - main: calculate the shortest path

TO DO:
Manually implement shortest path over graph using
the osmnx shortest path parallel computation method
and the a* method of networkx.

Also combine method with taxicab procedure
"""

import osmnx as ox
import taxicab as tc
import os.path
import pickle
import time
import warnings

# Ignore the userwarning from taxicab temporarily
warnings.simplefilter("ignore")

class CityNetwork():
    # Plot settings
    figsize=(8, 8) 
    bgcolor = 'white'
    edge_color = 'gray'
    node_color = 'black'
    route_linewidth = 1
    node_size=3

    def __init__(self, name: str, coordinates: list, graph = None, graph_edges_df = None, graph_nodes_df = None):
        self.name = name
        self.coordinates = coordinates
        self.graph = graph
        self.graph_edges_df = graph_edges_df
        self.graph_nodes_df = graph_nodes_df

    def __repr__(self):
        return "<CityNetwork object of {}>".format(self.name)

    # Load an osm graph from online or local saved file
    # Query indicates geocodable location such as 'Delft'
    def load_osm_graph(self, filepath: str, query=False):
        if os.path.isfile(filepath):
            # Load a graph from drive
            print("Loading...")
            graph = ox.load_graphml(filepath=filepath)

        else:
            # Retrieve graph online and save local
            print("Retrieving online data...")
            if query:
                graph = ox.graph.graph_from_place(query, network_type='drive')
            else:
                graph = ox.graph_from_bbox(*self.coordinates, simplify=True, retain_all=False, network_type='all', clean_periphery=True)
            
            print("Saving...")
            ox.save_graphml(graph, filepath=filepath)
        
        self.graph = graph
        print("Finished loading")

    # Add speed, length and travel time to graph
    def add_rel_attributes(self):
        graph = ox.speed.add_edge_speeds(self.graph)
        graph = ox.distance.add_edge_lengths(graph)
        graph = ox.speed.add_edge_travel_times(graph)

        self.graph = graph

    def project_graph(self):
        self.graph = ox.project_graph(self.graph, to_crs="EPSG:3857")

    # Create a dataframe for the edges    
    def convert_graph_edges_to_df(self):
        if self.graph is not None:
            self.graph_edges_df =  ox.graph_to_gdfs(self.graph, nodes=False)
        else:
            raise ValueError('CityNetwork object does not contain graph.')
    
    # Create a dataframe for the nodes
    def convert_graph_nodes_to_df(self):
        if self.graph is not None:
            self.graph_nodes_df = ox.graph_to_gdfs(self.graph, edges=False)
        else:
            raise ValueError('CityNetwork object does not contain graph.')
    
    # Get all node or edge attribute types included in data
    def get_edge_attribute_types(self):
        if self.graph_edges_df is None:
            raise ValueError('Dataframe of edges does not exist, please generate df with convert_graph_edges_to_df()')
        
        edge_attributes = self.graph_edges_df.columns
        return edge_attributes
    
    def get_node_attribute_types(self):
        if self.graph_nodes_df is None:
            raise ValueError('Dataframe of nodes does not exist, please generate df with convert_graph_nodes_to_df()')
        
        node_attributes = self.graph_nodes_df.columns
        return node_attributes
    
    # Access all node osmid's from graph
    # Returns a list
    def get_node_osmids(self):
        if self.graph is None:
            raise ValueError('Graph does not exist, please generate graph with load_osm_graph()')
        return list(self.graph.nodes())
    
    def get_edge_osmids(self):
        if self.graph is None:
            raise ValueError('Graph does not exist, please generate graph with load_osm_graph()')
        return self.graph_edges_df['osmid'].values

    """ Storing the graph as a pickle file avoids having to recalculate
    attributes such as speed, length etc. and is way faster
    """
    def save_graph(self, name: str):        
        object_name = name
        path = 'data/' + str(object_name) + '.pkl'
        print('Saving {} to {}'.format(object_name, path))

        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    def shortest_paths(self, orig: int or list or tuple, dest: int or list or tuple, weight = 'travel_distance', from_coordinates = False, plot=False):
        print('Calculating paths...')
        pass

    # Plot the map without any routes
    def plot(self):
        print("Loading plot...")
        ox.plot_graph(self.graph, figsize=self.figsize, bgcolor=self.bgcolor, edge_color=self.edge_color, node_color=self.node_color, node_size=self.node_size)

    # Loading the graph as a pickle file avoids having to recalculate
    # attributes such as speed, length etc. and is way faster
    # Call this function through var = CityNetwork.load_graph('var')
    @staticmethod
    def load_graph(name: str):
        object_name = name
        path = 'data/' + str(object_name) + '.pkl'

        print("Loading...")
        with open(path, 'rb') as file:
            graph = pickle.load(file)
        
        print('Loaded {} to object...'.format(object_name))
        
        return graph

# Decorator to time functions, just a useful decorator
def timer_decorator(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        print('Executed {} fuction in {}s'.format(func.__name__, (round(end-start,2))))
        return func
    return wrapper

@timer_decorator
def main():
    """
    Initialize network if not already saved as pickle file:
    """
    # # # Initialize CityNetwork object
    Delft = CityNetwork('Delft', [52.03, 51.96, 4.4, 4.3])
    
    # # # Load osm from local or online file
    Delft.load_osm_graph('data/Delft.osm')
    
    # # # Add speeds, lengths and distances to graph
    Delft.add_rel_attributes()

    # # # Project graph
    Delft.project_graph()
    # # Delft.plot()

    # # # Calculate dataframes of nodes and edges
    Delft.convert_graph_edges_to_df()
    Delft.convert_graph_nodes_to_df()

    # # Save Pickle file
    Delft.save_graph('Delft')

    # """
    # Load network from pickle file
    # """
    # Delft = CityNetwork.load_graph('Delft')

if __name__ == '__main__':
    main()