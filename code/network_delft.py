import osmnx as ox
import os.path
import pickle
import time

class CityNetwork():
    def __init__(self, name, coordinates, graph = None, graph_edges_df = None, graph_nodes_df = None):
        self.name = name
        self.coordinates = coordinates
        self.graph = graph
        self.graph_edges_df = graph_edges_df
        self.graph_nodes_df = graph_nodes_df

    def __repr__(self):
        return "<CityNetwork object of {}>".format(self.name)

    # Load an osm graph from online or local saved file
    # Query indicates geocodable location such as 'Delft'
    def load_osm_graph(self, filepath, query=False):
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
    
    """
    # Storing the graph as a pickle file avoids having to recalculate
    # attributes such as speed, length etc. and is way faster

    Args:
        - name: name of the CityNetwork, string
    """
    def save_graph(self, name):        
        object_name = name
        path = 'data/' + str(object_name) + '.pkl'
        print('Saving {} to {}'.format(object_name, path))

        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    # Loading the graph as a pickle file avoids having to recalculate
    # attributes such as speed, length etc. and is way faster
    # Call this function through var = CityNetwork.load_graph('var')
    @staticmethod
    def load_graph(name):
        object_name = name
        path = 'data/' + str(object_name) + '.pkl'

        print("Loading...")
        with open(path, 'rb') as file:
            graph = pickle.load(file)
        
        print('Loaded {} to object'.format(object_name))
        
        return graph

    # Coming ....
    # Access edge osmids, not sure how this data is formatted
    # u,v indicates osmids of origin and destination node osmids

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
    # Initialize CityNetwork object
    # Delft = CityNetwork('Delft', [52.03, 51.96, 4.4, 4.3])
    
    # Load osm from local or online file
    # Delft.load_osm_graph('data/Delft.osm')
    
    # Add speeds, lengths and distances to graph
    # Delft.add_rel_attributes()

    # Calculate dataframes of nodes and edges
    # Delft.convert_graph_edges_to_df()
    # Delft.convert_graph_nodes_to_df()

    # Save Pickle file
    # Delft.save_graph('Delft')

    """
    Load network from pickle file
    """
    Delft = CityNetwork.load_graph('Delft')

    print(Delft)
    print(Delft.graph_nodes_df)

if __name__ == '__main__':
    main()