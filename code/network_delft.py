import osmnx as ox
import taxicab as tc
import os.path
import pickle
import time
import matplotlib.pyplot as plt
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
        self.graph = ox.project_graph(self.graph, to_crs="crs")

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

    Args:
        - name: name of the CityNetwork, string
    """
    def save_graph(self, name: str):        
        object_name = name
        path = 'data/' + str(object_name) + '.pkl'
        print('Saving {} to {}'.format(object_name, path))

        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)

    """ Calculate the shortest path between one or multiple points
        Inputs can be lists of osmid's, single osmid's or coordinate_tuples

        Options:
            orig, dest = 78503718
            orig, dest = [8392089, 3289023, 238979234]
            Can also be one orig and list of dest
            
            orig = (52.048378, 4.2832923)

        Based on input type, taxicab or osmnx is selected
    """
    def shortest_paths(self, orig: int or list or tuple, dest: int or list or tuple, weight = 'travel_distance', from_coordinates = False, plot=False):
        print('Calculating paths...')
        start = time.time()
        
        # If from osmids, not from coordinates
        if not from_coordinates:
            if type(orig) != type(dest):
                if isinstance(orig, list):
                    dest = [dest for _ in range(len(orig))]
                else:
                    orig = [orig for _ in range(len(dest))]
            paths = ox.shortest_path(self.graph, orig, dest, weight=weight, cpus=None)
        
        # You are using coordinate tuples
        else:
            print('Taxicab initiated...')
            # Calculate shortest paths with taxicab package
            paths = tc.distance.shortest_path(self.graph, orig, dest)     
        
        end = time.time()
        print('Finished calculating {} path(s) in {} seconds.'.format(len(orig) if isinstance(orig, list) else 1, round(end-start, 2))) 

        # Plot the figure using osmnx or taxicab
        if plot and not from_coordinates:
            print("Loading plot...")
            if isinstance(orig, list) or isinstance(dest, list):
                ox.plot_graph_routes(self.graph, paths, route_linewidth=self.route_linewidth,
                    # WIP: Make this look nicer:
                    figsize=self.figsize, bgcolor=self.bgcolor, edge_color=self.edge_color, node_color=self.node_color, node_size=self.node_size)
            else:
                ox.plot_graph_route(self.graph, paths, route_linewidth=self.route_linewidth,
                    figsize=self.figsize, bgcolor=self.bgcolor, edge_color=self.edge_color, node_color=self.node_color, node_size=self.node_size)
        elif plot:
            # Now make it multiprocessing!

            """ EXAMPLE:
            from multiprocessing import Pool

            def print_range(lrange):
                print('First is {} and last is {}'.format(lrange[0], lrange[1]))


            def run_in_parallel():
                ranges = [[0, 10], [10, 20], [20, 30]]
                pool = Pool(processes=len(ranges))
                pool.map(print_range, ranges)


            if __name__ == '__main__':
                run_in_parallel()
            """

            tc.plot.plot_graph_route(self.graph, paths, route_linewidth=self.route_linewidth,
                    figsize=self.figsize, bgcolor=self.bgcolor, edge_color=self.edge_color, node_color=self.node_color, node_size=self.node_size)
        return paths
    
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
        
        print('Loaded {} to object'.format(object_name))
        
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
    # # Initialize CityNetwork object
    # Delft = CityNetwork('Delft', [52.03, 51.96, 4.4, 4.3])
    
    # # Load osm from local or online file
    # Delft.load_osm_graph('data/Delft.osm')
    
    # # Add speeds, lengths and distances to graph
    # Delft.add_rel_attributes()

    # # Project graph                   # Not sure how this works!!!
    # Delft.project_graph()

    # # Calculate dataframes of nodes and edges
    # Delft.convert_graph_edges_to_df()
    # Delft.convert_graph_nodes_to_df()

    # # Save Pickle file
    # Delft.save_graph('Delft')

    """
    Load network from pickle file
    """
    Delft = CityNetwork.load_graph('Delft')

    # Pick a random origin
    # orig = Delft.get_node_osmids()[945]
    orig = (51.99274, 4.35108)

    # Pick a random range of destinations
    # dest = Delft.get_node_osmids()[0:100]
    dest = (52.02184, 4.37690)

    Delft.shortest_paths(orig, dest, from_coordinates=True, plot=True)

if __name__ == '__main__':
    main()