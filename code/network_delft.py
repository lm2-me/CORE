"""
CityNetwork class which builds network from OSM
and calculates shortest paths

By: 
Job de Vogel, TU Delft (framework, street data and shortest paths)
Jirri van den Bos, TU Delft (housing)

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
        - nearest_edges: calculate nearest edges to coordinate
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
import os.path
import pickle
import time

from shapely.geometry import LineString

from utils.multicore_shortest_path import multicore_shortest_path
from utils.multicore_nearest_edges import multicore_nearest_edge

class CityNetwork():
    # Plot settings
    figsize=(8, 8) 
    bgcolor = 'white'
    edge_color = 'gray'
    node_color = 'black'
    edge_linewidth = 1
    node_size=3
    route_color = 'darkorange'
    route_width = 3
    cross_color = 'red'

    def __init__(self, name: str, coordinates: list, transport_type, graph = None, graph_edges_df = None, graph_nodes_df = None):
        self.name = name
        self.coordinates = coordinates
        self.transport_type = transport_type
        self.graph = graph
        self.graph_edges_df = graph_edges_df
        self.graph_nodes_df = graph_nodes_df
        self.ne = None

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
                graph = ox.graph.graph_from_place(query, network_type=self.transport_type)
            else:
                graph = ox.graph_from_bbox(*self.coordinates, simplify=True, retain_all=False, network_type=self.transport_type, clean_periphery=True)
            
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

    def nearest_edges(self, x, y, interpolate, cpus=None, _multiply=1):
        if self.ne is None:
            # If _multiply is given, multiply the number of samples
            x *= _multiply
            y *= _multiply

            if len(x) != len(y):
                raise ValueError('Please make sure x and y have the same length.') 

            self.ne = multicore_nearest_edge(self.graph, x, y, interpolate, cpus=cpus)
            return self.ne
        else:
            return self.ne

    def shortest_paths(self, orig, dest, orig_edge, dest_edge, weight='travel_time', method='dijkstra', return_path=False, cpus=None, _multiply=1):
        # Check if all inputs are lists, required for multicore calculation
        if isinstance(orig, tuple):
            orig = [orig]
        if isinstance(dest, tuple):
            dest = [dest]
        if not isinstance(orig_edge, list):
            orig_edge = [orig_edge]
        if not isinstance(dest_edge, list):
            dest_edge = [dest_edge]
        
        # If _multiply is given, multiply the number of samples
        orig *= _multiply
        dest *= _multiply
        orig_edge *= _multiply
        dest_edge *= _multiply

        # Check if inputs have same length
        if not (len(orig) == len(dest) == len(orig_edge) == len(dest_edge)):
            raise ValueError('Please make sure orig, dest, orig_edge and dest_edge have the same length.') 
        
        # Execute in multicore
        # If only one item, multicore will fall back on just single operation
        paths = multicore_shortest_path(self.graph, orig, dest, orig_edge, dest_edge, weight=weight, method=method, return_path=return_path, cpus=cpus)

        return paths

    # Plot the map without any routes
    # Copy from osmnx and taxicab, slightly changed to work for multiple routes
    def plot(self, routes, origins, destinations, save=False):
        fig, ax = ox.plot_graph(self.graph, show=False, save=False, close=False,
            figsize = self.figsize,
            bgcolor = self.bgcolor,
            edge_color = self.edge_color,
            node_color = self.node_color,
            edge_linewidth = self.edge_linewidth,
            node_size = self.node_size)

        for route, orig, dest in zip(routes, origins, destinations):
            weight, route_nodes, ls_orig, ls_dest = route

            x, y  = [], []
            for u, v in zip(route_nodes[:-1], route_nodes[1:]):
                # if there are parallel edges, select the shortest in length
                data = min(self.graph.get_edge_data(u, v).values(), key=lambda d: d["length"])
                if "geometry" in data:
                    # if geometry attribute exists, add all its coords to list
                    xs, ys = data["geometry"].xy
                    x.extend(xs)
                    y.extend(ys)
                else:
                    # otherwise, the edge is a straight line from node to node
                    x.extend((self.graph.nodes[u]["x"], self.graph.nodes[v]["x"]))
                    y.extend((self.graph.nodes[u]["y"], self.graph.nodes[v]["y"]))
            ax.plot(x, y, c=self.route_color, lw=self.route_width)
            
            x, y = zip(*ls_orig.coords)
            ax.plot(x, y, c=self.route_color, lw=self.route_width)

            x, y = zip(*ls_dest.coords)
            ax.plot(x, y, c=self.route_color, lw=self.route_width)

            ax.scatter(orig[1], orig[0],
                color=self.cross_color, marker='x', s=100, label='orig-point')
            
            ax.scatter(dest[1], dest[0],
                color=self.cross_color, marker='x', s=100, label='orig-point')

        fig, ax = ox.plot._save_and_show(fig, ax)

        if save:
            fig.savefig(f'data/plot_pngs/plot_{time.time()}.png')

        return fig, ax

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