"""
CityNetwork class which builds network from OSM
and calculates shortest paths

By: 
Job de Vogel, TU Delft (framework, street data and shortest paths)
Jirri van den Bos, TU Delft (load building addresses)

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
"""

import osmnx as ox
import networkx as nx
import os.path
import pickle
import time
import overpy

from utils.multicore_shortest_path import multicore_shortest_path
from utils.multicore_nearest_edges import multicore_nearest_edge
from utils.osm_data_request import get_addr_query, get_building_query, get_osm_addr, get_osm_building, compare_building_addr, load_csv, addxy_building_addr, get_CBS_query, get_CBS_data, read_CBS, compare_building_cbs

class CityNetwork():
    # Plot settings
    figsize=(8, 8) 
    bgcolor = '#181717'
    edge_color = 'lightgray'
    node_color = 'white'
    edge_linewidth = 1
    node_size=3
    route_color = '#C90808'
    route_width = 3
    origin_color = '#FFE54F'
    destination_color = '#82C5DA'
    marker_color = 'purple'
    font_color = 'lightgray'
    font_size = 7
    
    def __init__(self, name: str, coordinates: list, transport_type):
        self.name = name
        self.coordinates = coordinates
        self.transport_type = transport_type
        self.graph = None
        self.graph_edges_df = None
        self.graph_nodes_df = None
        self.building_addr_df = None
        self.url = [None, "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://lz4.overpass-api.de/api/interpreter"]
        self.ne = None
        self.interpolation = None

    def __repr__(self):
        return "<CityNetwork object of {}>".format(self.name)

    # Load an osm graph from online or local saved file
    # Query indicates geocodable location such as 'Delft'
    def load_osm_graph(self, filepath: str, query=False):
        # If Electric personal vehicle: use bike data
        transport_type = self.transport_type
        if transport_type == 'epv':
            transport_type = 'bike'
        
        if os.path.isfile(filepath):
            # Load a graph from drive
            print("Loading street network...")
            graph = ox.load_graphml(filepath=filepath)

        else:
            # Retrieve graph online and save local
            print("Retrieving online data...")
            if query:
                graph = ox.graph.graph_from_place(query, network_type=transport_type)
            else:
                graph = ox.graph_from_bbox(*self.coordinates, simplify=True, retain_all=False, network_type=transport_type, clean_periphery=True)
            
            print("Saving...")
            ox.save_graphml(graph, filepath=filepath)
        
        self.graph = graph
        print("Finished loading")

    def load_building_addr(self, building_addr_path: str, building_path: str, adress_path: str, cbs_path: str):
        print("Loading buildings...")
        cbs_properties = ['geom','gemiddeldeHuishoudsgrootte']#,'buurtcode','buurtnaam','gemeentenaam']

        if not (os.path.isfile(building_addr_path)):
            if not (os.path.isfile(adress_path) and os.path.isfile(building_path)):
                building_query = get_building_query(([''], self.coordinates))
                addr_query = get_addr_query(([''], self.coordinates))

                error = 0
                for i in self.url:
                    try:
                        addr_frame = get_osm_addr(addr_query, url=i)
                        addr_frame.to_csv(adress_path)
                        print(f"OSM address data saved to {adress_path}")

                        building_frame = get_osm_building(building_query, url=i)
                        building_frame.to_csv(building_path)
                        print(f"OSM building data saved to {building_path}")
                        break
                    except overpy.exception.OverpassGatewayTimeout as exc:
                        print(exc)
                        error += 1
                        print("Overpass Server Load too high for standard servers, retrying with different url")
                        pass
                    except TypeError as exc:
                        print(exc)
                        error += 1
                        print("Trying non standard server did not return a valid result, retrying with different server")
                        pass
                
                if error == len(self.url):
                    print("The request is currently unable to gather Overpass data, please retry manually in 30 seconds")
                    exit()
            # Load the building and adress data from csv
            addr_frame =  load_csv(adress_path)
            building_frame = load_csv(building_path)

            building_addr_df = compare_building_addr(building_frame, addr_frame)
            building_addr_df.to_csv(building_addr_path)
            print(f"Building/Adress data saved to {building_addr_path}")
        else:
            building_addr_df = load_csv(building_addr_path)
        
        building_addr_df = addxy_building_addr(building_addr_df)
        CBS_query = get_CBS_query(([''], self.coordinates), cbs_properties, buurt_index_skip=[0])
        if not (os.path.isfile(cbs_path)):
            get_CBS_data(CBS_query, cbs_path)
        print(cbs_path)
        CBS_data = read_CBS(cbs_path)

        building_addr_df= compare_building_cbs(building_addr_df, CBS_data, cbs_properties)

        # Save to CityNetwork object
        self.building_addr_df = building_addr_df

    # Add speed, length and travel time to graph
    def add_rel_attributes(self, overwrite_walk=None, overwrite_bike=None, overwrite_epv=False):

        graph = ox.distance.add_edge_lengths(self.graph)
        graph = ox.speed.add_edge_speeds(graph)

        if self.transport_type == 'walk' and overwrite_walk != None: 
            nx.set_edge_attributes(graph, values=overwrite_walk, name="speed_kph")
        elif self.transport_type == 'bike' and overwrite_bike != None:
            nx.set_edge_attributes(graph, values=overwrite_bike, name="speed_kph")
        elif self.transport_type == 'epv' and overwrite_epv != None:
            nx.set_edge_attributes(graph, values=overwrite_epv, name="speed_kph")

        graph = ox.speed.add_edge_travel_times(graph)

        self.graph = graph

    def add_street_experience(self, names=[], factors=[]):
        nx.set_edge_attributes(self.graph, 0, 'experience')

        for osmid in self.graph.edges:
            self.graph.edges[osmid]['experience'] = self.graph.edges[osmid]['length']

        for factor in factors:
            for osmid in self.graph.edges:
                if 'name' in self.graph.edges[osmid]:
                    if self.graph.edges[osmid]['name'] not in names:
                        self.graph.edges[osmid]['experience'] = self.graph.edges[osmid]['experience'] * factor
    
    def add_coord_experience(self, street_uvk=[], factors=[]):
        nx.set_edge_attributes(self.graph, 0, 'experience')
        
        for street, factor in zip(street_uvk, factors):
            node_1, node_2, _ = street
            print(street)

            for edge in self.graph.edges:
                if node_1 in edge and node_2 in edge:
                    self.graph.edges[edge]['experience'] = self.graph.edges[edge]['length']
                else:
                    self.graph.edges[edge]['experience'] = self.graph.edges[edge]['length'] * factor

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
    def save_graph(self, name: str, folder: str):        
        object_name = name
        path = folder + str(object_name) + '.pkl'
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
            
            multicore_edge_result = multicore_nearest_edge(self.graph, x, y, interpolate, cpus=cpus)
            self.ne, self.interpolation = multicore_edge_result[0], multicore_edge_result[1]
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
    # Copy from osmnx and taxicab, changed to work for multiple routes, annotations and marks
    def plot(self, routes=None, origins=None, destinations=None, annotations=False, marks=[], save=False):
        print('Plotting figure...')
        if routes is not None and origins is not None and destinations is not None:
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
                
                if not isinstance(ls_orig, list):
                    x, y = zip(*ls_orig.coords)
                    ax.plot(x, y, c=self.route_color, lw=self.route_width)

                if not isinstance(ls_dest, list):
                    x, y = zip(*ls_dest.coords)
                    ax.plot(x, y, c=self.route_color, lw=self.route_width)

                ax.scatter(orig[1], orig[0],
                    color=self.origin_color, marker='x', s=100, label='orig-point')
                
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
    @staticmethod
    def load_graph(name: str, folder: str):
        object_name = name
        path = folder + str(object_name) + '.pkl'

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