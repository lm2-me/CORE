"""
CityNetwork class which builds network from OSM
and calculates shortest paths

By: 
Job de Vogel, TU Delft (framework, street data and shortest paths)
Jirri van den Bos, TU Delft (load building addresses and osm graph)

Classes:
    - CityNetwork

    Methods:
        >>> load_osm_graph: load osm data to networkx graph online or local
        >>> load_building_addr: load building and adress data from cbs
        >>> add_rel_attributes: calculate speed, length and travel_time
        >>> add_street_experience: add experience scores based on street name
        >>> add_coord_experience: add experience scores based on coordinates
        >>> project_graph: project graph on different coordinate system
        >>> convert_graph_edges_to_df: build dataframe for all edges
        >>> convert_graph_nodes_to_df: build dataframe for all nodes
        >>> get_edge_attribute_types: get all attribute types from edge dataframe
        >>> get_node_attribute_types: get all attribute types from node dataframe
        >>> get_node_osmids: get all osm id's from nodes
        >>> get_edge_osmids: get all osm id's from edges
        >>> save_graph: save graph to pickle file
        >>> load_graph: load graph from pickle file
        >>> drop_outliers: drop outlier buildings in the graph
        >>> nearest_edges: calculate nearest edges to coordinate
        >>> shortest_paths: calculate shortest paths between nodes or coordinates
        >>> plot: plot city graph without routes
"""

import osmnx as ox
import networkx as nx
import os.path
import pickle
import time
import overpy

from .utils import multicore_shortest_path
from .utils.multicore_nearest_edges import multicore_nearest_edge
from .utils.osm_data_request import *

class CityNetwork():
    """A CityNetwork class that contains osm and cbs information about
    a city. It can be used in combination with clustering an shortest
    path computations
    
    Developed by Job de Vogel and Jirri van den Bos
    """
    
    # Plot settings
    figsize=(8, 8) 
    bgcolor = 'white'
    edge_color = 'black'
    node_color = 'black'
    edge_linewidth = 0.5
    node_size=1.5
    route_color = 'black'
    route_width = 0.6
    origin_color = 'black'
    destination_color = 'black'
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
        self.ne_dist = None
        self.interpolation = None

    def __repr__(self):
        return "<CityNetwork object of {}>".format(self.name)

    def load_osm_graph(self, filepath: str, query=False):
        """Load an online or local osm graph

        If osm graph already locally available, graph will not  be retrieved
        online.
        
        Developed by Jirri van den Bos

        Parameters
        ----------
        filepath : string
            Path to where osm graph should be stored.

        query : string or False bool
            Indicate if graph should be loaded based on query.
            
        
        Related class object parameters
        ---------
        self.transport_type : string
            The type of transport (walk, bike, epv, all).
        
        self.graph : networkx graph
            Multigraph of the streets and junctions.
        """
        
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
        """Load buildings and adress data, combine into one pandas Dataframe

        If osm server is not able to send results, please try again in 30 seconds.
        
        Developed by Jirri van den Bos

        Parameters
        ----------
        building_addr_path : string
            Path where building adresses are saved

        building : string
            Path where buildings are saved

        adress_add_path : string
            Path where adresses are saved
            
        cbs_path : string
            Path where cbs data is saved
            
        
        Related class object parameters
        ---------
        self.building_addr_df : Pandas DataFrame
            Pandas DataFrame with data of all buildings, their adresses,
            locations and residents.
        """
        
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
                    raise RuntimeError("The request is currently unable to gather Overpass data, please retry manually in 30 seconds")
            
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

        CBS_data = read_CBS(cbs_path)

        building_addr_df= compare_building_cbs(building_addr_df, CBS_data, cbs_properties)

        # Save to CityNetwork object
        self.building_addr_df = building_addr_df


    def add_rel_attributes(self, overwrite_walk=None, overwrite_bike=None, overwrite_epv=False):
        """Add relevant attributes such as speed and travel_time to the graph
        
        Developed by Job de Vogel

        Parameters
        ----------
        overwrite_walk : None or int
            Speed which should be used for walking (kph)
        
        overwrite_bike : None or int
            Speed which should be used for bikes (kph)
        
        overwrite_epv : None or int
            Speed which should be used for electric powered vehicles (kph)

            
        Related class object parameters
        ---------
        self.transport_type : string
            The type of transport (walk, bike, epv, all).
        
        self.graph : networkx graph
            Multigraph of the streets and junctions.
        """
        
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
        """Add an experience weight to the graph, based on street names
        
        Developed by Job de Vogel

        Parameters
        ----------
        names : list of strings
            Names of the streets that should get experience values
        
        factors : list of floats
            Corresponding importance factors for the street experience values

            
        Related class object parameters
        ---------        
        self.graph : networkx graph
            Multigraph of the streets and junctions.
        """
        
        nx.set_edge_attributes(self.graph, 0, 'experience')

        for osmid in self.graph.edges:
            self.graph.edges[osmid]['experience'] = self.graph.edges[osmid]['length']

        for factor in factors:
            for osmid in self.graph.edges:
                if 'name' in self.graph.edges[osmid]:
                    if self.graph.edges[osmid]['name'] not in names:
                        self.graph.edges[osmid]['experience'] = self.graph.edges[osmid]['experience'] * factor


    def add_coord_experience(self, street_uvk=[], factors=[]):
        """Add an experience weight to the graph, based on coordinates
        
        Developed by Job de Vogel

        Parameters
        ----------
        street_uvk : list of tuples
            Coordinates that should get experience values
        
        factors : list of floats
            Corresponding importance factors for the coordinate experience values

            
        Related class object parameters
        ---------        
        self.graph : networkx graph
            Multigraph of the streets and junctions.
        """
        
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
        """Project graph from 'sphere'(globe) to 'surface'
        
        Developed by Job de Vogel

        WGS 84 -- WGS84 World Geodetic System 1984, used in GPS
        More info at https://gist.github.com/keum/7441007

        Parameters
        ----------
        names : list of strings
            Names of the streets that should get experience values
        
        factors : list of floats
            Corresponding importance factors for the street experience values

            
        Related class object parameters
        ---------        
        self.graph : networkx graph
            Multigraph of the streets and junctions.
        """
        self.graph = ox.project_graph(self.graph, to_crs="EPSG:3857")

   
    def convert_graph_edges_to_df(self):
        """Convert the edges of a graph to a Pandas Dataframe
        
        Developed by Job de Vogel
            
        Related class object parameters
        ---------      
        self.graph_edges_df : Pandas DataFrame
            DataFrame with edges of the graphs stored
        """
        if self.graph is not None:
            self.graph_edges_df =  ox.graph_to_gdfs(self.graph, nodes=False)
        else:
            raise ValueError('CityNetwork object does not contain graph.')

  

    def convert_graph_nodes_to_df(self):
        """Convert the nodes of a graph to a Pandas Dataframe
        
        Developed by Job de Vogel
            
        Related class object parameters
        ---------      
        self.graph_nodes_df : Pandas DataFrame
            DataFrame with nodes of the graphs stored
        """
        if self.graph is not None:
            self.graph_nodes_df = ox.graph_to_gdfs(self.graph, edges=False)
        else:
            raise ValueError('CityNetwork object does not contain graph.')


    # Get all node or edge attribute types included in data
    def get_edge_attribute_types(self):
        """Get the attributes from the edges in the graph
        
        Developed by Job de Vogel
            
        Related class object parameters
        ---------      
        self.graph_nodes_df : Pandas DataFrame
            DataFrame with nodes of the graphs stored
        """
        if self.graph_edges_df is None:
            raise ValueError('Dataframe of edges does not exist, please generate df with convert_graph_edges_to_df()')
        
        edge_attributes = self.graph_edges_df.columns
        return edge_attributes
    
    
    def get_node_attribute_types(self):
        """Get the attributes from the nodes in the graph
        
        Developed by Job de Vogel
            
        Related class object parameters
        ---------      
        self.graph_nodes_df : Pandas DataFrame
            DataFrame with nodes of the graphs stored
        """
        if self.graph_nodes_df is None:
            raise ValueError('Dataframe of nodes does not exist, please generate df with convert_graph_nodes_to_df()')
        
        node_attributes = self.graph_nodes_df.columns
        return node_attributes
    
    
    def get_node_osmids(self):
        """Get the osm id's from the nodes in the graph
        
        Developed by Job de Vogel
            
        Related class object parameters
        ---------      
        self.graph : networkx graph
            Multigraph of the streets and junctions.
        """
        if self.graph is None:
            raise ValueError('Graph does not exist, please generate graph with load_osm_graph()')
        return list(self.graph.nodes())
    
    
    def get_edge_osmids(self):
        """Get the osm id's from the edges in the graph
        
        Developed by Job de Vogel
            
        Related class object parameters
        ---------      
        self.graph : networkx graph
            Multigraph of the streets and junctions.
        """
        if self.graph is None:
            raise ValueError('Graph does not exist, please generate graph with load_osm_graph()')
        return self.graph_edges_df['osmid'].values


    def get_yx_destinations(self):
        """Get the yx coordinates of the destinations in the graphs
        from the building_addr_df DataFrame.
        
        Developed by Job de Vogel
            
        Related class object parameters
        ---------      
        self.building_addr_df : Pandas DataFrame
            Pandas DataFrame with data about the buildings and adresses
        """
        return list(self.building_addr_df.loc[:, ['y', 'x']].itertuples(index=False, name=None))


    def save_graph(self, name: str, folder: str):
        """Save the CityNetwork class object to a .pkl file
        
        Developed by Job de Vogel
        
        Parameters
        ----------
        name : string
            Name of the file to save
        
        folder : string
            Folder to save the file
        """
        
        object_name = name
        path = folder + str(object_name) + '.pkl'
        print('Saving {} to {}'.format(object_name, path))

        with open(path, 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


    def nearest_edges(self, interpolate, cpus=None, _multiply=1):
        """Compute the nearest edges to the buildings in the graph
        
        Developed by Job de Vogel
        
        Parameters
        ----------
        interpolate : float
            The distance at which edges should be interpolated
        
        cpus : int
            Number of cpu cores used
        
        _multiply : int
            Multiplication factor to repeat the computation
            

        Related class object parameters
        ---------      
        self.ne : list of tuples
            The nearest edges to destinations
        
        self.ne_dist : list of floats
            The distances to the nearest edges
        
        self.interpolation : list of tuples
            The interpolated vertices of the graph edges
        
        
        Returns
        ---------
        self.ne : list of tuples       
        """

        buildings_yx = list(self.building_addr_df.loc[:, ['y', 'x']].itertuples(index=False, name=None))
        y, x = list(map(list, zip(*buildings_yx)))

        if self.ne is None:
            # If _multiply is given, multiply the number of samples
            x *= _multiply
            y *= _multiply

            if len(x) != len(y):
                raise ValueError('Please make sure x and y have the same length.') 
            
            multicore_edge_result = multicore_nearest_edge(self.graph, x, y, interpolate, cpus=cpus)
            self.ne_dist, self.ne, self.interpolation = multicore_edge_result[0], multicore_edge_result[1], multicore_edge_result[2]
            return self.ne
        else:
            return self.ne
    
    
    def drop_outliers(self, min_dist):
        """Remove destinations further than min_dist meters to the nearest edge
        
        Developed by Job de Vogel
        
        Parameters
        ----------
        min_dist : float
            The minimum distance at which destination is removed


        Related class object parameters
        ---------
        self.building_addr_df : Pandas DataFrame
            Pandas DataFrame with data about buildings and adresses
        
        self.ne : list of tuples
            The nearest edges to destinations
        
        self.ne_dist : list of floats
            The distances to the nearest edges       
        """
        
        drop_outliers = []
        for i, dist in enumerate(self.ne_dist):
            if dist > min_dist:
                drop_outliers.append(i)
        
        self.building_addr_df = self.building_addr_df.drop(drop_outliers)

        self.ne = [edge for i, edge in enumerate(self.ne) if i not in drop_outliers]
        self.ne_dist = [dist for i, dist in enumerate(self.ne_dist) if i not in drop_outliers]

        print(f'Dropped {len(drop_outliers)} outliers further than {min_dist}m from edges')


    def _shortest_paths(self, orig, dest, orig_edge, dest_edge, weight='travel_time', method='dijkstra', return_path=False, cpus=None, _multiply=1):
        """DEPRECATED
        
        This method is replaced by the muticore single source shortest path
        algorithm, because of speed and efficiency.    
        """
        
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


    def plot(self, routes=None, origins=None, destinations=None, annotations=False, marks=None, route_color_mask=None, orig_color_mask=None, dest_color_mask=None, fig_name = None, dpi=100, save=False, show=False):
        """Plot, save and show one or multiple images
        
        Developed by Job de Vogel
        
        Parameters
        ----------
        routes : list or None
            Result from the multicore single source shortest path algorithm
        
        origins : list of tuples or None
            The origin locations of the shortest paths

        destinations : list of tuples or None
            The destinations locations of the shortest paths

        annotations : string or None
            Information to be added to streets. E.g. 'name', 'speed', 'experience'
            
        route_color_mask : list of strings or None
            Colors to be used for routes
        
        orig_color_mask : list of strings or None
            Colors to be used for origins
            
        dest_color_mask : list of strings or None
            Colors to be used for destinations
        
        fig_name : string or None
            Name of the figure

        dpi : int
            Pixels per inch
        
        save : bool
            Indicates if image should be saved
            
        show : bool
            Indicates if images should be shown
        

        Related class object parameters
        ---------
        self.graph : networkx graph
            Multigraph of the streets and junctions.
            
        All class variables for plotting 
        """
        
        if routes is not None:
            fig, ax = ox.plot_graph(self.graph, show=False, save=False, close=False,
                figsize = self.figsize,
                bgcolor = self.bgcolor,
                edge_color = self.edge_color,
                node_color = self.node_color,
                edge_linewidth = self.edge_linewidth,
                node_size = self.node_size)
            
            if route_color_mask == None:
                for route in routes:
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
            # Use color mask to assign colors to route
            else:
                if len(routes) != len(route_color_mask):
                    raise ValueError("Routes and route_color_mask should have same length.")

                for route, route_color in zip(routes, route_color_mask):
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
                    ax.plot(x, y, c=route_color, lw=self.route_width)
                    
                    if not isinstance(ls_orig, list):
                        x, y = zip(*ls_orig.coords)
                        ax.plot(x, y, c=route_color, lw=self.route_width)

                    if not isinstance(ls_dest, list):
                        x, y = zip(*ls_dest.coords)
                        ax.plot(x, y, c=route_color, lw=self.route_width)
        
        # Only print the map
        else:
            fig, ax = ox.plot_graph(self.graph, show=False, save=False, close=False,
            figsize = self.figsize,
            bgcolor = self.bgcolor,
            edge_color = self.edge_color,
            node_color = self.node_color,
            edge_linewidth = self.edge_linewidth,
            node_size = self.node_size)

        # Add an x to specific points
        if marks is not None:
            for mark in marks:
                ax.scatter(mark[1], mark[0],
                    color=self.marker_color, marker='x', s=100, label='Mark')

        # Plot the origins with an x mark
        if origins is not None:
            if orig_color_mask == None:
                for orig in origins:
                    ax.scatter(orig[1], orig[0], color=self.origin_color, marker='.', s=50, label='orig-point')
            # Use color mask for origins
            else:
                if len(origins) != len(orig_color_mask):
                    raise ValueError("Origins and orig_color_mask should have same length.")

                for orig, orig_color in zip(origins, orig_color_mask):
                    ax.scatter(orig[1], orig[0], color=orig_color, marker='.', s=50, label='orig-point')

        # Plot the destinations with an x mark
        if destinations is not None:
            if dest_color_mask == None:
                for dest in destinations:
                    ax.scatter(dest[1], dest[0], color=self.destination_color, marker='.', s=3, label='orig-point')
            # Use color mask for destinations
            else:
                if len(destinations) != len(dest_color_mask):
                    raise ValueError("Destinations and dest_color_mask should have same length.")

                for dest, dest_color in zip(destinations, dest_color_mask):
                    ax.scatter(dest[1], dest[0], color=dest_color, marker='.', s=3, label='orig-point')

        # Add annotations to the edges, can be names, travel_times etc.
        if annotations:
            graph = ox.get_undirected(self.graph)
            for _, edge in ox.graph_to_gdfs(graph, nodes=False).fillna('').iterrows():
                c = edge['geometry'].centroid
                text = edge[annotations]
                ax.annotate(text, (c.x, c.y), c=self.font_color, fontsize=self.font_size)

        fig, ax = ox.plot._save_and_show(fig, ax, show=show)

        if save:
            path = 'data/plot_pngs'
            
            if not os.path.isdir(path):
                os.mkdir(path)
            
            if fig_name == None:
                fig.savefig(f'data/plot_pngs/plot_{time.time()}.png', format='png', dpi=dpi)
            else:
                fig.savefig(f'data/plot_pngs/{fig_name}.png', format='png', dpi=dpi)

        return fig, ax

    # Loading the graph as a pickle file avoids having to recalculate
    # attributes such as speed, length etc. and is way faster
    # Call this function through var = CityNetwork.load_graph('var')
    @staticmethod
    def load_graph(name: str, folder: str):
        """Load the CityNetwork class object from a .pkl file
        
        Loading the graph as a .pkl file avoids having to recompute
        and load a graph. Attributes are already saved and can be
        reused. Call this function through var = CityNetwork.load_graph('var')
        
        Developed by Job de Vogel
        
        Parameters
        ----------
        name : string
            Name of the file to load
        
        folder : string
            Folder to load the file from
        """
        
        object_name = name
        path = folder + str(object_name) + '.pkl'

        print("Loading...")
        print('path', path)
        with open(path, 'rb') as file:
            print('file', file)
            graph = pickle.load(file)
            print('opened')
        
        print('Loaded {} to object...'.format(object_name))
        
        return graph