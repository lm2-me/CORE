import networkx as nx

from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import substring

from osmnx.distance import great_circle_vec
from osmnx.utils_graph import get_route_edge_attributes

import numpy as np
import pandas as pd
import osmnx as ox

import multiprocessing as mp
from functools import partial

from pyproj import Transformer

import tqdm
import time

'''
    This code, written by Job de Vogel, builds further on the
    work of Taxicab (Nathan Rooy) and OSMnx (Gboeing), improving
    the functionalities and efficiency of shortest path computations.
    
    The following functionalies have been added:

    >>> The shortest paths can now be calculated using multicore
    processing. For this the code OSMnx by gboeing was implemented.
    >>> Different shortest path algorithms can now be used, such as
    astar, dijkstra, bellman-ford.
    >>> A specific weight can be used to calculate shortest paths,
    instead of only 'length', such as 'travel_time' or 'experience'.
    >>> The nearest_edge calculation has been moved to another
    function and optimized.
    >>> The code of Taxicab contains multiple bugs related to exceptional
    cases for short routes. This package solves that issue.
    >>> Adds support for multicore single source Dijkstra calculations
    with cutoff values.
    >>> Convert the shortest path results to a Dataframe.
    >>> Multiple upgrades to improve speed performance.
'''

def transform_coordinates(coordinate: tuple or list, from_crs="epsg:4326", to_crs="epsg:3857"):
    """ Transform coordinates from 'sphere'(globe) to 'surface'
    (map). Expects the coordinates in (latitude, longitude) e.g.
    (51.99274, 4.35108) or (y, x).

    Developed by Job de Vogel

    WGS 84 -- WGS84 World Geodetic System 1984, used in GPS
    More info at https://gist.github.com/keum/7441007

    Parameters
    ----------
    coordinate : tuple or list of tuples
        Coordinate in (latitude, longitude)

    from_crs : string
        Transform coordinates from crs
    
    to_crs : 
        Transform coordinates to crs

    Returns
    -------
    Coordinates in new geodesic projection.
    """
    
    # From epsg:4326 to epsg:3857
    transformer = Transformer.from_crs(from_crs, to_crs)
    
    if isinstance(coordinate, list): 
        result = []
        for coord in coordinate:
            lat, lon = coord
            x, y = transformer.transform(lat, lon)
            result.append((y, x))

            if lat < lon:
                print('WARNING: latitude and longitude probably in wrong order in tuple! (Netherlands)')
    elif isinstance(coordinate, tuple):
        lat, lon = coordinate

        x, y = transformer.transform(lat, lon)
        result = (y, x)

        if lat < lon:
            print('WARNING: latitude and longitude probably in wrong order in tuple! (Netherlands)')
    else:
        raise TypeError('Inputs should be Tuple or List')

    return result

# Calculate the total weight of the path, including partial edges
def _compute_route_weight(graph, route, weight, ls_orig, ls_dest, orig_edge, dest_edge):
    """ Compute the total weight of route and partial edges.
    
    Developed by Nathan Rooy.

    Parameters
    ----------
    graph : OSMnx graph

    route : list of integers or empty list
        The route of node idxs of a route.
    
    weight : float
        The weight of the route.
    
    ls_orig: LineString or empty list
        LineString at the origin of the route.

    ls_dest: LineString or empty list
        LineString at the destination of the route.
    
    orig_edge: tuple
        Tuple indicating the nearest edge of the network
        to the origin.

    dest_edge: tuple
        Tuple indicating the nearest edge of the network
        to the destination.

    Returns
    -------
    The total weight of the route in the network.
    """
    route_weight = get_route_edge_attributes(graph, route, weight)
    
    orig_edge_length = graph.edges[orig_edge]['length']
    dest_edge_length = graph.edges[dest_edge]['length']

    if type(ls_orig) == LineString:
        x, y = zip(*ls_orig.coords)
        
        y, x = transform_coordinates((y,x), from_crs="epsg:3857", to_crs="epsg:4326")
        
        partial_orig_edge_length = 0
        for i in range(0, len(x)-1):
            partial_orig_edge_length += great_circle_vec(y[i], x[i], y[i+1], x[i+1])
        
        if orig_edge_length > 0:
            partial_orig_factor = partial_orig_edge_length / orig_edge_length
            partial_orig_edge_weight = partial_orig_factor * graph.edges[orig_edge][weight]
        else:
            partial_orig_edge_weight = 0
    else:
        partial_orig_edge_weight = 0

    if type(ls_dest) == LineString:
        x, y = zip(*ls_dest.coords)
        
        y, x = transform_coordinates((y,x), from_crs="epsg:3857", to_crs="epsg:4326")

        partial_dest_edge_length = 0
        for i in range(0, len(x)-1):
            partial_dest_edge_length += great_circle_vec(y[i], x[i], y[i+1], x[i+1])
        
        if dest_edge_length > 0:
            partial_dest_factor = partial_dest_edge_length / dest_edge_length
            partial_dest_edge_weight = partial_dest_factor * graph.edges[orig_edge][weight]
        else:
            partial_dest_edge_weight = 0
    else:
        partial_dest_edge_weight = 0
    
    total_route_weight = 0
    if route:
        total_route_weight += sum(route_weight)
    if ls_orig:
        total_route_weight += partial_orig_edge_weight
    if ls_dest:
        total_route_weight += partial_dest_edge_weight

    return total_route_weight

def _get_edge_geometry(G, edge):
    """ Retrieve the points that make up a given edge.

    Developed by Nathan Rooy
    
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    edge : tuple
        graph edge unique identifier as a tuple: (u, v, key)
    
    Returns
    -------
    list :
        ordered list of (lng, lat) points.
    
    Notes
    -----
    In the event that the 'geometry' key does not exist within the
    OSM graph for the edge in question, it is assumed then that 
    the current edge is just a straight line. This results in an
    automatic assignment of edge end points.
    """
    
    if G.edges.get(edge, 0):
        if G.edges[edge].get('geometry', 0):
            return G.edges[edge]['geometry']
    
    if G.edges.get((edge[1], edge[0], 0), 0):
        if G.edges[(edge[1], edge[0], 0)].get('geometry', 0):
            return G.edges[(edge[1], edge[0], 0)]['geometry']

    return LineString([
        (G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']),
        (G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y'])])

def _get_partial_edges(graph, nearest_edge, coordinate_yx):
    """ Compute partial edges from a nearest edge

    Developed by Job de Vogel, adapted from Taxicab (Nathan Rooy)
    
    Parameters
    ----------
    graph : OSMnx graph
    
    nearest_edge : tuple
        Nearest edge to coordinate_yx in graph
    
    coordinate_yx : tuple
        Coordinate to find partial edges from
    
    Returns
    -------
    partial_edge_1 : LineString
        First partial edge
    
            
    partial_edge_2 : LineString
        Second partial edge
    """
    
    point = Point(coordinate_yx[::-1])

    edge_geo = _get_edge_geometry(graph, nearest_edge)

    edge_clip = edge_geo.project(point, normalized=True)

    partial_edge_1 = substring(edge_geo, edge_clip, 1, normalized=True)
    partial_edge_2 = substring(edge_geo, 0, edge_clip, normalized=True)

    return (partial_edge_1, partial_edge_2)
    

def closest_hub(paths_dict):
    """ Compute the closest hub to a destination

    Developed by Job de Vogel
    
    Parameters
    ----------
    paths_dict : dict
        Result from the multicore_single_source_shortest_path
        computation.
    
    Returns
    -------
    closest_hub_idx : list
        List of indices indicating the closest hub to each destination.
    
    assigned_demand_points : list
        List indicating if a destination is connected to a hub or not.
    """


    # List concatination to extract the weights for all paths for each hub
    path_weights = np.array([[data[0] for data in hub] for hub in paths_dict.values()])
    
    # Calculate lowest weight and closest hub
    lowest_weight = np.amin(path_weights.T, axis=1)
    closest_hub_idx = np.argmin(path_weights.T, axis=1)
    
    # Convert to Python list to add None values
    closest_hub_idx = closest_hub_idx.tolist()

    # Store demand points that are assigned as coordinate
    assigned_demand_points = []

    # Set idx with weight inf to None
    for i, weight in enumerate(lowest_weight):
        if weight == float('inf'):
            closest_hub_idx[i] = None
        else:
            assigned_demand_points.append(i)

    closest_hub_idx = np.array(closest_hub_idx)

    return closest_hub_idx, assigned_demand_points

def _single_shortest_path(G, orig_yx, dest_yx, orig_edge, dest_edge,
    method='dijkstra', 
    weight='length',
    return_path=False):

    """ Compute a single shortest path between an origin and
    as destination, used for the multicore_shortest_path function.

    Developed by Nathan Rooy, adapted by Job de Vogel
    
    Parameters
    ----------
    G : OSMnx graph
        OSMnx graph of the CityNetwork
    
    orig_yx : tuple or list of tuples
        Tuple with the yx coordinate of the origin

    dest_yx : tuple or list of tuples
        Tuple with the yx coordinate of the destination

    orig_edge : tuple or list of tuples
        Tuple indicating the nearest edge to the origin

    dest_yx : tuple or list of tuples
        Tuple indicating the nearest edge to the destination
    
    method : string
        Method to use for shortest path, can be 'dijkstra',
        'bellman-ford' or 'astar'. For dijkstra, the bidirectional
        version of the algorithm is used.
    
    weight: string
        The weight to use to compute the shortest path.

    return_path: bool
        Indicate if the algorithm should return a list of nodes
        with the shortest path or not. If False, will return an
        None value.

    
    Returns : list
    -------
    route_weight : float
        The total weight of the shortest path 
    
    nx_route : list of integers
        The nodes in the graph used for the shortest path. If return_path
        is False returns a None value.
    
    orig_partial_edge : LineString
        The LineString at the origin of the path.
    
    dest_partial_edge
        The Linestring at the destination of the path.
    """

    # Check if on samen line, if so, use simplified calculation
    if orig_edge == dest_edge:
        # Revert x and y coordinates        
        p_o, p_d = Point(orig_yx[::-1]), Point(dest_yx[::-1])

        # Get edges from graph
        edge_geo = G.edges[orig_edge]['geometry']

        # Project the edges
        orig_clip = edge_geo.project(p_o, normalized=True)
        dest_clip = edge_geo.project(p_d, normalized=True)

        # Calculate the linelength between two points on a line
        orig_partial_edge = substring(edge_geo, orig_clip, dest_clip, normalized=True)  
        dest_partial_edge = []
        nx_route = []
    else:
        try:
            if method == 'astar':
                nx_route = nx.astar_path(G, orig_edge[0], dest_edge[0], weight=weight)
            elif method == 'bellman-ford':
                nx_route = nx.bellman_ford_path(G, orig_edge[0], dest_edge[0], weight=weight)
            elif method == 'dijkstra':
                nx_route = nx.shortest_path(G, orig_edge[0], dest_edge[0], weight=weight)
            else:
                raise ValueError('Method does not exist')
        except nx.NetworkXNoPath:
            nx_route = []
            print(f"USER-WARNING: Path between {orig_yx} and {dest_yx} not possible!")
            print(f"Route weight will be set to euclidian distance (multiplied by walking speed, for travel_time).")

        p_o, p_d = Point(orig_yx[::-1]), Point(dest_yx[::-1])
        
        orig_geo = _get_edge_geometry(G, orig_edge)
        dest_geo = _get_edge_geometry(G, dest_edge)
            

        orig_clip = orig_geo.project(p_o, normalized=True)
        dest_clip = dest_geo.project(p_d, normalized=True)

        orig_partial_edge_1 = substring(orig_geo, orig_clip, 1, normalized=True)
        orig_partial_edge_2 = substring(orig_geo, 0, orig_clip, normalized=True)
        dest_partial_edge_1 = substring(dest_geo, dest_clip, 1, normalized=True)
        dest_partial_edge_2 = substring(dest_geo, 0, dest_clip, normalized=True)

        # when there is no path available, edge case:
        if len(nx_route) == 0:
            orig_partial_edge = []
            dest_partial_edge = []
        # when the nx route is just a single node, this is a bit of an edge case
        elif len(nx_route) == 1:
            nx_route = []
            if orig_partial_edge_1.intersects(dest_partial_edge_1):
                orig_partial_edge = orig_partial_edge_1
                dest_partial_edge = dest_partial_edge_1
                
            if orig_partial_edge_1.intersects(dest_partial_edge_2):
                orig_partial_edge = orig_partial_edge_1
                dest_partial_edge = dest_partial_edge_2
                
            if orig_partial_edge_2.intersects(dest_partial_edge_1):
                orig_partial_edge = orig_partial_edge_2
                dest_partial_edge = dest_partial_edge_1
                
            if orig_partial_edge_2.intersects(dest_partial_edge_2):
                orig_partial_edge = orig_partial_edge_2
                dest_partial_edge = dest_partial_edge_2
        elif len(nx_route) == 2:
            route_edge = _get_edge_geometry(G, (nx_route[0], nx_route[1], 0))
            if route_edge.intersects(orig_partial_edge_1):
                orig_partial_edge = orig_partial_edge_1
            
            if route_edge.intersects(orig_partial_edge_2):
                orig_partial_edge = orig_partial_edge_2

            if route_edge.intersects(dest_partial_edge_1):
                dest_partial_edge = orig_partial_edge_1

            if route_edge.intersects(orig_partial_edge_2):
                dest_partial_edge = orig_partial_edge_2          
        # when routing across two or more edges
        elif len(nx_route) >= 3:
            origin_overlapping = False
            destination_overlapping = False
            
            # check overlap with first route edge
            route_orig_edge = _get_edge_geometry(G, (nx_route[0], nx_route[1], 0))
            if route_orig_edge.intersects(orig_partial_edge_1) and route_orig_edge.intersects(orig_partial_edge_2):
                origin_overlapping = True

            # determine which origin partial edge to use
            route_orig_edge = _get_edge_geometry(G, (nx_route[1], nx_route[2], 0)) 
            if route_orig_edge.intersects(orig_partial_edge_1):
                orig_partial_edge = orig_partial_edge_1
            else:
                orig_partial_edge = orig_partial_edge_2

            ### resolve destination
            # check overlap with last route edge
            route_dest_edge = _get_edge_geometry(G, (nx_route[-2], nx_route[-1], 0))
            if route_dest_edge.intersects(dest_partial_edge_1) and route_dest_edge.intersects(dest_partial_edge_2):
                destination_overlapping = True

            # determine which destination partial edge to use
            route_dest_edge = _get_edge_geometry(G, (nx_route[-3], nx_route[-2], 0))

            if route_dest_edge.intersects(dest_partial_edge_1):
                dest_partial_edge = dest_partial_edge_1
            else:
                dest_partial_edge = dest_partial_edge_2
            
            if origin_overlapping:
                nx_route = nx_route[1:]
            if destination_overlapping:
                nx_route = nx_route[:-1]
            
            if len(nx_route) == 1:
                nx_route = []
    
        # final check
        if orig_partial_edge:
            if len(orig_partial_edge.coords) <= 1:
                orig_partial_edge = []
        if dest_partial_edge:
            if len(dest_partial_edge.coords) <= 1:
                dest_partial_edge = []

    route_weight = _compute_route_weight(G, nx_route, weight, orig_partial_edge, dest_partial_edge, orig_edge, dest_edge)

    # If the nodes do not have to be returned: replace route by None, so that
    # parallel computation does not have to store the route in memory every path
    if not return_path:
        nx_route = None

    return route_weight, nx_route, orig_partial_edge, dest_partial_edge

def _single_source_shortest_path(graph, orig, dest, orig_edge, dest_edges, orig_partial_edges, dest_partial_edges, path_weights=None, method='dijkstra', weight='travel_time', cutoff=None):
    """ Compute all shortest paths from a certain origin, considering
    a cutoff value.

    Developed by Job de Vogel, used Taxicab by Nathan Rooy
    
    Parameters
    ----------
    graph : OSMnx graph
        OSMnx graph of the CityNetwork
    
    orig : tuple
        Tuple with the yx coordinate of the origin

    dest : list of tuples
        Tuple with the yx coordinate of the destination

    orig_edge : tuple or list of tuples
        Tuple indicating the nearest edge to the origin

    dest_edge : tuple or list of tuples
        Tuple indicating the nearest edge to the destinations

    orig_partial_edges : list of tuples of LineStrings (empty list if no LineString)
        Tuple with precomputed partial edges at intersection
        between nearest point on nearest edge and nearest edge.

    dest_partial_edges : list of tuples of LineStrings (empty list if no LineString)
        Tuple with precomputed partial edges at intersection
        between nearest point on nearest edge and nearest edge.

    path_weights : list of floats or None
        If path_weights is given, considering single core computation
        the algorithm will skip computations of shortest paths for
        destinations that already have a closer hub. This can drastically
        reduce the computation time, but comes with a cost of accuracy.
    
    method : string
        Method to use for shortest path, can be 'dijkstra',
        'bellman-ford'.
    
    weight: string
        The weight to use to compute the shortest path.

    cutoff : float
        The weight value at which the algorithm should stop searching
        for shortest paths to destinations.

    Returns : dict
        A dictionary with the hub yx coordinate as key and a list of path data
        as value.
    -------
    route_weight : float
        The total weight of the shortest path 
    
    nx_route : list of integers
        The nodes in the graph used for the shortest path. If return_path
        is False returns a None value.
    
    orig_partial_edge : LineString
        The LineString at the origin of the path.
    
    dest_partial_edge : LineString
        The Linestring at the destination of the path.
    """

    if method == 'dijkstra':
        nx_routes = nx.single_source_dijkstra(graph, orig_edge[0], weight=weight, cutoff=cutoff)
    elif method == 'bellman-ford':
        nx_routes = nx.single_source_bellman_ford(graph, orig_edge[0], weight=weight, cutoff=cutoff)
    else:
        raise ValueError('Method does not exist')

    # Extract the weights and routes of the paths
    route_weights, nx_routes = nx_routes

    result = []
    for i, (destination, dest_edge) in enumerate(zip(dest, dest_edges)):

        # Check if destination within cutoff range
        if dest_edge[0] in nx_routes.keys():

            # Path computations from orig to dest will be skipped if an
            # earlier weight value for dest was already lower than 
            # the weight computed by single source computation. This might
            # result in very rare occasions of not finding the closest hub
            # but almost the closest hub, but saves on computation time.
            if path_weights is not None:
                 if path_weights[i] < route_weights[dest_edge[0]]:
                    data = [float('inf'), [], [], []]
                    result.append(data)
                    continue

            # Check if on same line, if so, use simplified calculation
            if orig_edge == dest_edge:
                # Revert x and y coordinates        
                p_o, p_d = Point(orig[::-1]), Point(destination[::-1])

                # Get edges from graph
                edge_geo = graph.edges[orig_edge]['geometry']

                # Project the edges
                orig_clip = edge_geo.project(p_o, normalized=True)
                dest_clip = edge_geo.project(p_d, normalized=True)

                # Calculate the linelength between two points on a line
                orig_partial_edge = substring(edge_geo, orig_clip, dest_clip, normalized=True)  
                dest_partial_edge = []
                nx_route = []
            else:
                try:
                    orig_partial_edge_1, orig_partial_edge_2 = orig_partial_edges
                except:
                    print(orig_partial_edges)
                dest_partial_edge_1, dest_partial_edge_2 = dest_partial_edges[i][0], dest_partial_edges[i][1]

                # Retrieve the right weigh based on the hub edge
                route_weight = route_weights[dest_edge[0]]

                # Retrieve the right nx_route based on the hub edge
                nx_route = nx_routes[dest_edge[0]]

                # When there is no path available, edge case:
                if len(nx_route) == 0:
                    orig_partial_edge = []
                    dest_partial_edge = []
                # When the nx route is just a single node, this is a bit of an edge case
                elif len(nx_route) == 1:
                    nx_route = []
                    if orig_partial_edge_1.intersects(dest_partial_edge_1):
                        orig_partial_edge = orig_partial_edge_1
                        dest_partial_edge = dest_partial_edge_1
                        
                    if orig_partial_edge_1.intersects(dest_partial_edge_2):
                        orig_partial_edge = orig_partial_edge_1
                        dest_partial_edge = dest_partial_edge_2
                        
                    if orig_partial_edge_2.intersects(dest_partial_edge_1):
                        orig_partial_edge = orig_partial_edge_2
                        dest_partial_edge = dest_partial_edge_1
                        
                    if orig_partial_edge_2.intersects(dest_partial_edge_2):
                        orig_partial_edge = orig_partial_edge_2
                        dest_partial_edge = dest_partial_edge_2
                elif len(nx_route) == 2:
                    route_edge = _get_edge_geometry(graph, (nx_route[0], nx_route[1], 0))
                    if route_edge.intersects(orig_partial_edge_1):
                        orig_partial_edge = orig_partial_edge_1
                    
                    if route_edge.intersects(orig_partial_edge_2):
                        orig_partial_edge = orig_partial_edge_2

                    if route_edge.intersects(dest_partial_edge_1):
                        dest_partial_edge = orig_partial_edge_1

                    if route_edge.intersects(orig_partial_edge_2):
                        dest_partial_edge = orig_partial_edge_2          
                # when routing across two or more edges
                elif len(nx_route) >= 3:
                    origin_overlapping = False
                    destination_overlapping = False
                    
                    # Check overlap with first route edge
                    route_orig_edge = _get_edge_geometry(graph, (nx_route[0], nx_route[1], 0))
                    if route_orig_edge.intersects(orig_partial_edge_1) and route_orig_edge.intersects(orig_partial_edge_2):
                        origin_overlapping = True

                    # determine which origin partial edge to use
                    route_orig_edge = _get_edge_geometry(graph, (nx_route[1], nx_route[2], 0)) 
                    if route_orig_edge.intersects(orig_partial_edge_1):
                        orig_partial_edge = orig_partial_edge_1
                    else:
                        orig_partial_edge = orig_partial_edge_2

                    # Resolve destination
                    # Check overlap with last route edge
                    route_dest_edge = _get_edge_geometry(graph, (nx_route[-2], nx_route[-1], 0))
                    if route_dest_edge.intersects(dest_partial_edge_1) and route_dest_edge.intersects(dest_partial_edge_2):
                        destination_overlapping = True

                    # determine which destination partial edge to use
                    route_dest_edge = _get_edge_geometry(graph, (nx_route[-3], nx_route[-2], 0))

                    if route_dest_edge.intersects(dest_partial_edge_1):
                        dest_partial_edge = dest_partial_edge_1
                    else:
                        dest_partial_edge = dest_partial_edge_2
                    
                    if origin_overlapping:
                        nx_route = nx_route[1:]
                    if destination_overlapping:
                        nx_route = nx_route[:-1]
                    
                    if len(nx_route) == 1:
                        nx_route = []
            
                # Final check
                if orig_partial_edge:
                    if len(orig_partial_edge.coords) <= 1:
                        orig_partial_edge = []
                if dest_partial_edge:
                    if len(dest_partial_edge.coords) <= 1:
                        dest_partial_edge = []

            route_weight = _compute_route_weight(graph, nx_route, weight, orig_partial_edge, dest_partial_edge, orig_edge, dest_edge)

            if path_weights is not None:
                path_weights[i] = route_weight

            data = [route_weight, nx_route, orig_partial_edge, dest_partial_edge]
            result.append(data)
        
        # If destination not within cutoff range
        else:
            data = [float('inf'), [], [], []]
            result.append(data)
    
    # Return all the paths from this origin
    if path_weights != None:
        return result, path_weights
    else:
        return result

def multicore_shortest_path(graph, orig, dest, orig_edge, dest_edge, method='dijkstra', weight='travel_time', return_path=False, cpus=1):
    """ Compute all shortest paths from a list of origins, to
    a list of destinations using multiple cores.

    Developed by Job de Vogel, used Taxicab by Nathan Rooy
    and OSMnx by Gboeing.
    
    Parameters
    ----------
    graph : OSMnx graph
        OSMnx graph of the CityNetwork
    
    orig : tuple or list of tuples
        Tuple with the yx coordinate of the origin

    dest : tuple or list of tuples
        Tuple with the yx coordinate of the destination

    orig_edge : tuple or list of tuples
        Tuple indicating the nearest edge to the origin

    dest_edge : tuple or list of tuples
        Tuple indicating the nearest edge to the destinations
    
    method : string
        Method to use for shortest path, can be 'dijkstra',
        'bellman-ford'.
    
    weight: string
        The weight to use to compute the shortest path.

    return_path: bool
        Indicate if the algorithm should return a list of nodes
        with the shortest path or not. If False, will return an
        None value.

    cpus: int
        The number of cpu cores used. If None, the algorithm
        uses the max number of cores available on device.


    Returns : list of lists
        Lists of shortest paths between origins and destinations.
    -------
    route_weight : float
        The total weight of the shortest path 
    
    nx_route : list of integers
        The nodes in the graph used for the shortest path. If return_path
        is False returns a None value.
    
    orig_partial_edge : LineString
        The LineString at the origin of the path.
    
    dest_partial_edge
        The Linestring at the destination of the path.
    """

    if isinstance(orig, tuple) or isinstance(dest, tuple):
        raise TypeError('Orig and dest should be lists, not tuples.')
    
    if not isinstance(orig_edge, list) or not isinstance(dest_edge, list):
        raise TypeError('Orig_edge and dest_edge should be lists, not tuples.')
    
    if not (hasattr(orig, "__iter__") or hasattr(dest, "__iter__")):
        # If neither orig nor dest is iterable, just return the shortest path
        return _single_shortest_path(graph, orig, dest, method=method, weight=weight, return_path=return_path)
    elif hasattr(orig, "__iter__") and hasattr(dest, "__iter__"):
        # If both orig and dest are iterables ensure they have same lengths
        if len(orig) != len(dest):  # pragma: no cover
            raise ValueError("Orig, dest, orig_edge and dest_edge must contain same number of elements")
        
        # Figure out how many cpu cores are available
        if cpus is None:
            cpus = mp.cpu_count()
        cpus = min(cpus, mp.cpu_count())
        print(f"Solving {max(len(orig), len(dest))} paths using {method} algorithm with {cpus} CPUs...")

        if cpus == 1:
            # if single-threading, calculate each shortest path one at a time
            # Return route_weight, route, partial_edge_1 and partial_edge_2
            result = [_single_shortest_path(graph, o, d, ls_o, ls_d, method=method, weight=weight, return_path=return_path) for o, d, ls_o, ls_d in zip(orig, dest, orig_edge, dest_edge)]
        else:
            print("USER-WARNING: Make sure you put the multicore_shortest_path function in a 'if __name__ == '__main__' statement!")
            # If multi-threading, calculate shortest paths in parallel
            args = ((graph, o, d, ls_o, ls_d) for o, d, ls_o, ls_d in zip(orig, dest, orig_edge, dest_edge))
            pool = mp.Pool(cpus)

            # Add kwargs using partial method
            sma = pool.starmap_async(partial(_single_shortest_path, method=method, weight=weight, return_path=return_path), tqdm.tqdm(args, total=len(orig)))

            result = sma.get()
            pool.close()
            pool.join()
        
        return result
    else:  # pragma: no cover
        raise ValueError("Please check shortest path inputs.")

def multicore_single_source_shortest_path(graph, orig, dest, dest_edges, skip_non_shortest=False, method='dijkstra', weight='travel_time', cutoff=None, cpus=1):
    """ Compute all shortest paths from multiple origins, considering
    a cutoff value and using multiple cores.

    Developed by Job de Vogel, used Taxicab by Nathan Rooy
    and OSMnx by Gboeing.
    
    Parameters
    ----------
    graph : OSMnx graph
        OSMnx graph of the CityNetwork
    
    orig : tuple
        Tuple with the yx coordinate of the origin

    dest : list of tuples
        Tuple with the yx coordinate of the destination

    dest_edges : list of tuples
        Tuple indicating the nearest edge to the destinations

    skip_none_shortest : bool
        Bool indicating if single core algorithm may skip
        shortest path computations skipping destinations
        that already have a shorter path. Can dramatically
        decrease computation time but also decrease accuracy.

    method : string
        Method to use for shortest path, can be 'dijkstra',
        'bellman-ford'.
    
    weight: string
        The weight to use to compute the shortest path.

    cutoff : float
        The weight value at which the algorithm should stop searching
        for shortest paths to destinations.

    cpus: int
        The number of cpu cores used. If None, the algorithm
        uses the max number of cores available on device.


    Returns : dict
        A dictionary with the hub yx coordinate as key and a list of path data
        as value.
    -------
    route_weight : float
        The total weight of the shortest path 
    
    nx_route : list of integers
        The nodes in the graph used for the shortest path. If return_path
        is False returns a None value.
    
    orig_partial_edge : LineString
        The LineString at the origin of the path.
    
    dest_partial_edge
        The Linestring at the destination of the path.
    """

    # Check the format of orig and change to list
    if isinstance(orig, tuple):
        orig = [orig]
    elif isinstance(orig, list):
        pass
    else:
        raise TypeError('Orig should be list, or single tuple.')

    # Check the format of dest and change to list
    if isinstance(dest, tuple):
        orig = [orig]
    elif isinstance(dest, list):
        pass
    else:
        raise TypeError('Dest should be list, or single tuple.')

    # Check the format of dest_edges and change to list   
    if isinstance(dest_edges, tuple):
        orig = [orig]
    elif isinstance(dest_edges, list):
        pass
    else:
        raise TypeError('Dest should be list, or single tuple.')

    y_orig, x_orig = list(map(list, zip(*orig)))
    orig_edges = ox.nearest_edges(graph, x_orig, y_orig)

    orig_partial_edges = []
    for coordinate, orig_edge in zip(orig, orig_edges):
        orig_partial_edges.append(
            _get_partial_edges(graph, orig_edge, coordinate)
            )

    dest_partial_edges = []
    for coordinate, dest_edge in zip(dest, dest_edges):
        dest_partial_edges.append(
            _get_partial_edges(graph, dest_edge, coordinate)
            )
    ########################################################################

    orig_paths = {}
    if cpus == 1:
        if skip_non_shortest:
            print(f"Solving {len(orig)} single sources using {method} algorithm with cutoff {cutoff} on weight '{weight}', skipping non-shortest paths using {cpus} CPUs...")

            path_weights = [float('inf')] * len(dest) 
            for i, (orig, orig_edge) in enumerate(zip(orig, orig_edges)):
                start = time.time()

                partial_orig = orig_partial_edges[i]
                paths, path_weights = _single_source_shortest_path(graph, orig, dest, orig_edge, dest_edges, partial_orig, dest_partial_edges, path_weights=path_weights, method=method, weight=weight, cutoff=cutoff)
                end = time.time()

                orig_paths[orig] = paths
                print(f"Finished hub {i + 1} in {round(end-start, 2)}s...")
        else:
            print(f"Solving {len(orig)} single sources using {method} algorithm with cutoff {cutoff} on weight '{weight}' using {cpus} CPUs...")

            for i, (orig, orig_edge) in enumerate(zip(orig, orig_edges)):
                partial_orig = orig_partial_edges[i]
                paths = _single_source_shortest_path(graph, orig, dest, orig_edge, dest_edges, partial_orig, dest_partial_edges, method=method, weight=weight, cutoff=cutoff)

                orig_paths[orig] = paths
    else:
        if cpus is None:
            cpus = mp.cpu_count()
        cpus = min(cpus, mp.cpu_count())

        if cpus > 1:
            print("USER-WARNING: Make sure you put the multicore_single_source_shortest_path function in a 'if __name__ == '__main__' statement!")

            if cutoff is not None:
                print("USER-WARNING: In some cases setting cpus>1 increases computation time when cutoff is a low value. Multicore processing boosts performance when cutoff=None.")

        print(f"Solving {len(orig)} single sources using {method} algorithm with cutoff {cutoff} on weight '{weight}' using {cpus} CPUs...")

        if cpus > len(orig):
            print("USER-WARNING: Number of origins to compute is lower than number of cpu cores used. It is recommended to set cpus=1 for better performance.")
        
        # If multi-threading, calculate shortest paths in parallel
        args = ((graph, o, dest, orig_edge, dest_edges, orig_partial_edges[i], dest_partial_edges) for i, (o, orig_edge) in enumerate(zip(orig, orig_edges)))
        pool = mp.Pool(cpus)

        # Add kwargs using partial method
        sma = pool.starmap_async(partial(_single_source_shortest_path, method=method, weight=weight, cutoff=cutoff), tqdm.tqdm(args, total=len(orig)))

        result = sma.get()
        pool.close()
        pool.join()
    
        for o, res in zip(orig, result):
            orig_paths[o] = res
        
    return orig_paths

def paths_to_dataframe(paths, hubs=None):
    """ Convert shortest path results to a Pandas Dataframe.

    Developed by Job de Vogel
    
    Parameters
    ----------
    paths : dict
        Paths result from single source shortest path computation
    hubs : list of tuples
        Optionally, adds x and y values of assigned hub to Dataframe
    
    Returns
    -------
    Dataframe :
        Dataframe with data of shortest path computation.
    """

    df = pd.DataFrame()

    closest_hubs, _ = closest_hub(paths)
    
    closest_paths = [list(paths.values())[hub_idx][num] if hub_idx != None else None for num, hub_idx in enumerate(closest_hubs)]

    df['Nearest_hub_idx'] = closest_hubs
    df['Nearest_hub_name'] = [str(f"hub_{i + 1}") if i != None else None for i in closest_hubs]
    df['Weight'] = [data[0] if data != None else None for data in closest_paths]
    df['Path_not_found'] = [True if hub == None else False for hub in closest_hubs]
    
    if hubs != None:
        df['hub_x'] = [hubs[i][1] if i != None else None for i in closest_hubs]
        df['hub_y'] = [hubs[i][0] if i != None else None for i in closest_hubs]

    df['Path'] = closest_paths
    
    return df