import networkx as nx

from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import substring

from osmnx.distance import nearest_edges
from osmnx.distance import great_circle_vec
from osmnx.utils_graph import get_route_edge_attributes

import multiprocessing as mp
from functools import partial

from pyproj import Transformer

'''
    The core of this code is written by Nathan Rooy in the
    taxicab Python package. This code, written by Job de Vogel
    adds several functionalities:

    - The shortest paths can now be calculated using multicore
    processing. For this the code OSMnx by gboeing was implemented.
    - Different shortest path algorithms can now be used, such as
    astar, dijkstra, bellman-ford.
    - A specific weight can be used to calculate shortest paths,
    instead of only 'length', such as travel_time.
    - The nearest_edge calculation has been moved to another
    function and optimized.
'''

'''
PROBLEMS:
- Length not correctly calculated due to coordinate system
'''

# Transform coordinates from 'sphere'(globe) to 'surface (map)
# Expects coordinates in (lat, lon) e.g. (51.99274, 4.35108)
def transform_coordinates(coordinate: tuple or list, from_crs="epsg:4326", to_crs="epsg:3857"):
    # # WGS 84 -- WGS84 World Geodetic System 1984, used in GPS
    # # More info at https://gist.github.com/keum/7441007
    
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

# Expects results in same coordinate system as graph.
# Make sure you first transform coordinates and project graph.
def find_nearest_edge(graph, orig_yx, dest_yx):
    """
    Optimize using ox.distance.nearest_edges with interpolate, see documentation
    """
    # # # Find the closest edges to origin and destination
    orig_edge = nearest_edges(graph, orig_yx[1], orig_yx[0]) 
    dest_edge = nearest_edges(graph, dest_yx[1], dest_yx[0])

    return orig_edge, dest_edge

# Calculate the total weight of the path, including partial edges
def _compute_route_weight(graph, route, weight, ls_orig, ls_dest, orig_edge, dest_edge):
    '''
    Compute the total weight of route and partial edges 
    '''
    route_weight = get_route_edge_attributes(graph, route, weight)
    
    orig_edge_length = graph.edges[orig_edge]['length']
    dest_edge_length = graph.edges[dest_edge]['length']

    if type(ls_orig) == LineString:
        x, y = zip(*ls_orig.coords)
        
        """
        PROBLEM

        great_circle_vec should be calculated with latitudes and longitudes!
        """

        y, x = transform_coordinates((y,x), from_crs="epsg:3857", to_crs="epsg:4326")
        
        partial_orig_edge_length = 0
        for i in range(0, len(x)-1):
            partial_orig_edge_length += great_circle_vec(y[i], x[i], y[i+1], x[i+1])

        partial_orig_factor = partial_orig_edge_length / orig_edge_length
        partial_orig_edge_weight = partial_orig_factor * graph.edges[orig_edge][weight]

    if type(ls_dest) == LineString:
        x, y = zip(*ls_dest.coords)
        
        y, x = transform_coordinates((y,x), from_crs="epsg:3857", to_crs="epsg:4326")

        partial_dest_edge_length = 0
        for i in range(0, len(x)-1):
            partial_dest_edge_length += great_circle_vec(y[i], x[i], y[i+1], x[i+1])
        
        partial_dest_factor = partial_dest_edge_length / dest_edge_length
        partial_dest_edge_weight = partial_dest_factor * graph.edges[orig_edge][weight]
    
    total_route_weight = 0
    if route:
        total_route_weight += sum(route_weight)
    if ls_orig:
        total_route_weight += partial_orig_edge_weight
    if ls_dest:
        total_route_weight += partial_dest_edge_weight

    return total_route_weight

# My own implementation for shortest path
def _get_edge_geometry(G, edge):
    '''
    Retrieve the points that make up a given edge.
    
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
    '''
    
    if G.edges.get(edge, 0):
        if G.edges[edge].get('geometry', 0):
            return G.edges[edge]['geometry']
    
    if G.edges.get((edge[1], edge[0], 0), 0):
        if G.edges[(edge[1], edge[0], 0)].get('geometry', 0):
            return G.edges[(edge[1], edge[0], 0)]['geometry']

    return LineString([
        (G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']),
        (G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y'])])

# Single_shortest path (based on taxicab, but using a* algorithm)
def _single_shortest_path(G, orig_yx, dest_yx, orig_edge, dest_edge,
    method='dijkstra', 
    weight='length',
    return_path=False):

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
        if method == 'astar':
            nx_route = nx.astar_path(G, orig_edge[0], dest_edge[0], weight=weight)
        elif method == 'bellman-ford':
            nx_route = nx.shortest_path(G, orig_edge[0], dest_edge[0], weight=weight)
        elif method == 'dijkstra':
            # Fall back on dijkstra
            nx_route = nx.shortest_path(G, orig_edge[0], dest_edge[0], weight=weight)
        else:
            raise ValueError('Method does not exist')

        p_o, p_d = Point(orig_yx[::-1]), Point(dest_yx[::-1])
        orig_geo = _get_edge_geometry(G, orig_edge)
        dest_geo = _get_edge_geometry(G, dest_edge)

        orig_clip = orig_geo.project(p_o, normalized=True)
        dest_clip = dest_geo.project(p_d, normalized=True)

        orig_partial_edge_1 = substring(orig_geo, orig_clip, 1, normalized=True)
        orig_partial_edge_2 = substring(orig_geo, 0, orig_clip, normalized=True)
        dest_partial_edge_1 = substring(dest_geo, dest_clip, 1, normalized=True)
        dest_partial_edge_2 = substring(dest_geo, 0, dest_clip, normalized=True)

        # when the nx route is just a single node, this is a bit of an edge case
        if len(nx_route) <= 2:
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
            
        # when routing across two or more edges
        if len(nx_route) >= 3:
            # check overlap with first route edge
            route_orig_edge = _get_edge_geometry(G, (nx_route[0], nx_route[1], 0))
            if route_orig_edge.intersects(orig_partial_edge_1) and route_orig_edge.intersects(orig_partial_edge_2):
                nx_route = nx_route[1:]
        
            # determine which origin partial edge to use
            route_orig_edge = _get_edge_geometry(G, (nx_route[0], nx_route[1], 0)) 
            if route_orig_edge.intersects(orig_partial_edge_1):
                orig_partial_edge = orig_partial_edge_1
            else:
                orig_partial_edge = orig_partial_edge_2

            ### resolve destination

            # check overlap with last route edge
            route_dest_edge = _get_edge_geometry(G, (nx_route[-2], nx_route[-1], 0))
            if route_dest_edge.intersects(dest_partial_edge_1) and route_dest_edge.intersects(dest_partial_edge_2):
                nx_route = nx_route[:-1]

            # determine which destination partial edge to use
            route_dest_edge = _get_edge_geometry(G, (nx_route[-2], nx_route[-1], 0)) 
            if route_dest_edge.intersects(dest_partial_edge_1):
                dest_partial_edge = dest_partial_edge_1
            else:
                dest_partial_edge = dest_partial_edge_2
            
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

# Convert to multicore process based on osmnx method
# This code is based on the shortest_path method
# from OSMnx
def multicore_shortest_path(graph, orig, dest, orig_edge, dest_edge, method='dijkstra', weight='travel_time', return_path=False, cpus=1):
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
            print("WARNING: Make sure you put the multicore_shortest_path function in a 'if __name__ == '__main__' statement!")
            # If multi-threading, calculate shortest paths in parallel
            args = ((graph, o, d, ls_o, ls_d) for o, d, ls_o, ls_d in zip(orig, dest, orig_edge, dest_edge))
            pool = mp.Pool(cpus)

            # Add kwargs using partial method
            sma = pool.starmap_async(partial(_single_shortest_path, method=method, weight=weight, return_path=return_path), args)
            result = sma.get()
            pool.close()
            pool.join()
        
        return result
    else:  # pragma: no cover
        raise ValueError("Please check shortest path inputs.")