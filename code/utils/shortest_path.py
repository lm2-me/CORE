import os
import osmnx as ox
import networkx as nx
import time
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import substring

from osmnx import load_graphml
from osmnx.distance import nearest_edges
from osmnx.distance import great_circle_vec
from osmnx.utils_graph import get_route_edge_attributes

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
    instead of only 'lenght', such as travel_time.
    - The nearest_edge calculation has been moved to another
    function and optimized.
'''

# Transform coordinates from 'sphere'(globe) to 'surface (map)
def transform_coordinates(lat, lon): 
    # From epsg:4326 to epsg:3857
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    x, y = transformer.transform(lat, lon)

    return (y, x)

def find_nearest_edge(graph, orig_yx, dest_yx):
    """
    Optimize using ox.distance.nearest_edges with interpolate, see documentation
    """
    # # WGS 84 -- WGS84 World Geodetic System 1984, used in GPS
    # # More info at https://gist.github.com/keum/7441007
    orig_yx = transform_coordinates(orig_yx[0], orig_yx[1])
    dest_yx = transform_coordinates(dest_yx[0], dest_yx[1])

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
        
        partial_orig_edge_length = 0
        for i in range(0, len(x)-1):
            partial_orig_edge_length += great_circle_vec(y[i], x[i], y[i+1], x[i+1])
        
        partial_orig_factor = partial_orig_edge_length / orig_edge_length
        partial_orig_edge_weight = partial_orig_factor * graph.edges[orig_edge][weight]
    
    if type(ls_dest) == LineString:
        x, y = zip(*ls_dest.coords)
        
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
        orig_partial_edge = None
        dest_partial_edge = None

    return route_weight, nx_route, orig_partial_edge_1, dest_partial_edge

# Convert to multicore process based on osmnx method
def multicore_shortest_path():
    pass

print('loading...')
path = os.path.abspath(__file__ + "/../../../data/Delft.osm")
graph = load_graphml(filepath=path)
graph = ox.speed.add_edge_speeds(graph)
graph = ox.distance.add_edge_lengths(graph)
graph = ox.speed.add_edge_travel_times(graph)

graph = ox.project_graph(graph, to_crs="EPSG:3857")

orig = (51.99274, 4.35108)
dest = (52.02184, 4.37690)

start = time.time()
orig_edge, dest_edge = find_nearest_edge(graph, orig, dest)
end = time.time()
print(end-start)

start = time.time()
# Returns the nodes of the graph, orig edge and dest edge
data = _single_shortest_path(graph, orig, dest, orig_edge, dest_edge, weight='travel_time', method = 'astar')
end = time.time()
print(end-start)

"""
To do:
    - Implement route_weight function \/
    - If nodes are not required, do not return them (better for parallel) \/
    - Make different function for nearest edges orig and dest \/
    - Make transform work with lists
    - Implement multicore
"""