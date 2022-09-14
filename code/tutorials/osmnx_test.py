from typing import Type
import osmnx as ox
import networkx as nx
import geopandas
import os.path

Delft = [52.03, 51.96, 4.4, 4.3]

# Load an osm graph from online or local saved file
# Query indicates geocodable location such as 'Delft'
def load_osm_graph(coordinates, filepath, query=False):
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
            graph = ox.graph_from_bbox(*coordinates, simplify=True, retain_all=False, network_type='all', clean_periphery=True)
        
        print("Saving...")
        ox.save_graphml(graph, filepath=filepath)

    print("Finished loading")
    return graph

# Add speed and travel time to graph
def add_speed_travel_time(graph):
    graph = ox.speed.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)

    return graph

# Convert graph of edges or nodes to dataframe
# Returns a pandas dataframe
# Specific columns and rows can be accessed by using the standard
# pandas method, such as graph.loc[:, 'x'] or graph.loc[:, ['x','y']]
def convert_graph_edges_to_df(graph):
    return ox.graph_to_gdfs(graph, nodes=False)

def convert_graph_nodes_to_df(graph):
    return ox.graph_to_gdfs(graph, edges=False)

# Get all node or edge attributes included in data
def get_edge_attribute_types(graph):
    edge_attributes = ox.graph_to_gdfs(graph, nodes=False).columns
    return edge_attributes

def get_node_attribute_types(graph):
    node_attributes = ox.graph_to_gdfs(graph, edges=False).columns
    return node_attributes

# Access all node osmid's from graph
def get_node_osmids_graph(graph):
    return list(graph.nodes())

# Access all node osmid's from pandas dataframe
def get_node_osmids_df(graph_df):
    if not isinstance(graph_df, geopandas.geodataframe.GeoDataFrame):
        raise TypeError("Please input a graph dataframe using the get_graph_..._df function")

    return graph_df.index

def main():
    graph = load_osm_graph(Delft, 'data/Delft.osm')

    node_osmids = get_node_osmids_graph(graph)
    
    orig = node_osmids[0]
    dest = node_osmids[1]

    print("Running...")
    route = ox.distance.shortest_path(graph, orig, dest, weight='travel_time', cpus=None)

    # Plot multiple routes with ox.plot_graph_routes()
    ox.plot_graph_route(graph, route, route_color='g')

if __name__ == '__main__':
    main()