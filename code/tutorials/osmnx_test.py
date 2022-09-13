import osmnx as ox
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

def main():
    graph = load_osm_graph(Delft, 'data/Delft.osm')
    ox.plot_graph(graph)

if __name__ == '__main__':
    main()