import osmnx as ox
#G = ox.graph_from_place('Delft', network_type='drive')
Delft = [52.03, 51.96, 4.4, 4.3]
G = ox.graph_from_bbox(*Delft, simplify=True, retain_all=False, network_type='all', clean_periphery=True)
#G = ox.graph.graph_from_place('Delft', network_type='drive')
ox.plot_graph(G)