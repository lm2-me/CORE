import networkx as nx

net = nx.DiGraph()
# nx.Graph()
# nx.DiGraph()
# nx.MultiGraph()
# nx.MultiDiGraph()

net.add_nodes_from([
    (4, {"color": "red", "name": "Job"}),
    (5, {"color": "green"}),
    (6, {"color": "blue"}),
    (7, {"color": "yellow"}),
])

net.add_edges_from([(4, 5), (5, 6), (6, 7), (7, 4)])

print(net.number_of_nodes())
print(net.number_of_edges())

# net.remove_node(4)

print(net.number_of_nodes())
print(net.number_of_edges())

# There are many types of graphs, we can experiment with different ones

# Draw graph
from matplotlib import pyplot as plt

# Other visualization:
options = {
    'node_color': 'black',
    'node_size': 100,
    'width': 1,
}
subax1 = plt.subplot(221)
nx.draw_random(net, **options)
subax2 = plt.subplot(222)
nx.draw_circular(net, **options)
subax3 = plt.subplot(223)
nx.draw_spectral(net, **options)

# Special graph
# All graphs can be found at https://networkx.org/documentation/networkx-1.10/reference/generators.html
G = nx.dodecahedral_graph()
subax4 = plt.subplot(224)
nx.draw(G, arrows=True, pos=nx.spring_layout(G))


# All visualization settings can be found at: https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html#networkx.drawing.nx_pylab.draw_networkx
# plt.savefig("data/graphs/path.png")
plt.show()



