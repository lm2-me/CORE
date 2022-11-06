"""
functions related to plotting results

By: 
Job de Vogel, TU Delft 
Lisa-Marie Mueller, TU Delft 

Classes:
    - None

Functions:
    - 

"""

import matplotlib.pyplot as plt
import osmnx as ox
import time

from re import X

### visualize the hub clusters based on travel time
def visualize_clusters(City, Clusters, text, hub_colors, save=False):
    hub_colors_dict = dict(zip(Clusters.hub_list_dictionary.keys(), hub_colors))

    print('Plotting figure...')

    fig, ax = ox.plot_graph(City.graph, dpi=600, show=False, save=False, close=False,
        figsize = City.figsize,
        bgcolor = City.bgcolor,
        edge_color = City.edge_color,
        node_color = City.node_color,
        edge_linewidth = City.edge_linewidth,
        node_size = City.node_size)
    
    # add spots for the hubs
    for hub_name, hub_value in Clusters.hub_list_dictionary.items():
        color_to_use = hub_colors_dict[hub_name]
        current_label = hub_name
        #print(current_label, point, color_to_use)
        ax.scatter(hub_value['x'], hub_value['y'],
            color=color_to_use, marker='o', s=100, label=current_label)

    for i, row in Clusters.hub_assignments_df.T.items():
        if row['Nearest_hub_name'] == 'None':
            color_to_use = 'white'
        elif row['Path_not_found']:
            color_to_use = 'lime'
        else:
            print(row['Nearest_hub_name'])
            color_to_use = hub_colors_dict[row['Nearest_hub_name']]

        current_label = hub_name
        ax.scatter(City.building_addr_df.iloc[i]['x'], City.building_addr_df.iloc[i]['y'],
                    color=color_to_use, marker='o', s=5, label=current_label) 
    
    ax.set_title(text)
    plt.show()

    if save:
        fig.savefig(f'{City.data_folder}plot_pngs/'+ text +f'_{time.time()}.png')

    return fig, ax

### visualize the hub clusters based on euclidean distance
def euclid_visualize_clusters(City, Clusters, text, hub_colors, save=False):
    
    hub_colors_dict = dict(zip(Clusters.hub_list_dictionary.keys(), hub_colors))

    print('Plotting figure...')

    fig, ax = ox.plot_graph(City.graph, show=False, save=False, close=False,
        figsize = City.figsize,
        bgcolor = City.bgcolor,
        edge_color = City.edge_color,
        node_color = City.node_color,
        edge_linewidth = City.edge_linewidth,
        node_size = City.node_size)
    
    # add spots for the hubs
    for hub_name, hub_value in Clusters.hub_list_dictionary.items():
        color_to_use = hub_colors_dict[hub_name]
        current_label = hub_name
        #print(current_label, point, color_to_use)
        ax.scatter(hub_value['x'], hub_value['y'],
            color=color_to_use, marker='o', s=100, label=current_label)

    for i, row in Clusters.hub_assignments_df.T.items():
        color_to_use = hub_colors_dict[row['Euclid_nearesthub']]
        current_label = hub_name
        ax.scatter(City.building_addr_df.iloc[i]['x'], City.building_addr_df.iloc[i]['y'],
                    color=color_to_use, marker='o', s=5, label=current_label) 
    
    ax.set_title(text)
    plt.show()

    if save:
        fig.savefig(f'{City.data_folder}plot_pngs/'+ text +f'{time.time()}.png')

    return fig, ax