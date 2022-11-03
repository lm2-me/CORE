# from network_delft import CityNetwork
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from unpack.city_network import CityNetwork
from unpack.utils.multicore_nearest_edges import multicore_nearest_edge

def main():
    data_folder = 'data/'
    Delft = CityNetwork.load_graph('Delft_center_walk', data_folder)

    orig_yx_transf = (6803701, 484081)

    '''
    It is recommended to combine all origins and destinations
    into one list so that you do not need to interpolate multiple
    times. Afterwards you can split the list halfway to get back
    the origins and destinations again.

    multicore_nearest_edge inputs:
        - graph: the osmnx graph
        - X: The X value from a transformed coordinate, can be a list or float
        - Y: The Y value from a transformed coordinate, can be a list or float
        - interpolate: distance in meters to interpolate edges
        - cpus=1: the amount of cpu cores used
    '''
    num = 1000
    multicore_nearest_edge(Delft.graph, [orig_yx_transf[1]] * num, [orig_yx_transf[0]] * num, 10, cpus=None)

if __name__ == '__main__':
    main()