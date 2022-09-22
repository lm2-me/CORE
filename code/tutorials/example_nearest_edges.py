# from network_delft import CityNetwork
from network_delft import CityNetwork
from utils.multicore_nearest_edges import multicore_nearest_edge

'''
ImportError? Move this file to code folder TEMPORATILY
'''

def main():
    Delft = CityNetwork.load_graph('Delft')

    orig_yx_transf = (6803701, 484081)
    dest_yx_transf = (6794547, 483281)

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
    multicore_nearest_edge(Delft.graph, [orig_yx_transf[1]] * 1000, [orig_yx_transf[0]] * 1000, 10, cpus=None)

if __name__ == '__main__':
    main()