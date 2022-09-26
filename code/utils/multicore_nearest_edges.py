import osmnx as ox
import numpy as np
import pandas as pd

import multiprocessing as mp

from scipy.spatial import cKDTree

import time
import tqdm

'''
This script was originally developed by gboing in OSMnx. It
has been upgraded by Job de Vogel to have the following
functionalities:

- nearest_edges is now splitted in finding nearest edge and
interpolation function.
- function now runs using multiple cpus.
'''

# Expects results in same coordinate system as graph.
# Make sure you first transform coordinates and project graph.

def _single_find_nearest_edge(X, Y, vertices, is_scalar):
    _, pos = cKDTree(vertices).query(np.array([X, Y]).T, k=1)
    ne = vertices.index[pos]

    return ne

def _interpolate_graph(G, X, Y, interpolate=10):
    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        # make coordinates arrays if user passed non-iterable values
        is_scalar = True
        X = np.array([X])
        Y = np.array([Y])

    if np.isnan(X).any() or np.isnan(Y).any():  # pragma: no cover
        raise ValueError("`X` and `Y` cannot contain nulls")
    geoms = ox.utils_graph.graph_to_gdfs(G, nodes=False)["geometry"]

    # interpolate points along edges to index with k-d tree or ball tree
    uvk_xy = list()
    for uvk, geom in zip(geoms.index, geoms.values):
        uvk_xy.extend((uvk, xy) for xy in ox.utils_geo.interpolate_points(geom, interpolate))
    labels, xy = zip(*uvk_xy)
    vertices = pd.DataFrame(xy, index=labels, columns=["x", "y"])

    return X, Y, vertices, is_scalar

def multicore_nearest_edge(graph, X, Y, interpolate, cpus=1):
    
    """
    Find the nearest edge to a point or to each of several points.

    If `X` and `Y` are single coordinate values, this will return the nearest
    edge to that point. If `X` and `Y` are lists of coordinate values, this
    will return the nearest edge to each point.

    `interpolate` search for the nearest edge to each point, one
    at a time, using an r-tree and minimizing the euclidean distances from the
    point to the possible matches. For accuracy, use a projected graph and
    points. This method is precise and also fastest if searching for few
    points relative to the graph's size.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        graph in which to find nearest edges
    X : float or list
        points' x (longitude) coordinates, in same CRS/units as graph and
        containing no nulls
    Y : float or list
        points' y (latitude) coordinates, in same CRS/units as graph and
        containing no nulls
    interpolate : float
        spacing distance between interpolated points, in same units as graph.
        smaller values generate more points.

    Returns
    -------
    ne or (ne, dist) : tuple or list
        nearest edges as (u, v, key) or optionally a tuple where `dist`
        contains distances between the points and their nearest edges
    """

    print(f"Interpolating {graph} for finding nearest edges...")

    start = time.time()
    X, Y, vertices, is_scalar = _interpolate_graph(graph, X, Y, interpolate=interpolate)
    end = time.time()
    print(f"Finished interpolating in {round(end-start, 2)}s...")

    start = time.time()
    # Figure out how many cpu cores are available
    if cpus is None:
        cpus = mp.cpu_count()
    cpus = min(cpus, mp.cpu_count())
    print(f"Finding {len(X)} nearest edges with {cpus} CPUs...")

    if cpus == 1:
        # if single-threading, calculate each shortest path one at a time
        # Return route_weight, route, partial_edge_1 and partial_edge_2
        result = [_single_find_nearest_edge(x, y, vertices, is_scalar) for x, y in zip(X, Y)]
    else:
        print("USER-WARNING: Make sure you put the multicore_nearest_edge function in a 'if __name__ == '__main__' statement!")
        # If multi-threading, calculate shortest paths in parallel           
        args = ((x, y, vertices, is_scalar) for x, y in zip(X, Y))
        pool = mp.Pool(cpus)

        # Add kwargs using partial method
        sma = pool.starmap_async(_single_find_nearest_edge, tqdm.tqdm(args, total=len(X)))
        result = sma.get()
        pool.close()
        pool.join()
    end = time.time()

    print(f'Found {len(X)} edges in {round(end-start, 2)}s with {cpus} CPUs...')
    return result