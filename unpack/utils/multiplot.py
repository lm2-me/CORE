import multiprocessing as mp
import tqdm

"""
This code, developed by Job de Vogel, is able to plot and save
multiple images for multicore shortest paths. The main goal is
to minimize the time required for plotting images on large networks,
and can be used on multiple computers at the same time.

Note:
For single images, it is still recommended to use the CityNetwork.plot()
function, which is faster than multiplot for single images. For smaller
networks it may be useful to CityNetwork.plot() as well.

This function is highly memory bound. For example: generating 50 images,
on full city networks with dpi=300 can use up to 40 GB of committed memory.
"""

def format_paths_for_plot(paths, orig_yx, dest_yx, closest_hubs, assigned_houses, colors):
    """Format paths to be able to plot them, compute the color per house,
    figure out which houses are assigned.

    Args:
        paths : list
            Result of the multicore shortest paths computation (single source).
        
        orig_yx : list of tuples
            y, x coordinates of the origins.
        
        dest_yx : list of tuples
            y, x coordinates of the destinations.
        
        closest_hubs : numpy array
            Array of integers indicating the origins idx.
        
        assigned_houses : list of integers
            List of destination idxs assigned to any origin. 
        
        colors : list of strings
            Colors that should be used.

    Returns:
        cleaned_paths : list of paths
            List of paths that can be plotted, only paths that are
            actually used are inside list.
            
        destinations : list of tuples
            List containing all the destinations corresponding to
            the cleaned_paths.
            
        color_mask : list of strings
            Colors that should be used for the paths in same order
            as cleaned_paths.
        
        orig_color_mask : list of strings
            Colors that should be used for the origins.
    """
    destinations = []
    for i, dest in enumerate(dest_yx):
        if i in assigned_houses:
            destinations.append(dest)
    
    orig_color_mask = []
    for i in range(len(orig_yx)):
        orig_color_mask.append(colors[i % len(colors)])
    
    color_mask = [colors[i % len(colors)] for i in closest_hubs if i != None]
    cleaned_paths = [list(paths.values())[hub_idx][num] for num, hub_idx in enumerate(closest_hubs) if hub_idx != None]

    return cleaned_paths, destinations, color_mask, orig_color_mask

def _plot(cluster_iteration, i, CityNetwork, destinations, closest_hub_func, colors, session_name, dpi):
    """Plot and save one image with clusters and paths

    Args:
        cluster_iteration : list
            The result of one cluster iteration using the multicore single
            shortest path computation.
        
        i : int
            Iteration idx
        
        CitytNetwork : CityNetwork
            The CityNetwork object used.
        
        destinations : list of tuples
            All destinations in same order as in cluster_iteration.
        
        closest_hub_func : function
            Function that is used to compute closest hubs.
        
        colors : list of strings
            Colors that should be used.
        
        session_name : string
            Name of the current session.
        
        dpi = int
            Pixels per inch for the image
    
    Note:
        This functions does not actually show the plot. For that matter, use
        CityNetwork.plot(show=True)
    """
    
    closest_hubs, assigned_houses = closest_hub_func(cluster_iteration)

    cleaned_paths, destinations, color_mask, orig_color_mask = format_paths_for_plot(cluster_iteration, cluster_iteration.keys(), destinations, closest_hubs, assigned_houses, colors)

    CityNetwork.plot(routes=cleaned_paths, origins=cluster_iteration.keys(), destinations=destinations, route_color_mask=color_mask, orig_color_mask=orig_color_mask, dest_color_mask=color_mask, fig_name=f"{session_name}_{i}", dpi=dpi, save=True, show=False)

def multiplot_save(cluster_iterations, CityNetwork, destinations, closest_hub_func, colors, session_name, dpi=100, cpus=None, show=False):
    """Plot and save multiple images with clusters and paths

    Args:
        cluster_iteration : list
            The result of one cluster iteration using the multicore single
            shortest path computation.
        
        i : int
            Iteration idx
        
        CitytNetwork : CityNetwork
            The CityNetwork object used.
        
        destinations : list of tuples
            All destinations in same order as in cluster_iteration.
        
        closest_hub_func : function
            Function that is used to compute closest hubs.
        
        colors : list of strings
            Colors that should be used.
        
        session_name : string
            Name of the current session.
        
        dpi : int
            Pixels per inch for the image
            
        cpus : int
            Number of cpu cores used.
    
    Note:
        This functions does not actually show the plot. For that matter, use
        CityNetwork.plot(show=True)
    """
    
    # Figure out how many cpu cores are available
    if cpus is None:
        cpus = mp.cpu_count()
    cpus = min(cpus, mp.cpu_count())
    print(f"Saving {len(cluster_iterations)} plots using {cpus} CPUs...")

    if cpus == 1:
        if cluster_iterations is not None:
            for i, iteration in enumerate(cluster_iterations):
                print(f"Plotting figure {i}...")
                closest_hubs, assigned_houses = closest_hub_func(iteration)

                cleaned_paths, destinations, color_mask, orig_color_mask = format_paths_for_plot(iteration, iteration.keys(), destinations, closest_hubs, assigned_houses, colors)

                CityNetwork.plot(routes=cleaned_paths, origins=iteration.keys(), destinations=destinations, route_color_mask=color_mask, orig_color_mask=orig_color_mask, dest_color_mask=color_mask, fig_name=f"{session_name}_plot_{i}", dpi=dpi, save=True, show=False)
        else:
            CityNetwork.plot(routes=None, origins=iteration.keys(), destinations=destinations, route_color_mask=color_mask, orig_color_mask=orig_color_mask, dest_color_mask=color_mask, fig_name=f"{session_name}_plot_{i}", dpi=dpi, save=True, show=False)
    else:
        print("USER-WARNING: Make sure you put the multiplot function in a 'if __name__ == '__main__' statement!")
        # If multi-threading, calculate shortest paths in parallel
        if cluster_iterations is not None:
            args = ((iteration, i, CityNetwork, destinations, closest_hub_func, colors, session_name, dpi) for i, iteration in enumerate(cluster_iterations))
        else:
            args = ((None, i, CityNetwork, destinations, closest_hub_func, colors, session_name, dpi) for i, _ in enumerate(cluster_iterations))
        pool = mp.Pool(cpus)

        sma = pool.starmap_async(_plot, tqdm.tqdm(args, total=len(cluster_iterations)))

        sma.get()
        pool.close()
        pool.join()