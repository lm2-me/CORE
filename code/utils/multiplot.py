import multiprocessing as mp
import tqdm

def format_paths_for_plot(paths, orig_yx, dest_yx, closest_hubs, assigned_houses, colors):
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
    closest_hubs, assigned_houses = closest_hub_func(cluster_iteration)

    cleaned_paths, destinations, color_mask, orig_color_mask = format_paths_for_plot(cluster_iteration, cluster_iteration.keys(), destinations, closest_hubs, assigned_houses, colors)

    CityNetwork.plot(routes=cleaned_paths, origins=cluster_iteration.keys(), destinations=destinations, route_color_mask=color_mask, orig_color_mask=orig_color_mask, dest_color_mask=color_mask, fig_name=f"{session_name}_{i}", dpi=dpi, save=True, show=False)

def multiplot_save(cluster_iterations, CityNetwork, destinations, closest_hub_func, colors, session_name, dpi=100, cpus=None):
    # Figure out how many cpu cores are available
    if cpus is None:
        cpus = mp.cpu_count()
    cpus = min(cpus, mp.cpu_count())
    print(f"Saving {len(cluster_iterations)} plots using {cpus} CPUs...")

    if cpus == 1:
        for i, iteration in enumerate(cluster_iterations):
            print(f"Plotting figure {i}...")
            closest_hubs, assigned_houses = closest_hub_func(iteration)

            cleaned_paths, destinations, color_mask, orig_color_mask = format_paths_for_plot(iteration, iteration.keys(), destinations, closest_hubs, assigned_houses, colors)

            CityNetwork.plot(routes=cleaned_paths, origins=iteration.keys(), destinations=destinations, route_color_mask=color_mask, orig_color_mask=orig_color_mask, dest_color_mask=color_mask, fig_name=f"{session_name}_plot_{i}", dpi=dpi, save=True, show=False)
    else:
        print("USER-WARNING: Make sure you put the multiplot function in a 'if __name__ == '__main__' statement!")
        # If multi-threading, calculate shortest paths in parallel
        args = ((iteration, i, CityNetwork, destinations, closest_hub_func, colors, session_name, dpi) for i, iteration in enumerate(cluster_iterations))
        pool = mp.Pool(cpus)

        sma = pool.starmap_async(_plot, tqdm.tqdm(args, total=len(cluster_iterations)))

        sma.get()
        pool.close()
        pool.join()