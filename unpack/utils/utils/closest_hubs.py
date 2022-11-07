import numpy as np

def closest_hubs(paths_dict):
    """ Compute the closest hub to a destination

    Developed by Job de Vogel
    
    Parameters
    ----------
    paths_dict : dict
        Result from the multicore_single_source_shortest_path
        computation.
    
    Returns
    -------
    closest_hub_idx : list
        List of indices indicating the closest hub to each destination.
    
    assigned_demand_points : list
        List indicating if a destination is connected to a hub or not.
    """


    # List concatination to extract the weights for all paths for each hub
    path_weights = np.array([[data[0] for data in hub] for hub in paths_dict.values()])
    
    # Calculate lowest weight and closest hub
    lowest_weight = np.amin(path_weights.T, axis=1)
    closest_hub_idx = np.argmin(path_weights.T, axis=1)
    
    # Convert to Python list to add None values
    closest_hub_idx = closest_hub_idx.tolist()

    # Store demand points that are assigned as coordinate
    assigned_demand_points = []

    # Set idx with weight inf to None
    for i, weight in enumerate(lowest_weight):
        if weight == float('inf'):
            closest_hub_idx[i] = None
        else:
            assigned_demand_points.append(i)

    closest_hub_idx = np.array(closest_hub_idx)

    return closest_hub_idx, assigned_demand_points