from pyproj import Transformer

def transform_coordinates(coordinate: tuple or list, from_crs="epsg:4326", to_crs="epsg:3857"):
    """ Transform coordinates from 'sphere'(globe) to 'surface'
    (map). Expects the coordinates in (latitude, longitude) e.g.
    (51.99274, 4.35108) or (y, x).

    Developed by Job de Vogel

    WGS 84 -- WGS84 World Geodetic System 1984, used in GPS
    More info at https://gist.github.com/keum/7441007

    Parameters
    ----------
    coordinate : tuple or list of tuples
        Coordinate in (latitude, longitude)

    from_crs : string
        Transform coordinates from crs
    
    to_crs : 
        Transform coordinates to crs

    Returns
    -------
    Coordinates in new geodesic projection.
    """
    
    # From epsg:4326 to epsg:3857
    transformer = Transformer.from_crs(from_crs, to_crs)
    
    if isinstance(coordinate, list): 
        result = []
        for coord in coordinate:
            lat, lon = coord
            x, y = transformer.transform(lat, lon)
            result.append((y, x))

            if lat < lon:
                print('WARNING: latitude and longitude probably in wrong order in tuple! (Netherlands)')
    elif isinstance(coordinate, tuple):
        lat, lon = coordinate

        x, y = transformer.transform(lat, lon)
        result = (y, x)

        if lat < lon:
            print('WARNING: latitude and longitude probably in wrong order in tuple! (Netherlands)')
    else:
        raise TypeError('Inputs should be Tuple or List')

    return result