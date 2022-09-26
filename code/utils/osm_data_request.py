from cmath import isnan
import numpy as np
import overpy
import pandas as pd
from owslib.wfs import WebFeatureService

'''
Retrieve OSM and BAG building data and adresses.

By: Jirri van den Bos, TU Delft
'''

def get_input():
    #Input the City or Area Name
    areas = [str(input("Please input one or multiple valid city or area names from the OSM database, comma seperated and case sensitive: "))]
    areas = [str(i).strip() for i in areas[0].split(',')]
    
    #Areas
    if areas != ['']:
        #Running a quick API check to see if input is valid
        api = overpy.Overpass(url="https://maps.mail.ru/osm/tools/overpass/api/interpreter")
        
        #API Calls for all inputs
        test_result_list = []
        for i in areas:
            test_query = '[out:xml][timeout:30];relation["name"="'+i+'"];out geom;'
            test_result = api.query(test_query)
            if len(test_result.relations) == 0:
                test_result_list.append(True)
            else:
                test_result_list.append(False)
        
        #Look up failed API calls and raise or pass
        fail_indices = np.where(test_result_list)[0]
        failed_results = []
        if len(fail_indices) != 0:
            for i in fail_indices:
                failed_results.append(areas[i])
            raise ValueError(str(failed_results) + " could not be found in the OSM database, mind case sensitivity and please refer to openstreetmap.org")
        else:
            print(str(areas) + "successfully found")

    #Giving an option to define the input as a BBox
    BBox = [input("Optional, provide a bounding box as degree lat&lon in format N,S,E,W: ")]
    
    #Check the validity of the BBox input passing if no input is provided
    if len(BBox[0]) != 0 and BBox[0] != 'n':
        try:
            BBox = [float(i) for i in BBox[0].split(',')]
        except ValueError:
            raise ValueError("BBox: " + str(BBox)+": input type is not valid")
        if len(BBox) != 4:
            raise ValueError("BBox: " + str(BBox)+": input amount is not valid")

        #Checking the BBox input order of N,S,E,W
        BBox_lat = BBox[0]-BBox[1]
        BBox_lon = BBox[2]-BBox[3]
        if BBox_lat < 0 or BBox_lon < 0:
            raise ValueError(str(BBox) + " is an invalid BBox, please check N,S,E,W order")
        
        #Checking the size of the BBox, raise if too large
        if BBox_lat > 0.2 or BBox_lon > 0.2:
            if input("The provided BBox is very large and could reach computational limits, do you want to continue (y/n)") == 'y':
                print("Continuing")
                pass
            else:
                print("Terminating")
                exit()

    #Final catch if no input is provided
    try:
        return(areas, BBox)
    except UnboundLocalError:
        raise UnboundLocalError("Neither area name or BBox was detected, terminating")

#Function to reorder a BBox
def reorder_BBox(BBox_in, new_order):
    BBox_out = []
    for i in new_order:
        BBox_out.append(BBox_in[i])
    return BBox_out

#Generate the OVERPASS API query for addresses 
def get_addr_query(input):
    areas = input[0]
    BBox = input[1]
    #If a BBox is provided, use it
    if BBox == ['']:
        query = '[out:xml][timeout:30];('
        for i in areas:
            query += 'area["name"="'+i+'"];'
        query += ')->.searchArea;(node["addr:housenumber"](area.searchArea););out;'
    #If no BBox is provided use the area name
    else:
        order = [1,3,0,2]
        BBox = reorder_BBox(BBox, order)
        query = '[out:xml][timeout:30];node["addr:housenumber"]('
        for i in BBox:
            query += str(i)
            if i != BBox[-1]:
                query += ', '
        query += ');out;'
    return query

#Generate the OVERPASS API query for buildings
def get_building_query(input):
    areas = input[0]
    BBox = input[1]
    #If a BBox is provided, use it
    if BBox == ['']:
        query = '[out:json][timeout:100];('
        for i in areas:
            query += 'area["name"="'+i+'"];'
        query += ')->.searchArea;(way["building"](area.searchArea););out center meta;'
    #If no BBox is provided use the area name
    else:
        order = [1,3,0,2]
        BBox = reorder_BBox(BBox, order)
        query = '[out:json][timeout:100];way["building"=yes]('
        for i in BBox:
            query += str(i)
            if i != BBox[-1]:
                query += ', '
        query += ');out center meta;'
    return query

#Use a OVERPASS API Query to call addresses
def get_osm_addr(query, url=None):
    #Running the API Call
    api = overpy.Overpass(url)
    print("Searching for adresses...")
    results = api.query(query)
    print("Addresses found:" + str(len(results.nodes)))

    #Addresses are provided as nodes, the lat and lon are retrieved specifically
    node_tags_list = []
    for node in results.nodes:
        node.tags['latitude'] = node.lat
        node.tags['longitude'] = node.lon
        node.tags['id'] = node.id
        node_tags_list.append(node.tags)
    nodes_frame = pd.DataFrame(node_tags_list)
    
    #Overpass will always provide full datasets, specific columns are selected
    addr_frame = nodes_frame[['latitude', 'longitude', 'amenity']]
    
    return addr_frame

#Use a OVERPASS API Query to call buildings
def get_osm_building(query, url=None):
    #Running the API Call
    api = overpy.Overpass(url)
    print("Searching for buildings...")
    results = api.query(query)
    print("Buildings found:" + str(len(results.ways)))

    #Buildings are provided as ways, when 'center' is added to the query it returns the center tag which is accessed here
    way_tags_list = []
    for way  in results.ways:
        way.tags['center_lat'] = way.center_lat
        way.tags['center_lon'] = way.center_lon
        way_tags_list.append(way.tags)
    ways_frame = pd.DataFrame(way_tags_list)
    
    #Overpass will always provide full datasets, specific columns are selected
    building_frame = ways_frame[['center_lat', 'center_lon']]
    
    return(building_frame)

#Running closest point comparisons between building centers and adresses
def compare_building_addr(building_frame:pd.DataFrame, addr_frame:pd.DataFrame):
    #Initialize arrays from dataframes
    building_array = building_frame[['center_lat', 'center_lon']].to_numpy()
    addr_array = addr_frame[['latitude', 'longitude', 'amenity']].to_numpy(na_value=None)
    
    #Initialize an array to fill
    building_addr = np.zeros((building_array.shape[0], 2), object)
    for i in building_addr:
        i[0] = int(0)
        i[1] = []
    
    #Display a simple progress amount
    max_length = ' / ' + str(len(addr_array))
    print("Assigning addresses and amenities to buildings:")

    #Count addresses per building and add addr amenities
    for c, i in enumerate(addr_array):
        print(str(c) + max_length, end='\r')
        coords = [i[0], i[1]]
        index = closest_node(coords, building_array)
        building_addr[index][0] += 1
        if addr_array[c][2] != None:
            building_addr[index][1].append(addr_array[c][2])
    
    #Add the building Lat, Lon back to the array
    building_addr = np.concatenate((building_array, building_addr), axis=1)

    #Remove all buildings that contain 0 addresses
    mask = (building_addr[:, 2] != 0)
    building_addr = building_addr[mask, :]

    building_addr = pd.DataFrame(building_addr, columns=['latitude', 'longitude', 'addr', 'amenity'])

    return building_addr

#Optimized closest point comparison with numpy vector magic
def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

#Load a CSV File
def load_csv(file_path):
    frame = pd.read_csv(file_path)
    return frame

#UNFINISHED
# def get_CBS():
#     CBS_buurt_wijk_url = "https://service.pdok.nl/cbs/wijkenbuurten/2021/wfs/v1_0?request=getcapabilities&service=wfs"
#     CBS_wfs = WebFeatureService(url=CBS_buurt_wijk_url)
#     print(list(CBS_wfs.contents))
#     print([operation.name for operation in CBS_wfs.operations])
#     test_cbs = CBS_wfs.get_schema('cbs_buurten_2021')
#     print(test_cbs)

#OLD
# def get_BAG(areas):
    
#     BAG_url = "https://service.pdok.nl/lv/bag/wfs/v2_0?request=getCapabilities&service=WFS"
#     BAG_wfs = WebFeatureService(url=BAG_url)
#     print(list(BAG_wfs.contents))
#     print([operation.name for operation in BAG_wfs.operations])
#     test_BAG = BAG_wfs.get_schema('pand')
#     test_pand = BAG_wfs.getfeature('pand', srsname='epsg:3857', outputFormat='xml', filter='Geometrygml:Polygongml:outerBoundaryIsgml:Linear' )
#     print(test_BAG)

def main():
    #BBOX IN DEGREES, ORDERED: [N, S, E, W]
    #Old_BBox = [52.03, 51.96, 4.4, 4.3]
    BBox = [52.026, 51.974, 4.394, 4.308]
    url = [None, "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://lz4.overpass-api.de/api/interpreter"]

    #GET USER INPUT
    #user_input = get_input()
    user_input = ([''], BBox)
    
    #GET OVERPASS QUERIES
    addr_query = get_addr_query(user_input)
    building_query = get_building_query(user_input)

    #GET BUILDING & ADDR FROM OVERPASS
    #Error catching for overpass server load too high, this can happen decently frequently for the standard servers. Retrying afterwards with stronger public servers. 
    #These however can return a vague error in http: "TypeError: '<=' not supported between instances of 'str' and 'int'". Reason unclear, happens infrequently and seems temporary. Retrying is the best solution for now
    error = 0
    for i in url:
        try:
            building_frame = get_osm_building(building_query, url=i)
            addr_frame = get_osm_addr(addr_query, url=i)
            ("Overpass Query succesfully completed")
            break
        except overpy.exception.OverpassGatewayTimeout as exc:
            print(exc)
            error += 1
            print("Overpass Server Load too high for standard servers, retrying with different url")
            pass
        except TypeError as exc:
            print(exc)
            error += 1
            print("Trying non standard server did not return a valid result, retrying with different server")
            pass
    if error == len(url):
        print("The script is currently unable to gather Overpass data, please retry manually in 30 seconds")

    #LOAD THE BUILDING AND ADDR DATA FROM CSV
    addr_frame =  load_csv('data/addr_saved.csv')
    building_frame = load_csv('data/building_saved.csv')

    building_addr = compare_building_addr(building_frame, addr_frame)

    #get_CBS()

if __name__ == '__main__':
    main()