import numpy as np
import overpy
import pandas as pd
from pyproj import Transformer
import requests
import geopandas as gpd
from utils.multicore_shortest_path import transform_coordinates
from shapely.geometry import Point
from shapely.geometry import Polygon

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
def get_addr_query(user_input):
    areas = user_input[0]
    BBox = user_input[1]
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
def get_building_query(user_input):
    areas = user_input[0]
    BBox = user_input[1]
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
    print(max_length[3:] + max_length)
    #Add the building Lat, Lon back to the array
    building_addr = np.concatenate((building_array, building_addr), axis=1)

    #Remove all buildings that contain 0 addresses
    mask = (building_addr[:, 2] != 0)
    building_addr = building_addr[mask, :]

    #Exporting to a dataframe
    building_addr = pd.DataFrame(building_addr, columns=['latitude', 'longitude', 'addr', 'amenity'])

    return building_addr

#Function to add EPSG:3857 X,Y coordinates to the building_addr dataframe
def addxy_building_addr(building_addr):
    #Creating and transforming tuples of lat,lon
    coordinates = building_addr.loc[:, ['latitude', 'longitude']]
    origins = list(coordinates.itertuples(index=False, name=None))
    orig_yx_transf = transform_coordinates(origins)

    #Adding the transformed tuples to the dataframe
    building_addr["x"] = np.nan
    building_addr["y"] = np.nan
    for c,i in enumerate(orig_yx_transf):
        building_addr.loc[c,['x']] = i[1]
        building_addr.loc[c,['y']] = i[0]
    print('Finished adding EPSG:3857 coordinates')
    return building_addr

#Function to add CBS Buurt data to the building_addr dataframe
def compare_building_cbs(building_addr, cbs_data, cbs_properties):
    print('Assigning buildings to CBS buurt')
    #Creating shapely points of X,Y coordinates
    coordinates = building_addr.loc[:, ['x', 'y']]
    coordinates = list(coordinates.itertuples(index=False, name=None))
    points = []
    for i in coordinates:
        points.append(Point(i[0], i[1]))

    #Getting Buurt Polygon from CBS
    buurt_outlines = cbs_data.geometry
    
    #Check in which Buurt a building is located
    print('Checking if a building is inside a buurt')
    hold = np.empty(len(buurt_outlines))
    result = np.empty(len(points))
    max_length = ' / ' + str(len(points))
    for c, pt in enumerate(points):
        print(str(c) + max_length, end='\r')
        for d, poly in enumerate(buurt_outlines):
            hold[d] = poly.contains(pt)
        result[c] = np.asarray(hold==1).nonzero()[0]
    print(max_length[3:] + max_length)

    #Remove geom column and add empty columns for new data
    cbs_properties.remove('geom')
    for i in cbs_properties:
        building_addr[i] = np.nan

    #JOB CHECK THIS MAINLY
    #Adding the Buurt data to the dataframe
    print('Adding the CBS data to the dataframe')
    max_length = ' / ' + str(len(building_addr))
    for i in range(0,len(building_addr)):
        print(str(i) + max_length, end='\r')
        for j in cbs_properties:
            data = cbs_data.loc[result[i],[j]]
            building_addr.at[i, j] = data[0]
    print(max_length[3:] + max_length)
    building_addr['people'] = building_addr['addr'] * building_addr['gemiddeldeHuishoudsgrootte']
    return building_addr

#Optimized closest point comparison with numpy vector math
def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

#Load a CSV File
def load_csv(file_path):
    frame = pd.read_csv(file_path)
    return frame

def get_CBS_query(user_input, cbs_properties, buurt_index_skip=[]):
    #Example of a CBS Query:
    #https://service.pdok.nl/cbs/wijkenbuurten/2021/wfs/v1_0?service=wfs&version=2.0.0&srsName=EPSG:3857&request=GetFeature&typeName=cbs_buurten_2021&propertyName=(wijkenbuurten:geom,wijkenbuurten:buurtcode,wijkenbuurten:buurtnaam,wijkenbuurten:aantalHuishoudens)&bbox=80847.89481955457,443393.6485061926,86835.79389681925,449094.39207776875
    
    #Split user input
    areas = user_input[0]
    BBox = user_input[1]
    database = "cbs_buurten_2021"
    
    #Add base and typename requests to the query
    query = "https://service.pdok.nl/cbs/wijkenbuurten/2021/wfs/v1_0?service=wfs&version=2.0.0&srsName=EPSG:3857&request=GetFeature&typeName="
    query += database + '&propertyName=('
    for c, property in enumerate(cbs_properties):
        query += ('wijkenbuurten:'+ property)
        if c+1 != len(cbs_properties):
            query += ','

    #If placenames are defined and a bounding box is not
    if BBox == ['']:
        buurten = []
        #Establish a test directory to test for query results
        file_path = 'Data/runtime/test.xml'
        for c, i in enumerate(areas):
            #First run as if every input is a municipality
            query_current = query + ')&filter=<ogc:Filter><ogc:PropertyIsEqualTo><ogc:PropertyName>gemeentenaam</ogc:PropertyName><ogc:Literal>'+i+'</ogc:Literal></ogc:PropertyIsEqualTo></ogc:Filter>'
            get_CBS_data(query_current, file_path)
            cbs_data = read_CBS(file_path)
            #If a placename is not found check if it does exist as a buurtname
            if len(cbs_data.buurtcode) == 0:
                print(i + ' was not found when looking for municipalities, looking for '+i+ ' in buurten')
                query_current = query + ')&filter=<ogc:Filter><ogc:PropertyIsEqualTo><ogc:PropertyName>buurtnaam</ogc:PropertyName><ogc:Literal>'+i+'</ogc:Literal></ogc:PropertyIsEqualTo></ogc:Filter>'
                get_CBS_data(query_current, file_path)
                cbs_data = read_CBS(file_path)
                
                #Add results of buurtsearch to list, change area input to 'buurt'
                buurten_hold = cbs_data['buurtcode']
                areas[c] = 'buurt'
                #If no result is found as buurt: pass
                if len(cbs_data) == 0:
                    print(i+ ' will be skipped because it could not be found as a municipality or buurt')
                    areas.pop(c)
                    pass
                #If multiple results are found give the option to exclude choices (e.g. Den Hoorn on Texel)
                elif len(cbs_data)>1:
                    selection = list(range(0,len(cbs_data)))
                    if buurt_index_skip == []:
                        print(i+' was found multiple times:')
                        print(cbs_data[['buurtcode', 'buurtnaam', 'gemeentenaam']])
                        select_input = input('Optional: input index exlusions (comma seperated)')
                        if select_input != '':
                            exclusion = [int(x) for x in select_input.split(',')]
                            selection = list(set(selection) - set(exclusion))
                    #Option is given to provide a exclusion indices to skip input
                    else:
                        print(i+' was found multiple times as a buurt, exclusion skip applied')
                        selection = list(set(selection) - set(buurt_index_skip))
                    #Add the selected buurten to a list
                    for j in selection:
                        buurten.append(buurten_hold[j])
                else:
                    print(i+' was found once as a buurt' )
                    buurten.append(buurten_hold[0])
        #After establishing which municipalities and buurten are required, the rest of the query is finished
        query += ')&filter=<ogc:Filter>'
        #If multiple filters are to be added and 'Or' statement is requiredt
        if len(areas)>1:
            query += '<ogc:Or>'
            for i in areas:
                if i != 'buurt':
                    query += '<ogc:PropertyIsEqualTo><ogc:PropertyName>gemeentenaam</ogc:PropertyName><ogc:Literal>'+i+'</ogc:Literal></ogc:PropertyIsEqualTo>'
            for i in buurten:
                    query += '<ogc:PropertyIsEqualTo><ogc:PropertyName>buurtcode</ogc:PropertyName><ogc:Literal>'+i+'</ogc:Literal></ogc:PropertyIsEqualTo>'
            query += '</ogc:Or>'
        else:
            if areas[0] != 'buurt':
                query += '<ogc:PropertyIsEqualTo><ogc:PropertyName>gemeentenaam</ogc:PropertyName><ogc:Literal>'+areas[0]+'</ogc:Literal></ogc:PropertyIsEqualTo>'
            for i in buurten:
                query += '<ogc:PropertyIsEqualTo><ogc:PropertyName>buurtcode</ogc:PropertyName><ogc:Literal>'+i+'</ogc:Literal></ogc:PropertyIsEqualTo>'
        query += '</ogc:Filter>'
        return(query)

    #If a BBOX is provided, use it
    else:
        #Transform and reorder the bbox to be CBS compatible
        BBox_trans = [0,0,0,0]
        for i in range(0,2):
            coord = transform_coordinates((BBox[i],BBox[i+2]), from_crs='epsg:4326', to_crs='epsg:28992')
            BBox_trans[i] = coord[0]
            BBox_trans[i+2] = coord[1]
        new_order = [3,1,2,0]
        BBox = reorder_BBox(BBox_trans, new_order)

        #Build the rest of the query by adding the BBox
        query += (')&bbox=' +str(BBox).strip("[]"))

        return query

def get_CBS_data(query, path):
    response = requests.get(query)
    with open(path, 'wb') as file:
        file.write(response.content)

def read_CBS(path):
    data = gpd.read_file(path)
    return data
    

def main():
    #BBOX IN DEGREES, ORDERED: [N, S, E, W]
    #Old_BBox = [52.03, 51.96, 4.4, 4.3]
    #BBox = [52.026, 51.974, 4.394, 4.308]
    BBox = [52.018347, 52.005217, 4.369142, 4.350504]
    url = [None, "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://lz4.overpass-api.de/api/interpreter"]

    #GET USER INPUT
    #user_input = get_input()
    user_input = ([''], BBox)
    #user_input = (['Delft', 'Den Hoorn'], [''])

    cbs_properties = ['geom','gemiddeldeHuishoudsgrootte','buurtcode','buurtnaam','gemeentenaam']

    
    #GET OVERPASS QUERIES
    # addr_query = get_addr_query(user_input)
    # building_query = get_building_query(user_input)

    #GET BUILDING & ADDR FROM OVERPASS
    #Error catching for overpass server load too high, this can happen decently frequently for the standard servers. Retrying afterwards with stronger public servers. 
    #These however can return a vague error in http: "TypeError: '<=' not supported between instances of 'str' and 'int'". Reason unclear, happens infrequently and seems temporary. Retrying is the best solution for now
    # error = 0
    # for i in url:
    #     try:
    #         building_frame = get_osm_building(building_query, url=i)
    #         addr_frame = get_osm_addr(addr_query, url=i)
    #         ("Overpass Query succesfully completed")
    #         break
    #     except overpy.exception.OverpassGatewayTimeout as exc:
    #         print(exc)
    #         error += 1
    #         print("Overpass Server Load too high for standard servers, retrying with different url")
    #         pass
    #     except TypeError as exc:
    #         print(exc)
    #         error += 1
    #         print("Trying non standard server did not return a valid result, retrying with different server")
    #         pass
    # if error == len(url):
    #     print("The script is currently unable to gather Overpass data, please retry manually in 30 seconds")

    #LOAD THE BUILDING AND ADDR DATA FROM CSV
    addr_frame =  load_csv('data/Delft_center_walk_addresses.csv')
    building_frame = load_csv('data/Delft_center_walk_buildings.csv')

    building_addr = compare_building_addr(building_frame, addr_frame)
    building_addr = addxy_building_addr(building_addr)
    
    #building_addr = load_csv('data/building_addr.csv')

    CBS_query = get_CBS_query(user_input, cbs_properties, buurt_index_skip=[0])
    print(CBS_query)
    #get_CBS_data(CBS_query, 'data/runtime/CBS.xml')
    CBS_data = read_CBS('data/runtime/CBS.xml')

    test = compare_building_cbs(building_addr, CBS_data, cbs_properties)


if __name__ == '__main__':
    main()