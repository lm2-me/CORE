from ensurepip import version
from json import load
from multiprocessing.resource_sharer import stop
from platform import node
from re import A
import numpy as np
from multiprocessing.sharedctypes import Value
import overpy
import pandas as pd
from owslib.wfs import WebFeatureService

  
def get_input():
    #Input the City or Area Name
    areas = [str(input("Please input one or multiple valid city or area names from the OSM database, comma seperated and case sensitive: "))]
    areas = [str(i).strip() for i in areas[0].split(',')]
    
    #Areas
    if areas != ['']:
        #Running a quick API check to see if input is valid
        api = overpy.Overpass(url="https://maps.mail.ru/osm/tools/overpass/api/interpreter")
        
        test_result_list = []
        for i in areas:
            test_query = '[out:xml][timeout:30];relation["name"="'+i+'"];out geom;'
            test_result = api.query(test_query)
            if len(test_result.relations) == 0:
                test_result_list.append(True)
            else:
                test_result_list.append(False)
        
        fail_indices = np.where(test_result_list)[0]
        failed_results = []
        if len(fail_indices) != 0:
            for i in fail_indices:
                failed_results.append(areas[i])
            raise ValueError(str(failed_results) + " could not be found in the OSM database, mind case sensitivity and please refer to openstreetmap.org")
        else:
            print(str(areas) + "successfully found")

    #BBox 
    BBox = [input("Optional, provide a bounding box as degree lat&lon in format N,S,E,W: ")]
    if len(BBox[0]) != 0 and BBox[0] != 'n':
        try:
            BBox = [float(i) for i in BBox[0].split(',')]
        except ValueError:
            raise ValueError("BBox: " + str(BBox)+": input type is not valid")
        
        if len(BBox) != 4:
            raise ValueError("BBox: " + str(BBox)+": input amount is not valid")

        BBox_lat = BBox[0]-BBox[1]
        BBox_lon = BBox[2]-BBox[3]
        if BBox_lat < 0 or BBox_lon < 0:
            raise ValueError(str(BBox) + " is an invalid BBox, please check N,S,E,W order")
        if BBox_lat > 0.2 or BBox_lon > 0.2:
            if input("The provided BBox is very large and could reach computational limits, do you want to continue (y/n)") == 'y':
                print("Continuing")
                pass
            else:
                print("Terminating")
                exit()

    
    try:
        return(areas, BBox)
    except UnboundLocalError:
        raise UnboundLocalError("Neither area name or BBox was detected, terminating")

def reorder_BBox(BBox_in, new_order):
    BBox_out = []
    for i in new_order:
        BBox_out.append(BBox_in[i])
    return BBox_out

def get_addr_query(input):
    areas = input[0]
    BBox = input[1]
    if BBox == ['']:
        query = '[out:xml][timeout:30];('
        for i in areas:
            query += 'area["name"="'+i+'"];'
        query += ')->.searchArea;(node["addr:housenumber"](area.searchArea););out;'
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

def get_building_query(input):
    areas = input[0]
    BBox = input[1]
    if BBox == ['']:
        query = '[out:json][timeout:100];('
        for i in areas:
            query += 'area["name"="'+i+'"];'
        query += ')->.searchArea;(way["building"](area.searchArea););out center meta;'
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


def get_osm_addr(query):
    #Establishing the Overpass API server to use. Specific server chosen for load limits
    api = overpy.Overpass()
    print("Busy looking for adresses")
    results = api.query(query)
    print("Addresses found:" + str(len(results.nodes)))

    #Overpass will always provide full datasets, specific columns are selected
    node_tags_list = []
    for node in results.nodes:
        node.tags['latitude'] = node.lat
        node.tags['longitude'] = node.lon
        node.tags['id'] = node.id
        node_tags_list.append(node.tags)
    nodes_frame = pd.DataFrame(node_tags_list)
    addr_frame = nodes_frame[['latitude', 'longitude', 'id', 'addr:housenumber', 'addr:street',]]
    addr_frame.to_csv('addr_saved.csv')
    print("OSM address data saved to addr_saved.csv")
    return addr_frame

def get_osm_building(query):
    api = overpy.Overpass(url="https://maps.mail.ru/osm/tools/overpass/api/interpreter")
    print("Busy looking for buildings")
    results = api.query(query)
    print("Buildings found:" + str(len(results.ways)))

    way_tags_list = []
    for way  in results.ways:
        way.tags['id'] = way.id
        way.tags['center_lat'] = way.center_lat
        way.tags['center_lon'] = way.center_lon
        way_tags_list.append(way.tags)
    ways_frame = pd.DataFrame(way_tags_list)
    building_frame = ways_frame[['center_lat', 'center_lon', 'id']]
    building_frame.to_csv('building_saved.csv')
    print("OSM building data saved to building_saved.csv")
    return(building_frame)

def compare_building_addr(building_frame:pd.DataFrame, addr_frame:pd.DataFrame):
    building_array = building_frame[['center_lat', 'center_lon']].to_numpy()
    addr_array = addr_frame[['latitude', 'longitude']].to_numpy()
    building_addr = np.zeros_like(building_array)
    c = 0
    for i in addr_array:
        print(c)
        index = closest_node(i, building_array)
        building_addr[index] += 1
        c += 1
    np.savetxt("building_addr.csv", building_addr, delimiter=",")


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def load_csv(file_path):
    frame = pd.read_csv(file_path)
    return frame

def get_CBS(BBox):
    CBS_buurt_wijk_url = "https://service.pdok.nl/cbs/wijkenbuurten/2021/wfs/v1_0?request=getcapabilities&service=wfs"
    CBS_wfs = WebFeatureService(url=CBS_buurt_wijk_url)
    print(list(CBS_wfs.contents))
    print([operation.name for operation in CBS_wfs.operations])
    test_cbs = CBS_wfs.get_schema('cbs_buurten_2021')
    print(test_cbs)

def get_BAG(areas):
    
    BAG_url = "https://service.pdok.nl/lv/bag/wfs/v2_0?request=getCapabilities&service=WFS"
    BAG_wfs = WebFeatureService(url=BAG_url)
    print(list(BAG_wfs.contents))
    print([operation.name for operation in BAG_wfs.operations])
    test_BAG = BAG_wfs.get_schema('pand')
    test_pand = BAG_wfs.getfeature('pand', srsname='epsg:3857', outputFormat='xml', filter='Geometrygml:Polygongml:outerBoundaryIsgml:Linear' )
    print(test_BAG)


def main():
    #Delft Degrees [N, S, E, W]
    #Delft = [52.03, 51.96, 4.4, 4.3]
    #url="https://maps.mail.ru/osm/tools/overpass/api/interpreter"
    #url="https://overpass.kumi.systems/api/"

    #user_input = get_input()
    
    #addr_query = get_addr_query(user_input)
    
    #building_query = get_building_query(user_input)
    
    #building_frame = get_osm_building(building_query)
    building_frame = load_csv('building_saved.csv')

    #addr_frame = get_osm_addr(addr_query)
    addr_frame =  load_csv('addr_saved.csv')

    building_addr = compare_building_addr(building_frame, addr_frame)

    #get_BAG(['Delft', 'Den Hoorn'])


if __name__ == '__main__':
    main()