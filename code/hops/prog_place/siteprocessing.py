import ghhops_server as hs
import rhino3dm as r3d
import helpers as h

def divide_site(site: r3d.Surface, road_curve: r3d.PolylineCurve, sidewalk_curves_tree, grid_size: float):
    sidewalk_curves: list[r3d.PolylineCurve] = sidewalk_curves_tree['{0}']
    sidewalk_curve = sidewalk_curves[0]
        
    print('called method')

    print(sidewalk_curves)

    srfpts = []
    values = []
    
    srf_u_start = int(site.Domain(0).T0)
    srf_u_end = int(site.Domain(0).T1)
    srf_v_start = int(site.Domain(1).T0)
    srf_v_end = int(site.Domain(1).T1)

    print('pre-loop')

    for y in range(srf_v_start, srf_v_end, int(grid_size)):
        rowpts = []
        rowpts_vals = []
        for x in range(srf_u_start, srf_u_end, int(grid_size)):
            point = r3d.Point3d(x,y,0)
            rowpts.append(point)
            rowpts_vals.append(0)
        srfpts.append(rowpts)
        values.append(rowpts_vals)

    return h.list_to_tree(srfpts), h.list_to_tree(values)

def component_2(a, b):
    return 1