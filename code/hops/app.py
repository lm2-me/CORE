from flask import Flask
import ghhops_server as hs

### Add additional component files under here
import prog_place.siteprocessing as siteprocessing

### END COMPONENT FILES

# register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)

#http://127.0.0.1:5000/dividesite

### Add addtitonal component registrations under here
@hops.component(
    "/dividesite",
    name="DivideSite",
    description="Turn site into grid",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsSurface("Geometry", "G", "Geometry of site"),
        hs.HopsLine("Road Lines", "RL", "Curve of road", access=hs.HopsParamAccess.TREE),
        hs.HopsLine("Sidewalk Lines", "SL", "Curve of sidewalk", access=hs.HopsParamAccess.TREE),
        hs.HopsNumber("Integer", "I", "Grid size"),
    ],
    outputs=[
        hs.HopsPoint("Points", "P", "Field of points", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Cost", "$", "Value function results for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE),
    ]
)
def dividesite(site, road_lines_tree, sidewalk_lines_tree, grid_size):
    return siteprocessing.divide_site(site, road_lines_tree, sidewalk_lines_tree, grid_size)

@hops.component(
    "/placepackages",
    name="PlacePackages",
    description="Place package module",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsPoint("Points", "P", "Field of points", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Cost", "$", "Value function results for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Module Use", "M_U", "Package module use info"),
        hs.HopsSurface("Module Geometry", "M_G", "Package module geometry info"),
        hs.HopsString("Module Mask", "M_M", "Package module mask"),
    ],
    outputs=[
        hs.HopsSurface("Surface", "S", "Relocated surface"),
        hs.HopsString("Cost", "$", "Value function results for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Module Grid Edges", "E", "Field of points that represent the edge of the surface", access=hs.HopsParamAccess.LIST),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE),
    ]
)
def placepackages(srfpts_tree, cost_function_tree, lables_tree, module_use, module_geometry, module_mask):
    return siteprocessing.place_packages(srfpts_tree, cost_function_tree, lables_tree, module_use, module_geometry, module_mask)

@hops.component(
    "/placemodules",
    name="Place Modules",
    description="Place modules",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsPoint("Points", "P", "Field of points", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Cost", "$", "Value function results for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Module Use", "M_U", "Tree with each module use info", access=hs.HopsParamAccess.TREE),
        hs.HopsSurface("Module Geometry (as flattened list)", "M_G", "Module geometry info", access=hs.HopsParamAccess.LIST),
        hs.HopsString("Module Mask", "M_M", "Tree with each module mask", access=hs.HopsParamAccess.TREE),
    ],
    outputs=[
        hs.HopsSurface("Surface", "S", "Relocated surface", access=hs.HopsParamAccess.LIST),
        #hs.HopsString("Cost", "$", "Value function results for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Module Grid Edges", "E", "Field of points that represent the edge of the surface", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE),
    ]
)
def placemodules(srfpts_tree, cost_function_tree, lable_array, module_use_tree, module_geometry_list, module_mask_tree):
    return siteprocessing.place_modules(srfpts_tree, cost_function_tree, lable_array, module_use_tree, module_geometry_list, module_mask_tree)

### END COMPONENT REGISTRATION

if __name__ == "__main__":
    app.run()

