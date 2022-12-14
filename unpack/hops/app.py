from flask import Flask
import ghhops_server as hs

### Add additional component files under here
import prog_place.siteprocessing as siteprocessing
import WFC.WaveFunctionCollapse as WaveFunctionCollapse

### END COMPONENT FILES

# register hops app as middleware
app = Flask(__name__)
hops = hs.Hops(app)

#http://127.0.0.1:5000/dividesite

### Add addtitonal component registrations under here

### components written by LM2
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
        hs.HopsPoint("Context Point(s)", "C", "Context", access=hs.HopsParamAccess.TREE, optional=True),
        hs.HopsNumber("Sun Analysis", "S", "Sun Analysis", access=hs.HopsParamAccess.TREE),
    ],
    outputs=[
        hs.HopsPoint("Points", "P", "Field of points", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Cost", "$", "Value function results for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE),
    ]
)
def dividesite(site, road_lines_tree, sidewalk_lines_tree, grid_size, context, sun_hours):
    return siteprocessing.divide_site(site, road_lines_tree, sidewalk_lines_tree, grid_size, context, sun_hours)

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
        hs.HopsString("Door", "D", "Door Location")
    ],
    outputs=[
        hs.HopsSurface("Surface", "S", "Relocated surface"),
        hs.HopsString("Cost", "$", "Value function results for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Module Grid Edges", "E", "Field of points that represent the edge of the surface", access=hs.HopsParamAccess.LIST),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE),
    ]
)
def placepackages(srfpts_tree, cost_function_tree, lables_tree, module_use, module_geometry, module_mask, door):
    return siteprocessing.place_packages(srfpts_tree, cost_function_tree, lables_tree, module_use, module_geometry, module_mask, door)

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
        hs.HopsString("Doors", "D", "Doors defined as points", access=hs.HopsParamAccess.TREE),
    ],
    outputs=[
        hs.HopsSurface("Surface", "S", "Relocated surface", access=hs.HopsParamAccess.LIST),
        #hs.HopsString("Cost", "$", "Value function results for each point", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Module Grid Edges", "E", "Field of points that represent the edge of the surface", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE),
    ] 
)
def placemodules(srfpts_tree, cost_function_tree, lable_array, module_use_tree, module_geometry_list, module_mask_tree, door_tree):
    return siteprocessing.place_modules(srfpts_tree, cost_function_tree, lable_array, module_use_tree, module_geometry_list, module_mask_tree, door_tree)

@hops.component(
    "/labelstopoints1",
    name="LabelsToPoints",
    description="Labels to points",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsPoint("Points", "P", "Field of points", access=hs.HopsParamAccess.TREE),
        hs.HopsString("Lables", "L", "Lables for each point", access=hs.HopsParamAccess.TREE)
    ],
    outputs=[
        hs.HopsPoint("Site Points", "s", "Field of points that represent the site", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Site Edge Points", "e", "Field of points that represent the edge of the site", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Structure Boundary Points", "b", "Field of points that represent the boundary of the modules", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Structure Interior Points", "i", "Field of points that represent the interior of the modules", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Door Points", "x", "Field of points that represent the doors", access=hs.HopsParamAccess.TREE),
        hs.HopsPoint("Points Outside of Site", "o", "Field of points that represent the points outside of the site", access=hs.HopsParamAccess.TREE),
    ]
)
def labelstopoints(srfpts_tree, lable_array):
    return siteprocessing.labelstopoints(srfpts_tree, lable_array)

### components written by Seb
@hops.component(
    "/wavefunctioncollapse3",
    name="Wave Function Collapse3",
    description="Create surface structure from tiles using WFC",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsNumber("Dim x", "X", "Dimension in x of grid"),
        hs.HopsNumber("Dim y", "Y", "Dimension in y of grid"),
        hs.HopsNumber("Max Height", "H", "Max height offset of tile allowed"),
        hs.HopsNumber("Empty", "E", "Empty if True or 1, no tile at location", access=hs.HopsParamAccess.LIST)
    ],
    outputs=[
        hs.HopsNumber("Name Indices", "N", "Name index of tiles to place", access=hs.HopsParamAccess.LIST),
        hs.HopsNumber("Rotations", "R", "Rotation of tile", access=hs.HopsParamAccess.TREE),
        hs.HopsNumber("Heights", "H", "Height offset for tile", access=hs.HopsParamAccess.TREE),
    ]
)
def run_my_WFC(dx, dy, max_height, empty_list):
    return WaveFunctionCollapse.runWFC(dx, dy, max_height, empty_list)


### END COMPONENT REGISTRATION

if __name__ == "__main__":
    app.run()

