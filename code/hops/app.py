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
        hs.HopsCurve("Road Curve", "RC", "Curve of road"),
        hs.HopsCurve("Sidewalk Curve", "SC", "Curve of sidewalk", access=hs.HopsParamAccess.TREE),
        hs.HopsNumber("Integer", "I", "Grid size"),
    ],
    outputs=[
        hs.HopsPoint("Points", "P", "Field of points", access=hs.HopsParamAccess.TREE),
        hs.HopsNumber("Value", "V", "Value function results for each point", access=hs.HopsParamAccess.TREE),
    ]
)
def dividesite(site, road_curve, sidewalk_curves_tree, grid_size):
    return siteprocessing.divide_site(site, road_curve, sidewalk_curves_tree, grid_size)

### END COMPONENT REGISTRATION

if __name__ == "__main__":
    app.run()

