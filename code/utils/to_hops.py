
### DO NOT USE THIS FILE, see 'hops' folder app.py ###

# Import packages
from flask import Flask
import ghhops_server as hs
import rhino3dm

app = Flask(__name__)
hops = hs.Hops(app)

# This decorator is an example
# Change the parameters here for your own Grasshopper components
# The functions that is actually is running can be find under decorator
@hops.component(
    "/pointat",
    name="PointAt",
    description="Get point along curve",
    icon="examples/pointat.png",
    inputs=[
        hs.HopsCurve("Curve", "C", "Curve to evaluate"),
        hs.HopsNumber("t", "t", "Parameter on Curve to evaluate"),
    ],
    outputs=[
        hs.HopsPoint("P", "P", "Point on curve at t")
    ]
)
def pointat(curve, t):
    print(t)
    return curve.PointAt(t)

if __name__ == "__main__":
    app.run()
    # After running this script:
    # Open Rhino and GH and add hops component
    # Run the printed server with the path input: http://127.0.0.1:5000/