# Excercise 1 Calculate Moment on a beam with a distributed load
# Henk Rusting 9717445
import math


# Get user input (Length and distributed load)
l = float(input("Enter Beam Length (m): "))
q = float(input("Enter Distributed Load (kN): "))

# Calculate moment using function
def moment(q, l):
    m = round((q*(l**2))/8,1)

    return str(m) + " kNm"

# Call function
result = moment(q, l)
print(result)
