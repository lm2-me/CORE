# Excercise 1 Calculate Moment on a beam with a distributed load
# Henk Rusting 9717445

# Define moment caluculation function
def moment(q, l):
    m = round((q*(pow(l, 2)))/8,1)
    return str(m) + " kNm"

# Get user input (Length of beam). Including Check for valid input.
while True:
    try:
        l = float(input("Enter Beam Length (m): "))
        break
    except ValueError:
         print("Not a valid number.  Try again...")

# Get user input (Distributed load). Including Check for valid input.
while True:
    try:
        q = float(input("Enter Distributed Load (kN): "))
        break
    except ValueError:
         print("Not a valid number.  Try again...")

# Call function
result = moment(q, l)
print(result)
