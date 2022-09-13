# Excercises Chapter 2 Book A Practical Introduction to Python Programming
# Henk Rusting 9717445

# Excercise 1

# for i in range(100):
#     print("Henk")

# Excercise 2

# for i in range(10000):
#     print("Henk", end='')

# # Excercise 3

# for i in range(100):
#     print(i+1, ' Henk')

# # Exercise 4

# for i in range(20):
#     print(i+1, (i+1)**2)

# Exercise 5

# for i in range(8,90,3 ):
#     print(i)

# Exercise 6

# for i in range(100,0,-2):
#     print(i)

# Exercise 7
 
# for i in range(10):
#     print("A", end = '')
# for i in range(7):
#     print("B", end = '')
# for i in range(4):
#     print("CD", end = '')
# for i in range(1):
#     print("EFFFFFFG", end = '')

# Exercise 8



while True:
    try:
        number = int(input("How many times do you want to print your name: "))
        break
    except ValueError:
        print("Not a valid number.  Try again...")
print (number)

while True:
    try:
        name = str(input("Enter your name: "))
        break
    except:
        print("Not a valid number.  Try again...")

print(name, "is your name.")


    






