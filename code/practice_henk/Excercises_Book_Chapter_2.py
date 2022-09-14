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

# # Exercise 8



# while True:
#     try:
#         number = int(input("How many times do you want to print your name: "))
#         break
#     except ValueError:
#         print("Not a valid number.  Try again...")
# print (number)

# while True:
#     try:
#         name = str(input("Enter your name: "))
#         break
#     except:
#         print("Not a valid number.  Try again...")

# for i in range(number):
#     print(name)

# # Exercise 9

# # Display the Fibonacci sequence  n-th terms

# nterms = int(input("How many terms? "))

# # first two terms
# n1, n2 = 1, 1
# count = 0

# # check if the number of terms is valid
# if nterms <= 0:
#    print("Please enter a positive integer")
# # if there is only one term, return n1
# elif nterms == 1:
#    print("Fibonacci sequence upto",nterms,":")
#    print(n1)
# # generate fibonacci sequence
# else:
#    print("Fibonacci sequence:")
#    while count < nterms:
#        print(n1)
#        nth = n1 + n2
#        # update values
#        n1 = n2
#        n2 = nth
#        count += 1


    






