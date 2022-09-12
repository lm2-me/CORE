# # Excercises Chapter 1 Book A Practical Introduction to Python Programming
# # Henk Rusting 9717445


# #Excercise 1
# from sys import float_repr_style


# for i in range(4):
#     print('********************')
# print(' ')

# #Excercise 2
# print('********************')
# for i in range(2):
#     print("*                  *")
# print('********************')
# print(' ')

# #Excercise 3

# for i in range(4):
#     number_of_stars = i + 1
#     print('*'*number_of_stars)
# print(' ')

# #Excercise 4

# print((512-282)/((47*48)+5))
# print(' ')

# #Excercise 5

# try:
#     temp = float(input("Enter a number: "))
#     print("The square of ", temp, " is ", pow (temp, 2), ".", sep='')
# except ValueError:
#          print("Not a valid number.  Try again...")
# print(' ')

# #Exercise 6
# while True:
#     try:
#         x = float(input("Enter a number: "))
#         break
#     except ValueError:
#         print("Not a valid number.  Try again...")
# print(x ,2*x ,3*x ,4*x ,5*x , sep='---')
# print(' ')

#Exercise 7

# Define function weight kilograms to pounds
def weigth_in_pounds(weigth):
    lbs = round(kg*2.2, 1)
    return str(lbs) + " is your weigth in Pounds."

# # Get user input (weigth in kg). Check input
# while True:
#     try:
#         kg = float(input("Enter your weigth (kg): "))
#         break
#     except ValueError:
#         print("Not a valid number.  Try again...")
# print (kg)

# # Call function
# result = weigth_in_pounds(kg)
# print(result)

# print(' ')


