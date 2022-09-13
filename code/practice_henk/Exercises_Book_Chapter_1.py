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
from tkinter import E


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

# #Excercise 8

# # Define Total Value Function

# def total_value(num_1, num_2, num_3):
#     total = num_1 + num_2 + num_3
#     return (total)

# # Define Average Value Function

# def average_value(result_2):
#     average = int(result_2/3)
#     return (average)

# # Get user input number 1

# while True:
#     try:
#         num_1 = float(input("Enter number 1: "))
#         break
#     except ValueError:
#         print("Not a valid number.  Try again...")
# print(num_1)

# # Get user input number 2

# while True:
#     try:
#         num_2 = float(input("Enter number 2: "))
#         break
#     except ValueError:
#         print("Not a valid number.  Try again...")
# print(num_2)

# # Get user input number 3

# while True:
#     try:
#         num_3 = float(input("Enter number 3: "))
#         break
#     except ValueError:
#         print("Not a valid number.  Try again...")
# print(num_3)

# # Call funtion Total Value

# result_1 = total_value(num_1, num_2, num_3)
# print(result_1, " is the total value.")

# # Call funtion Average Value
# result_2 = total_value(num_1, num_2, num_3)
# result_2 = average_value(result_1)
# print(result_2, " is the average value.")
# print(' ')

# # Excercise 9

# # Define Tip Value Function 

# def tip_value(meal_value, tip_percentage):
#     tip = meal_value * (tip_percentage/100)
#     return (tip)

# # Define Total Bill Function

# def total_bill(meal_value, tip):
#     total = meal_value + tip
#     return (total)

# while True:
#     try:
#         meal_value = float(input("Enter meal value: "))
#         break
#     except ValueError:
#         print("Not a valid number.  Try again...")
# print ("EURO", meal_value, "Meal value")

# while True:
#     try:
#         tip_percentage = float(input("Enter tip percentage: "))
#         break
#     except ValueError:
#         print("Not a valid number.  Try again...")
# print(tip_percentage, " %")

# # Call function Tip Value

# result_1 = tip_value(meal_value, tip_percentage)
# print(result_1, " is the tip value.")

# # Call function total value

# result_2 = total_bill(meal_value, result_1)
# print("EURO", result_2, "is the total bill.")







 
