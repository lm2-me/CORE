# # Exercise 1 

# from random import randint

# for i in range(104):
#     x= randint(3, 6)
#     print(x)

# Exercise 2

# from random import randint

# # Generate random number x
# x = randint(0, 50)
# print(x)

# # Generate random y
# y = randint(2, 5)
# print(y)

# print(x)

# Exercise 3

# Generate random number x between and 10 and print name that number of times
# from random import randint

# x = randint(0, 10)
# print(x*"Henk")

# Exercise 4

# Generate a float x between 1 and 10 with 2 decimals

# import random

# y = round(random.uniform(1.00, 10.00), 2)

# print(y)

# Exercise 5

# Write a program that generates 50 random numbers such that the first number is between 1
# and 2, the second is between 1 and 3, the third is between 1 and 4, . . . , and the last is between
# 1 and 51.

# from random import randint

# for i in range (1,51):
#     x = randint (1, i+1)
#     print(i+1, " ", x)

# Exercise 6

# Write a program that asks the user to enter two numbers, x and y, and computes |x−y|/(x+y)

# x = int(input("Enter a number:"))
# y = int(input("Enter a number:"))

# z=abs(x-y)/(x+y)
# print(z)

# Exercise 7

# Write a program that asks the user to enter an angle between −180◦ and 180◦. Using an
# expression with the modulo operator, convert the angle to its equivalent between 0◦ and
# 360◦

# x = int(input("Enter an angle between -180 and 180 degrees: "))
# print (x % 360)

# Exercise 8

# Write a program that asks the user for a number of seconds and prints out how many minutes
# and seconds that is. For instance, 200 seconds is 3 minutes and 20 seconds. [Hint: Use the //
# operator to get minutes and the % operator to get seconds.]

# x = int(input("Enter a number of seconds:"))
# minutes = x//60
# seconds = x % (minutes*60)
# print(minutes, " Minutes", seconds, "Seconds")

# Exercise 9

# Enter hour: 8
# How many hours ahead? 5
# New hour: 1 o'clock

# hours = int(input("Enter number of hours: "))
# hours_ahead = int(input("Enter how many hours ahead: "))

# if hours + hours_ahead > 24 and hours + hours_ahead % 24 > 12:
#     print("New hour:", ((hours + hours_ahead) % 24) - 12, "o'clock ",(hours + hours_ahead) // 24, "Days later")
# elif hours + hours_ahead > 24:
#     print("New hour:", ((hours + hours_ahead) % 24), "o'clock ",(hours + hours_ahead) // 24, "Days later")

# elif hours_ahead > 12:
#     print("New hour:", (hours + hours_ahead)-12, "o'clock")
# elif hours_ahead < 12:
#     print("New hour:", (hours + hours_ahead), "o'clock")

# Exercise 10a

# One way to find out the last digit of a number is to mod the number by 10. Write a
# program that asks the user to enter a power. Then find the last digit of 2 raised to that
# power.

# x = int(input("Enter a power (Integer): "))
# y = 2**x
# print (y)
# z = (y % 10)
# print(z)

# Exercise 10b

# One way to find out the last two digits of a number is to mod the number by 100. Write
# a program that asks the user to enter a power. Then find the last two digits of 2 raised to
# that power.

# x = int(input("Enter a power (Integer): "))
# y = 2**x
# print (y)
# z = (y % 100)
# print(z)

# Exercise 10c

# Write a program that asks the user to enter a power and how many digits they want.
# Find the last that many digits of 2 raised to the power the user entered.

# x = int(input("Enter a power (Integer): "))
# d = int(input("How many digits do you want?: "))
# mod = (10 ** d)
# y = 2**x
# print (y)
# z = (y % mod)
# print(z)

# Exercise 11

# Write a program that asks the user to enter a weight in kilograms. The program should
# convert it to pounds, printing the answer rounded to the nearest tenth of a pound.

# weight_in_kg = float(input("What is your weight in kilograms?: "))
# weight_in_lbs = round((weight_in_kg * 2.2), 2)
# print('%.2f' % weight_in_lbs)

# Exercise 12

# Write a program that asks the user for a number and prints out the factorial of that number.

# n = int(input("Enter a number: "))
# fact = 1
  
# for i in range(1,n+1):
#     fact = fact * i
# print ("The factorial of 23 is : ",end="")
# print (fact)

# Exercise 13 degrees

# Write a program that asks the user for a number and then prints out the sine, cosine, and
# tangent of that number.

# from math import radians, sin, cos, tan, radians

# n = float(input("Enter a number (float): "))

# print ((sin(radians(n))))
# print(cos(radians(n)))
# print(tan(radians(n)))

# Exercise 13 radians

# Write a program that asks the user for a number and then prints out the sine, cosine, and
# tangent of that number.

# from math import radians, sin, cos, tan, radians

# n = float(input("Enter a number (float): "))

# print (sin(n))
# print(cos(n))
# print(tan(n))

# Exercise 14

# Write a program that asks the user to enter an angle in degrees and prints out the sine of that
# angle.
# from math import sin, radians

# angle = int(input("Enter an angle in degrees: "))
# print ("The sin of the angle is: " ,((sin(radians(angle)))))

# Exercise 15

# Write a program that prints out the sine and cosine of the angles ranging from 0 to 345◦ in
# 15◦ increments. Each result should be rounded to 4 decimal places.

# from math import sin, cos, radians

# for n in range(0,360,15):
#     print(n, "---", round((sin(radians(n))), 4), " ", round((cos(radians(n))),4))

# Exercise 16

# Write a program that asks the user to enter a year and prints out the date of Easter in that
# year.

Y = int(input("Enter a year to calculate Easter: "))
C = Y // 100
m = ((15 + C - (C/4)- (8*C + 13)/25)) % 30
n = (4 + C -(C/4)) % 7
a = Y % 4
b = Y % 7
c = Y % 19
d = (19 * c + m) % 30
e = (2 * a + 4 * b + 6 * d + n) % 7
