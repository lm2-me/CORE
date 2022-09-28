# Exercise_1

# Write a program that counts how many of the squares of the numbers from 1 to 100 end in a 1.

# count = 0
# for i in range(1,100):
#     square = i**2
#     if square%10 == 1:
#         count = count + 1
#         print(i, square)
# print(count)

# Exercise_2

# Write a program that counts how many of the squares of the numbers from 1 to 100 end in a
# 4 and how many end in a 9.

# count_4 = 0
# count_9 = 0
# for i in range(1,100):
#     square = i**2
#     if square%10 == 4:
#         count_4 = count_4 + 1
#         print(i,"---",square)
#     if square%10 == 9:
#         count_9 = count_9 + 1
#         print(i,"---",square)
# print(count_4,"---", count_9)

# Exercise_3

# Write a program that asks the user to enter a value n, and then computes (1+ 1/2 
# + 1/3 +· · ·+ 1/n )−ln(n). The ln function is log in the math module.

# from math import log as ln
# n=int(input("Enter the number of terms: "))
# sum1=0
# for i in range(1,n+1):
#     sum1=sum1+(1/i)
# print ("The sum of series is",round(sum1 - ln(n),2))
# print(sum1,"---",ln(n))

# Exercise_4

# Write a program to compute the sum 1 − 2 + 3 − 4 + · · · + 1999 − 2000.
# sum = 0
# for i in range(1,2001):
#     if i % 2 != 0:
#         sum += i
#     else: 
#         sum -= i
# print(sum)

# Exercise_5

# Write a program that asks the user to enter a number and prints the sum of the divisors of
# that number. The sum of the divisors of a number is an important function in number theory.

# sum = 0
# number = eval(input("Enter an integer: "))
# for i in range(1,number + 1):
#     if (number % i == 0):
#         sum += i
#         print(i, "---", sum)

# Exercise_6

# A number is called a perfect number if it is equal to the sum of all of its divisors, not including
# the number itself. For instance, 6 is a perfect number because the divisors of 6 are 1, 2, 3, 6
# and 6 = 1 + 2 + 3. As another example, 28 is a perfect number because its divisors are 1, 2, 4,
# 7, 14, 28 and 28 = 1 + 2 + 4 + 7 + 14. However, 15 is not a perfect number because its divisors
# re 1, 3, 5, 15 and 15 ̸= 1 + 3 + 5. Write a program that finds all four of the perfect numbers
# that are less than 10000.


 
# for n in range(1, 10001):
#     sum = 0
#     for i in range(1, n):
#         if n%i == 0:
#             sum += i
#     #print(n, sum)

#     if n == sum:
#         print(n)

# Exercise_7

# An integer is called squarefree if it is not divisible by any perfect squares other than 1. For
# instance, 42 is squarefree because its divisors are 1, 2, 3, 6, 7, 21, and 42, and none of those
# numbers (except 1) is a perfect square. On the other hand, 45 is not squarefree because it is
# divisible by 9, which is a perfect square. Write a program that asks the user for an integer and
# tells them if it is squarefree or not.

# number = eval(input("Enter an integer: "))
# count = 0
# for i in range(1, number + 1):
#     if number % (i**2) == 0 and i != 1:
#         count = count + 1
#         print(count, i, i**2)
# if count == 0:
#         print(number, " is squarefree!")
# else:
#     print(number, " is not squarefree!")

# Exercise_8

# Write a program that swaps the values of three variables x, y, and z, so that x gets the value
#  of y, y gets the value of z, and z gets the value of x.

# x = 1
# y = 2
# z = 3

# x, y = y, x
# y, z = z, y


# print(x)
# print(y)
# print(z)

# Exercise_9

# Write a program to count how many integers from 1 to 1000 are not perfect squares, perfect
# cubes, or perfect fifth powers.

# import math
# count = 0
# for n in range (1,1001):
    
#     root = math.sqrt(n)
#     if int(root) ** 2 == n:
#         count += 1
#         print(n, "Perfect square")
#     cube = (n**(1/3))
#     if int(cube) ** 3 == n:
#         count += 1
#         print(n, " Perfect cube")
#     five = (n**(1/5))
#     if int(five) ** 5 == n:
#         count += 1
#         print(n, " Perfect fifth power")
# print(count)

# Exercise_10

# Ask the user to enter 10 test scores. Write a program to do the following:
# (a) Print out the highest and lowest scores.
# (b) Print out the average of the scores.
# (c) Print out the second largest score.
# (d) If any of the scores is greater than 100, then after all the scores have been entered, print
#     a message warning the user that a value over 100 has been entered.
# (e) Drop the two lowest scores and print out the average of the rest of them.


# scores  = []
# n = 10

# print("\n")
# for i in range(0, n):
#     print("Enter number at index", i, )
#     item = int(input())
#     scores.append(item)
# print("User list is ", scores)

# # d > 100

# for i in range (10):
#  if scores[i] > 100:
#     print("A value over 100 has been entered!")

# # a min/max

# maximum = max(scores)
# print(maximum, " is the maximum")

# minimum = min(scores)
# print(minimum, " is the minimum")

# # b average
# average = sum(scores)/len(scores)
# print(average, " is the average")

# # c 2nd largest

# scores.sort()
# print("The second largest element of the list is:", scores[-2])

# # e 


# for i in range(0, 2):           # This removes n=2 items
#     scores.remove(min(scores))
# print(scores)
# average = sum(scores)/len(scores)
# print(average, " is the average")

# Exercise_11

# Write a program that computes the factorial of a number. The factorial, n!, of a number n is the
# product of all the integers between 1 and n, including n. For instance, 5! = 1 · 2 · 3 · 4 · 5 = 120.
# [Hint: Try using a multiplicative equivalent of the summing technique.]

# n = eval(input("Enter an integer: "))
# fact = 1
  
# for i in range(1,n+1):
#     fact = fact * i
      
# print ("The factorial of 23 is : ",end="")
# print (fact)

# Exercise_12

# Write a program that asks the user to guess a random number between 1 and 10. If they guess
# right, they get 10 points added to their score, and they lose 1 point for an incorrect guess. Give
# the user five numbers to guess and print their score after all the guessing is done.

# from random import randint

# counter = 0
# for i in range (5):
#     number = randint(1,10)
#     print(number)
#     guess = eval(input("Guess the number: "))
#     if guess == number:
#          counter = counter + 10
#          print("You guessed the number")
#     else:
#         counter = counter - 1
# print(counter)

# Exercise_13

# In the last chapter there was an exercise that asked you to create a multiplication game for
# kids. Improve your program from that exercise to keep track of the number of right and
# wrong answers. At the end of the program, print a message that varies depending on how
# many questions the player got right.

# Write a multiplication game program for kids. The program should give the player ten randomly
# generated multiplication questions to do. After each, the program should tell them
# whether they got it right or wrong and what the correct answer is.

# for i in range(1,11):
#     number_1 = int(input("Enter a number: "))
#     number_2 = int(input("Enter another number: "))
#     outcome_input = int(input("Enter the outcome of the multiplication of number 1 and 2: "))
#     output_calculated = number_1 * number_2
#     if outcome_input == output_calculated:
#         print("Correct!")
#     else:
#         print("Wrong.", " The correct answer is ", output_calculated)
# right = 0
# wrong = 0
# for i in range(1,11):
#     number_1 = int(input("Enter a number: "))
#     number_2 = int(input("Enter another number: "))
#     outcome_input = int(input("Enter the outcome of the multiplication of number 1 and 2: "))
#     output_calculated = number_1 * number_2
#     if outcome_input == output_calculated:
#         print("Correct!")
#         right += 1
#     else:
#         print("Wrong.", " The correct answer is ", output_calculated)
#         wrong += 1
# percentage = (right / 10) * 100 
# print("You answered ", percentage, "% of the questions correctly!")

# Exercise_14

# This exercise is about the well-known Monty Hall problem. In the problem, you are a contestant
# on a game show. The host, Monty Hall, shows you three doors. Behind one of those
# doors is a prize, and behind the other two doors are goats. You pick a door. Monty Hall, who
# knows behind which door the prize lies, then opens up one of the doors that doesn’t contain
# the prize. There are now two doors left, and Monty gives you the opportunity to change your
# choice. Should you keep the same door, change doors, or does it not matter?
# (a) Write a program that simulates playing this game 10000 times and calculates what percentage
# of the time you would win if you switch and what percentage of the time you
# would win by not switching.
# (b) Try the above but with four doors instead of three. There is still only one prize, and
# Monty still opens up one door and then gives you the opportunity to switch.

from numpy import random
import numpy as np
import time

def MontyHallSimulation (N):
    ChoiceUnchanged=[]
    ChoiceChanged=[]
    NN=1
    for i in range(0,N):
        
        # 1) The car is placed behind a random door.
        WinningDoor=random.choice(['Door 1', 'Door 2', 'Door 3'])

        # 2) The contestant selects a random door.
        FirstSelection=random.choice(['Door 1', 'Door 2', 'Door 3'])
        
        # 3) The host opens a door that is different than the contestants choice 
        #    and not the door with the car.
        HostOpens=list(set(['Door 1', 'Door 2', 'Door 3'])-set([FirstSelection,WinningDoor]))[0]
        
        # 4) The other door is not the participant's selected door and not the opened door. 
        OtherDoor=list(set(['Door 1', 'Door 2', 'Door 3'])-set([FirstSelection,HostOpens]))[0]
        
        # 5) Add "True" to a list where the participant DOES NOT change their selection AND thier 
        #    selection identified the door with the car. 
        ChoiceUnchanged.append(FirstSelection==WinningDoor)
        
        # 6) Add "True" to a list where the participant DOES change their selection and thier 
        #    new selected door has the car behind it.
        ChoiceChanged.append(OtherDoor==WinningDoor)
        
    # NOTE: The boolean object "TRUE" is equal to 1 and "False" is equal to 0.
    #       As such, we can use the "sum" function to get the total number of wins
    #       for each strategy.
    print(f'\n\
    {N:,} games were played \n\
    Chances of winning the car based on the following strategies:\n\
    Remaining with initial selection: {"{:.1%}".format(sum(ChoiceUnchanged)/N)}\n\
    Switching doors: {"{:.1%}".format(sum(ChoiceChanged)/N)}')
            
###############################            
###### Run the Simulation######
###############################
Start_time = time.time()
MontyHallSimulation(N=100000)         
print(f'\nSimulation Completed in: {round(time.time()-Start_time,2)} Seconds')





        






