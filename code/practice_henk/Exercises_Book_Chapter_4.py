# # Excercises Chapter 4 Book A Practical Introduction to Python Programming
# # Henk Rusting 9717445

#Exercise 1

# Write a program that asks the user to enter a length in centimeters. If the user enters a negative
# length, the program should tell the user that the entry is invalid. Otherwise, the program
# should convert the length to inches and print out the result. There are 2.54 centimeters in an
# inch.

# length_cm = float(input("Enter a length in cm: "))

# if length_cm < 0:
#     print ("Length is smaller than zero.")
# else:
#     length_inch = length_cm * 0.393700787
#     print ("The length is ", length_inch, " in inches.")

# Exercise 2

# Ask the user for a temperature. Then ask them what units, Celsius or Fahrenheit, the temperature
# is in. Your program should convert the temperature to the other unit. The conversions
# are F = 9/5(C + 32) and C = 5/9(F − 32).

# temperature = float(input("Enter the temperature: "))
# unit = str(input("Is the temperature in C or F? "))

# if unit =="C":
#     print("That is ", round((9/5)*(temperature + 32), 2), "in Fahrenheit!")
# else:
#     print("That is ", round((5/9)*(temperature - 32), 2), "in Fahrenheit!")

# Exercise 3
# Ask the user to enter a temperature in Celsius. The program should print a message based
# on the temperature:
# • If the temperature is less than -273.15, print that the temperature is invalid because it is
#   below absolute zero.
# • If it is exactly -273.15, print that the temperature is absolute 0.
# • If the temperature is between -273.15 and 0, print that the temperature is below freezing.
# • If it is 0, print that the temperature is at the freezing point.
# • If it is between 0 and 100, print that the temperature is in the normal range.
# • If it is 100, print that the temperature is at the boiling point.
# • If it is above 100, print that the temperature is above the boiling point.

# temperature_Celsius = float(input ("Enter a temperature in Celsius: "))
# if temperature_Celsius < -273.15:
#     print("The temperature is invalid because it is below absolute zero")
# elif temperature_Celsius == -273.15:
#     print("The temperature is absolute 0")
# elif temperature_Celsius <0 and temperature_Celsius > -273.15:
#     print("The temperature is below freezing.")
# elif temperature_Celsius == 0:
#     print ("The temperature is at the freezing point.")
# elif temperature_Celsius >0 and temperature_Celsius <100:
#     print("Temperature is in the normal range.")
# elif temperature_Celsius == 100:
#     print("Temperature is at the boiling point.")
# else:
#     print("Temperature is above the boiling point.")

# Exercise 4

# Write a program that asks the user how many credits they have taken. If they have taken 23
# or less, print that the student is a freshman. If they have taken between 24 and 53, print that
# they are a sophomore. The range for juniors is 54 to 83, and for seniors it is 84 and over.

# number_credits = int(input("Input your number of credits: "))

# if number_credits <= 23:
#     print("You are a freshman.")
# elif number_credits >= 24 and number_credits <=53:
#     print("You are a sophomore.")
# elif number_credits >=54 and number_credits <=83:
#     print("You are a junior")
# else:
#     print("You are a senior.")

# Exercise 5

# Generate a random number between 1 and 10. Ask the user to guess the number and print a
# message based on whether they get it right or not.

# from random import randint

# number = randint(1, 10)
# guess = int(input("Guess the number between 1 and 10: "))
# if guess == number:
#     print("You guessed the number!")
# else:
#     print("You guessed wrong!")

# Exercise 6

# A store charges $12 per item if you buy less than 10 items. If you buy between 10 and 99
# items, the cost is $10 per item. If you buy 100 or more items, the cost is $7 per item. Write a
# program that asks the user how many items they are buying and prints the total cost.

# number_items = int(input ("What is the number of items you want to buy? "))
# if number_items < 10:
#     print("If you buy", number_items, "items, the price is $12,- per item which brings the total to $", number_items * 12,",-")
# elif number_items >=10 and number_items < 100:
#     print("If you buy", number_items, "items, the price is $10,- per item which brings the total to $", number_items * 10,",-")
# else:
#     print("If you buy", number_items, "items, the price is $7,- per item which brings the total to $", number_items * 7,",-")

# Exercise 7

# Write a program that asks the user for two numbers and prints Close if the numbers are
# within .001 of each other and Not close otherwise.

# number_1 = float(input("Enter a number with 3 decimals: "))
# number_2 = float(input("Enter another number with 3 decimals: "))
# difference = round(abs(number_1 - number_2),3)
# if difference <= 0.001:
#     print("Close!")
# else:
#     print("Not close.")

# Exercise 8

# A year is a leap year if it is divisible by 4, except that years divisible by 100 are not leap years
# unless they are also divisible by 400. Write a program that asks the user for a year and prints
# out whether it is a leap year or not.

# year = int(input("Enter a year: "))
# if year % 4 == 0 and (year % 100 == 0 and year % 400 == 0):
#     print(year, "is a leap year!")
# else:
#     print(year, "is not a leap year")

# Exercise 9

# Write a program that asks the user to enter a number and prints out all the divisors of that
# number. [Hint: the % operator is used to tell if a number is divisible by something. See Section
# 3.2.]

# number = int(input("Enter a number:"))

# for i in range(1, number + 1):
#     if number % i == 0:
#         print(i)

# Exercise 10

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

# Exercise 11

# Write a program that asks the user for an hour between 1 and 12, asks them to enter am or pm,
# and asks them how many hours into the future they want to go. Print out what the hour will
# be that many hours into the future, printing am or pm as appropriate. An example is shown
# below.

# time = int(input("Enter the number of hours between 1 and 12: "))
# am_pm = str.lower(input("AM or PM?"))
# hours_ahead = int(input("Enter the number of hours ahead: "))

# if time + hours_ahead > 24 and time + hours_ahead % 24 > 12:
#     print("New hour:", ((time + hours_ahead) % 24) - 12, "o'clock ",(time + hours_ahead) // 24, "Days later")
# elif time + hours_ahead > 24:
#     print("New hour:", ((time + hours_ahead) % 24), "o'clock ",(time + hours_ahead) // 24, "Days later")

# elif hours_ahead > 12:
#     print("New hour:", (time + hours_ahead)-12, "o'clock")
# elif hours_ahead < 12:
#     print("New hour:", (time + hours_ahead), "o'clock")

# Exercise 12

# A jar of Halloween candy contains an unknown amount of candy and if you can guess exactly
# how much candy is in the bowl, then you win all the candy. You ask the person in charge the
# following: If the candy is divided evenly among 5 people, how many pieces would be left
# over? The answer is 2 pieces. You then ask about dividing the candy evenly among 6 people,
# and the amount left over is 3 pieces. Finally, you ask about dividing the candy evenly among
# 7 people, and the amount left over is 2 pieces. By looking at the bowl, you can tell that there
# are less than 200 pieces. Write a program to determine how many pieces are in the bowl.

# number_divided_5 = int(input("If the candy is divided evenly among 5 people, how many pieces would be left over? "))
# number_divided_6 = int(input("If the candy is divided evenly among 6 people, how many pieces would be left over? "))
# number_divided_7 = int(input("If the candy is divided evenly among 7 people, how many pieces would be left over? "))

# if number_divided_5 == 2 and number_divided_6 == 3 and number_divided_7 == 2:
#     print ("There are less than 200 pieces of candy in the bowl!")
# else:
#     print("There are more than 200 pieces of candy in the bowl!")

# Exercise 13

# Write a program that lets the user play Rock-Paper-Scissors against the computer. There
# should be five rounds, and after those five rounds, your program should print out who won
# and lost or that there is a tie.

from multiprocessing import RLock
from random import randint
turns = 0
draws = 0
player_wins = 0
computer_wins = 0
print("Rocks, paper, scissor")
for turns in range (1,6):
    print("Turn number:", turns)
    player = str.lower(input("Enter rock, paper or scissor: "))
    computer = randint(1,3)
    if computer == 1:
        print("Computer entered rock")
    elif computer == 2:
        print("Computer entered paper")
    else:
        print("computer entered scissor")
    
    if player == "rock" and computer == 1:
        print("It's a draw!")
        draws == draws +1

    elif player == "rock" and computer == 2:
        print("The computer wins!")
        computer_wins = computer_wins + 1
    elif player == "rock" and computer == 3:
        print("You win!")
        player_wins = player_wins +1
    elif player =="paper" and computer == 1:
        print("You win!")
        player_wins = player_wins +1
    elif player == "paper" and computer == 2:
        print("It's a draw!")
        draws == draws +1
    elif player == "paper" and computer == 3:
        print("The computer wins!")
        computer_wins = computer_wins + 1
    elif player == "scissor" and computer == 1:
        print("The computer wins!")
        computer_wins = computer_wins +1
    elif player == "scissor" and computer == 2:
        print("The computer wins!")
        computer_wins = computer_wins + 1
    else:
     print("It's a draw!")
     draws == draws +1
print ("Player wins: ", player_wins, "Computer wins: ", computer_wins, "Number of draws: ", draws)
if draws == 5:
    print ("Nobody won!")
if player_wins > computer_wins:
    print("You have won!")
else:
    print("The computer has won!")