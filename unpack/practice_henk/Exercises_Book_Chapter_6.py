# Exercise 1

# Write a program that asks the user to enter a string. The program should then print the
# following:
# (a) The total number of characters in the string
# (b) The string repeated 10 times
# (c) The first character of the string (remember that string indices start at 0)
# (d) The first three characters of the string
# (e) The last three characters of the string
# (f) The string backwards
# (g) The seventh character of the string if the string is long enough and a message otherwise
# (h) The string with its first and last characters removed
# (i) The string in all caps
# (j) The string with every a replaced with an e
# (k) The string with every letter replaced by a space

# a

# s = input ('Enter a string: ')
# print (len(s))

# # b
# print((s)*10)

# c
# print(s[0])

# d
# print(s[0:3])

# e
# print(s[-3:])

# f
# print(s[ : :-1])

# g
# if len(s) >=7:
#     print(s[6])
# else:
#     print("String not long enough!")

# h
# i = len(s)
# print(s[1:len(s)-1])

# i
# s=s.upper()
# print(s)

# j
# s=s.replace('a','e')
# print(s)

# k
# c = ''
# for c in s:
#     s=s.replace(c, ' ')
# print(s)

# Exercise 2

# A simple way to estimate the number of words in a string is to count the number of spaces
# in the string. Write a program that asks the user for a string and returns an estimate of how
# many words are in the string.

# s = input('Enter a number of words seperated by spaces:')
# print(s.count(' ')+1," is the number of words in the sentence!" )

# Exeercise 3

# People often forget closing parentheses when entering formulas. Write a program that asks
# the user to enter a formula and prints out whether the formula has the same number of opening
# and closing parentheses.

# s = input("Enter a formula with the correct number of ( and ):")
# if s.count('(') == s.count(')'):
#     print("The number of opening and closing parentheses is equal! ")
# else:
#     print("The number of opening and closing parentheses is not equal! ")

# Exercise 4

# Write a program that asks the user to enter a word and prints out whether that word contains
# any vowels.


# s=input("Enter a sentence:").lower()
# vowels=0
# for i in s:
#       if(i=='a' or i=='e' or i=='i' or i=='o' or i=='u'):
#             vowels=vowels+1
# print("Number of vowels is:")
# print(vowels)

# Exercise 5

# Write a program that asks the user to enter a string. The program should create a new string
# called new_string from the user’s string such that the second character is changed to an
# asterisk and three exclamation points are attached to the end of the string. Finally, print
# new_string. Typical output is shown below:
# Enter your string: Qbert
# Q*ert!!!

# s = input ('Enter a string: ')
# s=s.replace(s[1],'*')
# print(s, end='!!!')

# Exercise 6

# Write a program that asks the user to enter a string s and then converts s to lowercase, removes
# all the periods and commas from s, and prints the resulting string.

# s = input("Enter a string:").lower()
# print(s)

# for c in ',.':
#     s = s.replace(c, '')
# print(s)

# Exercise 7

# Write a program that asks the user to enter a word and determines whether the word is a
# palindrome or not. A palindrome is a word that reads the same backwards as forwards.

# s = input('Enter a word:').lower()
# if s == s[ : :-1]:
#     print("The word you entered is a palindrome!")
# else:
#     print("The word you entered is not a palindrome!")

# Exercise 8

# At a certain school, student email addresses end with @student.college.edu, while professor
# email addresses end with @prof.college.edu. Write a program that first asks the
# user how many email addresses they will be entering, and then has the user enter those addresses.
# After all the email addresses are entered, the program should print out a message
# indicating either that all the addresses are student addresses or that there were some professor
# addresses entered.


# counter = 0
# email_addresses  = []
# number = eval(input("Enter the number of e-mail addresses you want to enter: "))

# print("\n")
# for i in range(0, number):
#     print("Enter an email address at index", i, )
#     address = input().lower()
#     if 'prof' in address:
#         counter = counter + 1
#     email_addresses.append(address)
# print("User list is ", email_addresses)
# if counter != 0:
#     print("There were some professor addresses entered.")
# else:
#     print("All addresses are student addresses.")

# Exercise 9

# Ask the user for a number and then print the following, where the pattern ends at the number
#that the user enters.
# 1
#  2
#   3
#    4

# number = eval(input("Enter a number:"))
# for i in range (1,number+1):
#     print ((i-1)*' ', i )

# Exercise 10

# Write a program that asks the user to enter a string, then prints out each letter of the string
# doubled and on a separate line. For instance, if the user entered HEY, the output would be
#      HH
#      EE
#      YY


# word = input("Enter a word:")

# doubled_s = ''
# for c in word:
#     doubled_s = c*2
#     print("   ",doubled_s)


