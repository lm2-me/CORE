# # # Exercise 2 20220914
# # # Henk Rusting 9717445


# def bisection_root(x):
#     epsilon = 0.001
#     iteration = 0
#     lower = 0.0
#     upper = max(1.0, x)
#     result = (upper + lower)/2.0

#     while abs(result ** 2 - x) > epsilon and iteration < 1200:
#        if result**2 < x:
#          lower = result
#        else:
#         upper = result
#         result = (upper + lower) /2
#         iteration +=1
#     return (result, iteration)

# find_square = 16
# resultaat, iteratie = bisection_root(find_square)
# print(bisection_root(find_square))
# print(find_square)
# create a file if it does not exist and
# return a file object with write access

import getpass
import datetime

cur_user = getpass.getuser()
cur_date = datetime.datetime.now()
file_header = '# created by {cuser} on [cdate}\n .format (cuser = cur_user, cdate = cur_date)'
print (cur_user, cur_date)
