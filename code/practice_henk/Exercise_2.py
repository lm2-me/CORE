# # Exercise 2 20220914
# # Henk Rusting 9717445


def bisection_root(x):
    epsilon = 0.001
    iteration = 0
    lower = 0.0
    upper = max(1.0, x)
    result = (upper + lower)/2.0

    while abs(result ** 2 - x) > epsilon and iteration < 1200:
       if result**2 < x:
         lower = result
       else:
        upper = result
        result = (upper + lower) /2
        iteration +=1
    return (result, iteration)

find_square = 250567234234
print(bisection_root(find_square))