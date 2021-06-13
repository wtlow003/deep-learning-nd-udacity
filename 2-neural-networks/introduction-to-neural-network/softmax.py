"""
Implementing Softmax Function

To compute Softmax function for an entire array of inputs:

1. Computing the exponentials of each value in the entire array
2. Calculate the exponential sum of all the values
3. Divide each individual exponential by the exponential sum

NOTE: By using exponential instead of linear function, we prevent problem from
occurring when the inputs are negative (ZeroDivisionError). Using exp() enables
to convert every number within a given array into positive numbers.
"""


import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    # obtain the exponential sum of the input array
    exponentials = np.exp(L)
    # we need to divide each exponential by the sum of all exponentials
    exponential_sum = np.sum(exponentials)

    # using broadcasting to get individual probabilities
    return exponentials / exponential_sum
