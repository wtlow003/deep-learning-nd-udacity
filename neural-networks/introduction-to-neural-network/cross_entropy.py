"""
Implementing Error Function: Cross-Entropy

Instead of maximising probabilities of a model, you can also decrease the Error
Function = Cross Entropy, where a model who fits better have lower Cross Entropy.

To implement Cross-Entropy:

1. CE = -sum(yi * ln(pi) + (1 - yi) * ln(1 - pi))

There is an inverse relationship between Cross Entropy and the probability of an
event. A higher cross-entropy implies a lower probability for an event.
"""


import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    # turning list into np arrays
    Y = np.float_(Y)
    P = np.float_(P)

    # formula: CE = -sum (yi * ln(pi) + (1 - yi) * ln(1 - pi))
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
