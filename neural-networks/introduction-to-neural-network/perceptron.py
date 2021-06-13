"""
Implementing the Perceptron Algorithm

Pseudocode for Perceptron Algorithm:

Given points with coordinate (p,q), label y and prediction given by the equation
y-hat = step(w1x1 + w2x2 + b):

1. If the point is correctly classified, do nothing.
2. If the point is misclassified, but it has a negative label, subtract
ap, aq and a to w1, w2 and b respectively.
3. If the point is misclassified, but it has a positive label, add ap, aq and a
to w1, w2 and b respectively.

In this case, a = learning_rate.
"""


import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # looping through each row of data in X
    for idx, val in enumerate(X):
        y_hat = prediction(val, W, b)
        # check whether y_hat is misclassified based on existing label
        if y_hat != y[idx]:
            # if misclassified data == 0, wi = wi + a * xi
            if y_hat == 0:
                W[0] += learn_rate * val[0]
                W[1] += learn_rate * val[1]
                b += learn_rate
            # elif misclassified data == 1, wi = wi - a * xi
            else:
                W[0] -= learn_rate * val[0]
                W[1] -= learn_rate * val[1]
                b -= learn_rate

    return W, b

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
