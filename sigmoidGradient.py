from numpy import *
from sigmoid import sigmoid


def sigmoidGradient(z):
    # SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z

    g = zeros(z.shape)
    # =========================== TODO ==================================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z.

    g = sigmoid(z)

    for i in range(len(z)):
        if z[i] == 0:
            g[i] = 0.25
        else:
            g[i] = g[i] * (1 - g[i])

    return g
