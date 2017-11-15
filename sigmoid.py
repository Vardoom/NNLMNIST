from numpy import *


def sigmoid(z):
    # SIGMOID returns sigmoid function evaluated at z
    g = zeros(shape(z))

    # ============================= TODO ================================
    # Instructions: Compute sigmoid function evaluated at each value of z.

    for i in range(len(z)):
        if z[i] == 0:
            g[i] = 0.5
        else:
            g[i] = 1 / (1 + math.exp(-z[i]))

    return g
