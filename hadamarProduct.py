import numpy as np

def hadamarProduct(x, y):
    n = len(y)
    # print('n = {} (hadamard)'.format(n))
    u = np.zeros(n)
    for i in range(n):
        u[i] = x[i] * y[i]
    return u