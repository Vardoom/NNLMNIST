from numpy import *
from sigmoid import sigmoid

def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels for each one of the instances
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    # ================================ TODO ================================
    # You need to return the following variables correctly
    p = zeros((1,m))

    for i in range(m):
        a = copy(X[i])
        a = append(1., a)
        for j in range(num_layers - 2):
            z = Theta[j].dot(a)
            a = sigmoid(z)
            a = append(1, a)
        z = Theta[num_layers - 2].dot(a)
        a = sigmoid(z)
        # print(a)
        p[0, i] = list(a).index(max(a))

    return p

