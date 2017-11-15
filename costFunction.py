from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params


def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor

    # Setup some useful variables
    m = X.shape[0]
    # print(X.shape)
    num_layers = len(layers)
    # print(num_layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    # Theta is a list (length: num_layers - 1) of matrices of size layers[i + 1]*(layers[i] + 1)

    # You need to return the following variables correctly 
    J = 0;

    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for j in range(m):
        yv[y[j], j] = 1

    # ================================ TODO ================================
    # In this point calculate the cost of the neural network (feedforward)

    # Première partie du calcul de J
    J1 = 0;
    for i in range(m):
        # xOp = Xi
        xOp = append([1], X[i])
        # print(shape(xOp))
        for j in range(num_layers - 2):
            z = matrix.dot(Theta[j], xOp)
            xOp = sigmoid(z)
            xOp = append([1], xOp)
        z = matrix.dot(Theta[num_layers - 2], xOp)
        xOp = sigmoid(z)
        # print(shape(xOp))
        # print(num_labels)

        for k in range(num_labels):
            J1 += ((yv[k, i]) * log(xOp[k])) + ((1 - yv[k, i]) * (log(1 - xOp[k])))
    J1 *= (- 1 / m);
    # print(J1)

    # Second part of J's calculus
    J2 = 0
    for i in range(num_layers - 1):
        for j in range(layers[i + 1]):
            for k in range(layers[i]):
                J2 += (Theta[i][j][k + 1]) ** 2
    J2 *= (lambd / (2 * m));
    # print(J2)

    J = J1 + J2;
    return J
