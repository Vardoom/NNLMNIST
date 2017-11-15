from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params
from hadamarProduct import hadamarProduct

def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the gradient for the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)
    # Theta is a list (length num_layers) of matrices of size layers[i + 1]*layers[i]
  
    # You need to return the following variables correctly 
    Theta_grad = [zeros(w.shape) for w in Theta]
    # Theta_grad has same shape as Theta
    # This step does the initialization

    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for j in range(m):
        yv[y[j], j] = 1

    # ================================ TODO ================================
    # In this point implement the backpropagation algorithm

    """
    deltaCap = [zeros(w.shape) for w in Theta]
    for i in range(m):  # among the sample vector X
        a = list()
        # we create the output
        xOp = append([1], X[i])
        a.append(xOp)
        z = list()
        z.append([])
        for j in range(num_layers - 2):
            zOp = matrix.dot(Theta[j], xOp)
            z.append(zOp)
            xOp = sigmoid(zOp)
            xOp = append([1], xOp)
            a.append(xOp)
        zOp = matrix.dot(Theta[num_layers - 2], xOp)
        z.append(zOp)
        xOp = sigmoid(zOp)
        a.append(xOp)

        # we intialize delta for the output layer
        delta = list()
        for d in range(num_layers):
            delta.append(array([]))
        delta[num_layers - 1] = array([a[num_layers - 1][k] - yv[k][i] for k in range(num_labels)])
        # print(shape(delta[num_layers - 1]))

        # we compute delta by descending
        for j in reversed(range(1, num_layers - 1)):
            # print(j)
            # print(shape(Theta[j]))
            # print(shape(delta[j+1]))
            prod = matrix.dot(matrix.transpose(Theta[j]), delta[j + 1])
            # print(shape(prod))
            # print(shape(z[j]))
            prod = prod[1:]
            delta[j] = hadamarProduct(prod, sigmoidGradient(z[j]))
            # print(shape(delta[j]))

        # we compute the great delta
        for l in range(1, num_layers - 1):
            print(l)
            # print(deltaCap[l])
            # print(delta[l+1])
            # print(a[l])
            deltaCap[l - 1] += matrix.dot(delta[l], matrix.transpose(a[l]))

    for i in range(num_layers - 1):
        for j in range(layers[i + 1]):
            for k in range(layers[i]):
                Theta_grad[i][j][k + 1] = deltaCap[i][j][k + 1] / m
    """

    for i in range(m):
        # Firstly, feed-forward
        a = []
        z = []
        x = copy(X[i]).transpose()
        a.append(x)
        for j in range(num_layers - 1):
            x = append(1., x)
            x = Theta[j].dot(x)
            z.append(x)
            x = sigmoid(x)
            a.append(x)
        # Then, backwards propagation
        delta = x - yv[::, i]
        for j in range(num_layers - 1):
            ly = num_layers - 2 - j
            Theta_grad[ly] += outer(delta, append(1., a[ly]))
            if ly > 0:
                delta = multiply(Theta[ly].transpose().dot(delta)[1::], sigmoidGradient(z[ly - 1]))

    # Regularization
    for i in range(len(Theta_grad)):
        Th = copy(Theta[i])
        Th[::, 0] = 0.
        Theta_grad[i] = Theta_grad[i] / m + lambd / m * Th

    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad