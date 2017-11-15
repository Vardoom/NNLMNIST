from numpy import *
from read_dataset import read_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from displayData import displayData
from randInitializeWeights import randInitializeWeights
from costFunction import costFunction
from unroll_params import unroll_params
from roll_params import roll_params
from scipy.optimize import *
from predict import predict
from backwards import backwards
from checkNNCost import checkNNCost
from checkNNGradients import checkNNGradients
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def mainAux(size_training, size_test, num_of_hidden_layers, lay, lambd, m):

    # ================================ Step 1: Loading and Visualizing Data ================================
    print("\nLoading and visualizing Data ...\n")

    # Reading of the dataset
    # You are free to reduce the number of samples retained for training, in order to reduce the computational cost
    # size_training = int(input('Please select the size of the training set: '))  # number of samples retained for training
    # size_test = int(input('Please select the size of the test set: '))  # number of samples retained for testing
    images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)


    # ================================ Step 2: Setting up Neural Network Structure &  Initialize NN Parameters ================================
    print("\nSetting up Neural Network Structure ...\n")

    # Setup the parameters you will use for this exercise
    input_layer_size = 784  # 28x28 Input Images of Digits
    num_labels = 10  # 10 labels, from 0 to 9 (one label for each digit)

    # num_of_hidden_layers = int(input('Please select the number of hidden layers: '))
    # num_of_hidden_layers = 1
    print("\n")

    layers = [input_layer_size]
    for i in range(num_of_hidden_layers):
        layers.append(lay[i])
        # layers = layers + [1]
    layers = layers + [num_labels]

    #input('\nProgram paused. Press enter to continue!!!')

    print("\nInitializing Neural Network Parameters ...\n")

    # Fill the randInitializeWeights.py in order to initialize the neural network weights.
    Theta = randInitializeWeights(layers)

    # Unroll parameters
    nn_weights = unroll_params(Theta)


    # ================================ Step 9: Training Neural Networks & Prediction ================================
    print("\nTraining Neural Network... \n")

    #  You should also try different values of the regularization factor
    # lambd = double(input('Please enter the value of lambda: '))

    # m = int(input('Please select the value of maxfun: '))

    res = fmin_l_bfgs_b(costFunction, nn_weights, fprime=backwards,
                        args=(layers, images_training, labels_training, num_labels, lambd), maxfun=m, factr=1., disp=True)
    Theta = roll_params(res[0], layers)

    # input('\nProgram paused. Press enter to continue!!!')

    print("\nTesting Neural Network... \n")

    pred = predict(Theta, images_test)
    accuracy = mean(labels_test == pred) * 100
    print('\nAccuracy: ' + str(accuracy))
    return(accuracy)