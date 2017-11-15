from mainAux import mainAux
import pickle

s = input('Tape the name of the machine: ')

accuracy = list()
accuracy.append(s)

if s == 'aerides':
    size_training = [5000, 7500, 10000, 20000, 40000, 50000, 60000]
    size_test = 5000
    hidden_layers = 1
    nodes = [100 for i in range(hidden_layers)]
    lambd = 3
    maxfun = 50

    for i in range(7):
        accuracy.append(mainAux(size_training[i], size_test, hidden_layers, nodes, lambd, maxfun))

if s == 'barlia':
    size_training = 20000
    size_test = [100, 500, 1000, 2000, 3000, 4000, 5000]
    hidden_layers = 1
    nodes = [100 for i in range(hidden_layers)]
    lambd = 3
    maxfun = 50

    for i in range(7):
        accuracy.append(mainAux(size_training, size_test[i], hidden_layers, nodes, lambd, maxfun))


if s == 'calanthe':
    size_training = 20000
    size_test = 5000
    hidden_layers = 1
    nodes = [[10], [20], [50], [75], [100], [150], [200]]
    lambd = 3
    maxfun = 50

    for i in range(7):
        accuracy.append(mainAux(size_training, size_test, hidden_layers, nodes[i], lambd, maxfun))


if s == 'diuris':
    size_training = 20000
    size_test = 5000
    hidden_layers = 1
    nodes = [100 for i in range(hidden_layers)]
    lambd = [0, 0.2, 0.5, 1, 3, 5, 10]
    maxfun = 50

    for i in range(7):
        accuracy.append(mainAux(size_training, size_test, hidden_layers, nodes, lambd[i], maxfun))


if s == 'encyclia':
    size_training = 20000
    size_test = 5000
    hidden_layers = 1
    nodes = [100 for i in range(hidden_layers)]
    lambd = 3
    maxfun = [10, 25, 50, 75, 100, 200, 400]

    for i in range(7):
        accuracy.append(mainAux(size_training, size_test, hidden_layers, nodes, lambd, maxfun[i]))


if s == 'epipactis':
    size_training = 20000
    size_test = 5000
    hidden_layers = [1, 2, 3, 4, 5, 6, 7]
    lambd = 3
    maxfun = 50

    for i in range(7):
        nodes = [50 for i in range(hidden_layers[i])]
        accuracy.append(mainAux(size_training, size_test, hidden_layers[i], nodes, lambd, maxfun))


with open(s, 'wb') as file:
    mon_pi = pickle.Pickler(file)
    mon_pi.dump(accuracy)