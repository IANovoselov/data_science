import numpy as np
import sys
from keras.datasets import mnist
from neural import ActivationFunc, DifferensiateFunc, Network


(x_tarin, y_train), (x_test, y_test) = mnist.load_data()

net = Network([3, 4, 1])
net.activation_func = ActivationFunc.relu
net.differenciate_func = DifferensiateFunc.relu

data = np.array([[0, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 ])

expectation = np.array([0, 1, 1, 0])

for iterations in range(1000):
    common_error = 0
    for i in range(len(expectation)):

        data_input = data[i]
        goal = expectation[i]

        result = net.forward(data_input)
        error = net.back_propagation(result, goal)

        common_error += error
print(common_error)
print(net.weights)