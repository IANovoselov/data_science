import numpy as np
import sys
from keras.datasets import mnist
from neural import ActivationFunc, DifferensiateFunc, Network


(x_tarin, y_train), (x_test, y_test) = mnist.load_data()

# Преобразовать матрицу в массив - чтобы подавать на вход сети
images = x_tarin[0:1000].reshape(1000, 28*28)
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

net = Network([784, 40, 10])
net.activation_func = ActivationFunc.relu
net.differenciate_func = DifferensiateFunc.relu

for iterations in range(350):
    common_error = 0
    for i in range(len(images)):

        data_input = images[i]
        goal = labels[i]

        result = net.forward(data_input)
        error = net.back_propagation(result, goal)

        common_error += error
print(common_error)
print(net.weights)