import numpy as np
import sys
from keras.datasets import mnist
from neural import ActivationFunc, DifferensiateFunc, Network
from png_to_mnist import imageprepare


(x_tarin, y_train), (x_test, y_test) = mnist.load_data()

# Преобразовать матрицу в массив - чтобы подавать на вход сети
images = x_tarin[0:1000].reshape(1000, 28*28)
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), 10))

for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

net = Network([784, 40, 10])
net.activation_func = np.vectorize(ActivationFunc.sigmoid)
net.differenciate_func = np.vectorize(DifferensiateFunc.sigmoid)
net.alpha = 0.001
batch_size = 100
iterations_num = 50

for iterations in range(iterations_num):
    common_error = 0
    for i in range(int(len(images) / batch_size)):
        batch_start = i * batch_size
        batch_stop = (i+1) * batch_size

        data_input = images[batch_start:batch_stop]
        goal = labels[batch_start:batch_stop].T

        result = net.forward(data_input)
        error = net.back_propagation(result, goal)

        common_error += np.sum(error) / (batch_size * 10)
    print(common_error / len(images), "Итерация: ", iterations)
print(net.weights)

data_input = np.array(imageprepare('test.png'))
result = net.forward(data_input[1:])