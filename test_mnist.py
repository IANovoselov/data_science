import numpy as np
import sys
from keras.datasets import mnist
from neural import ActivationFunc, DifferensiateFunc, Network
from png_to_mnist import imageprepare


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Данные для обучения
# Преобразовать матрицу в массив - чтобы подавать на вход сети
images = x_train[0:1000].reshape(1000, 28*28) / 255
labels = y_train[0:1000]

one_hot_labels = np.zeros((len(labels), 10))
for i, l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels

# Данные для проверки
test_images = x_test.reshape(len(x_test), 28*28) / 255
test_labels = np.zeros((len(y_test), 10))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

net = Network([784, 40, 10])
net.activation_func = [ActivationFunc.sigmoid, ActivationFunc.sigmoid]
net.derivative_func = [DifferensiateFunc.sigmoid, DifferensiateFunc.sigmoid]
net.alpha = 5
net.batch_size = 100
net.need_dropout = False

net.train(images, labels, iterations_num=300)

# data_input = np.array([imageprepare('test.png')])
# result = net.forward(data_input)
# print(result)

result = net.forward(test_images)
error = np.sum(np.round((result - test_labels.T)**2, 3))
correct_answers = sum([np.argmax(result[:, k]) == np.argmax(test_labels.T[:, k]) for k in range(len(test_labels))])
print(error/len(test_labels))
print(correct_answers/len(test_labels))