import numpy as np
import sys
from keras.datasets import mnist, imdb
from neural import ActivationFunc, DerivativeFunc, Network
from png_to_mnist import imageprepare


# Свёрточная сеть
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

# На вход изображение 28х28 пикселей
input_rows = 28
input_cols = 28

# Свёртка 3х3
kernel_rows = 3
kernel_cols = 3
num_kernels = 16

hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels
kernels = np.random.randn(num_kernels, kernel_rows * kernel_cols)

net = Network([784, hidden_size, 10])
net.activation_func = [ActivationFunc.tanh, ActivationFunc.softmax]
net.derivative_func = [DerivativeFunc.tanh, DerivativeFunc.simple]
net.alpha = 2
net.batch_size = 128
net.need_dropout = False

net.weights = [kernels,
               np.random.randn(10, hidden_size)]


net.train_with_kernels(images, labels, iterations_num=300)

# data_input = np.array([imageprepare('test.png')])
# result = net.forward(data_input)
# print(result)

result = net.forward(test_images)
error = np.sum(np.round((result - test_labels.T)**2, 3))
correct_answers = sum([np.argmax(result[:, k]) == np.argmax(test_labels.T[:, k]) for k in range(len(test_labels))])
print(error/len(test_labels))
print(correct_answers/len(test_labels))