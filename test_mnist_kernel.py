import numpy as np
np.random.seed(1)

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
kernels = 0.02 * np.random.random((kernel_rows*kernel_cols, num_kernels)) - 0.1

net = Network([784, hidden_size, 10])
net.activation_func = [ActivationFunc.tanh, ActivationFunc.softmax_ud]
net.derivative_func = [DerivativeFunc.tanh, DerivativeFunc.simple]
net.alpha = 2
net.batch_size = 128
net.need_dropout = True

net.weights = [kernels,
               0.02 * np.random.random((hidden_size, 10)) - 0.1]


net.train_with_kernels(images, labels, iterations_num=50)

data_input = np.array([imageprepare('test.png')])
# result = net.forward(data_input)
# print(result)

net.batch_size = 1
correct_answers = 0

data_input = data_input.reshape(data_input.shape[0], 28, 28)

sections = []
for row_start in range(data_input.shape[1] - 3):
    for col_start in range(data_input.shape[2] - 3):
        section = net.get_image_section(data_input,
                                         row_start,
                                         row_start+3,
                                         col_start,
                                         col_start+3)
        sections.append(section)

expanded_input = np.concatenate(sections, axis=1)
flatten_input = expanded_input.reshape(expanded_input.shape[0] * expanded_input.shape[1], -1)
result = net.forward_du(flatten_input)

print('Ответ: ', np.argmax(result))