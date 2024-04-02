from typing import List, Any
from random import random
import numpy as np


class ActivationFunc:
    """
    Функции активации
    """

    @classmethod
    def sigmoid(cls, value):
        """
        Сигмоидальная функция активации
        :param value:
        :return:
        """
        return 1 / (1 + np.exp(-value))

    @classmethod
    def relu(cls, value):
        if value >= 0:
            return value

        return 0.0

    @classmethod
    def simple(cls, value):
        """

        :param value:
        :return:
        """
        return value

    @classmethod
    def softmax(cls, value):
        """

        """
        temp = np.exp(value)
        return temp / np.sum(temp)

    @classmethod
    def tanh(cls, value):
        """

        """
        return np.tanh(value)

class DifferensiateFunc:
    """
    Производные
    """
    @classmethod
    def sigmoid(cls, value):
        """
        Сигмоидальная функция активации
        :param value:
        :return:
        """
        return ActivationFunc.sigmoid(value)*(1-ActivationFunc.sigmoid(value))

    @classmethod
    def relu(cls, value):
        if value >= 0:
            return 1.0
        else:
            return 0.0

    @classmethod
    def simple(cls, value):
        """

        :param value:
        :return:
        """
        return 1

    @classmethod
    def tanh(cls, value):
        """

        """
        return np.tanh(value)


class Network:
    """
    Объект нейронной сети
    """

    def __init__(self, neurons_counts: List[int]):
        """
        Конструктор нейронной сети
        :param neurons_counts: Количестов нейронов в i-ом слое
        """
        assert isinstance(neurons_counts, list)
        assert len(neurons_counts) >= 2

        self._activation_func = []
        self._derivative_func = []

        self._weights = []
        self.layer_inputs = []
        self.activations = []
        self.layers = len(neurons_counts)
        self.neurons_counts = neurons_counts

        self._weights = [np.random.randn(y, x)
                         for x, y in zip(neurons_counts[:-1], neurons_counts[1:])]

        self.alpha = 0.005
        self.batch_size = 1

        self.dropout_masks = []
        self.need_dropout = False

    def forward(self, inputs: np.ndarray[Any]) -> np.ndarray:
        """
        Прямое распространение
        :param inputs:
        :return:
        """
        self.activations = []
        self.layer_inputs = []
        self.dropout_masks = []

        result = np.array(inputs).T

        self.layer_inputs.append(result.copy())
        self.activations.append(result.copy())

        for layer_num in range(self.layers-1):

            # Запомнить результат выисления на каждом слое
            layer_result = self.weights[layer_num].dot(result)
            self.layer_inputs.append(layer_result.copy())

            result = self.activate(layer_result.copy(), layer_num)

            if self.need_dropout:
                if layer_num != self.layers-2:
                    dropout_mask = np.random.randint(2, size=result.shape)
                    result *= dropout_mask * 2
                    self.dropout_masks.append(dropout_mask.copy())

            self.activations.append(result.copy())

        return result

    def back_propagation(self, calc_result: np.ndarray, goal: np.ndarray) -> None:
        """
        Обратное распространение ошибки
        :param calc_result: Расчитанный результат
        :param goal: Ожидаемый резульат
        :return:
        """

        error = np.sum(np.round((calc_result - goal)**2, 3))

        delta_output = (calc_result - goal) * self.derivative(self.layer_inputs[-1], -1)
        delta_output = delta_output/(self.batch_size) #* delta_output.shape[0])

        weights_old = self.weight[-1].copy()

        self.weight[-1] -= self.alpha * delta_output.dot(self.activations[-2].T)

        for layer_num in range(2, self.layers):

            i = -layer_num

            delta_hidden = weights_old.T.dot(delta_output) * self.derivative(self.layer_inputs[i], i)

            if self.need_dropout:
                delta_hidden *= self.dropout_masks[i+1]

            weights_old = self.weight[i].copy()

            self.weight[i] -= self.alpha * delta_hidden.dot(self.activations[i-1].T)

            delta_output = delta_hidden

        return error

    def train(self, x_train, y_train, iterations_num=100):

        for iterations in range(iterations_num):
            common_error, correct_answers = 0, 0
            for i in range(int(len(x_train) / self.batch_size)):
                batch_start = i * self.batch_size
                batch_stop = (i + 1) * self.batch_size

                data_input = x_train[batch_start:batch_stop]
                goal = y_train[batch_start:batch_stop].T

                result = self.forward(data_input)

                if len(goal) == 1:
                    correct_answers += (result[0][0] > (goal[0]* 0.9 or -0.1) and result[0][0] < (goal[0]* 1.1 or 0.1))
                else:
                    correct_answers += sum([np.argmax(result[:, k]) == np.argmax(goal[:, k]) for k in range(self.batch_size)])

                error = self.back_propagation(result, goal)

                common_error += error

            print('Итерация: ', iterations,
                  'Ошибок: ', np.round(common_error / len(y_train), 3),
                  'Правильных ответов: ', np.round(correct_answers / len(y_train), 3))


    @property
    def weights(self):
        """
        Вернуть веса
        :return:
        """
        return self._weights

    @weights.setter
    def weight(self, value):
        """

        :param value:
        :return:
        """
        self._weights = value

    def activate(self, layer_result: np.ndarray, layer) -> np.ndarray:
        """
        Активировать
        :param layer_result:
        :return:
        """
        func = self._activation_func[layer]

        return func(layer_result)

    @property
    def activation_func(self):
        """
        Вернуть функцию активации
        :return:
        """
        return self._activation_func

    @activation_func.setter
    def activation_func(self, funcs):
        """

        :param value:
        :return:
        """
        self._activation_func = funcs

    def derivative(self, value, layer):
        """
        Вернуть функцию активации
        :return:
        """
        func = self._derivative_func[layer]

        return func(value)

    @property
    def derivative_func(self, layer):
        """
        Вернуть функцию активации
        :return:
        """
        return self._derivative_func[layer]

    @derivative_func.setter
    def derivative_func(self, funcs):
        """

        :param value:
        :return:
        """
        self._derivative_func = funcs

    def __repr__(self):
        return f'Network({self.neurons_counts})'
