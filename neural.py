from typing import List
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
    def simple(cls, value):
        """

        :param value:
        :return:
        """
        return value


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


class Network:
    """
    Объект нейронной сети
    """

    def __init__(self, neurons_counts: List[int], activation_func=None):
        """
        Конструктор нейронной сети
        :param neurons_counts: Количестов нейронов в i-ом слое
        """
        assert isinstance(neurons_counts, list)
        assert len(neurons_counts) >= 2

        self._activation_func = None
        self._differenciate_func = None

        self._weights = []
        self.layer_inputs = []
        self.layers = len(neurons_counts) - 1  # Количество слоёв, за исключением входа
        self.neurons_counts = neurons_counts

        # Заполнить веса для i-ого слоя
        for i, neuron_count in enumerate(neurons_counts[1:]):

            # Количество входов для i-ого слоя
            # i - индекс предыдущего слоя
            layer_inputs_num = neurons_counts[i]

            # Матрица весов i-ого слоя M x N - где М - количевто выходов из предыдущего слоя,
            # N - количество нейронов в i-ом слое
            self._weights.append(np.random.random((layer_inputs_num, neuron_count)))

    def forward(self, inputs: List[int]) -> np.ndarray:
        """
        Прямое распространение
        :param inputs:
        :return:
        """
        self.layer_inputs = []

        result = np.array([inputs])

        for layer_num in range(self.layers):

            # Запомнить результат выисления на каждом слое
            self.layer_inputs.append(result)

            layer_result = result.dot(self.weights[layer_num])
            result = self.activate(layer_result)




        return result

    def back_propagation(self, calc_result: np.ndarray, goal: np.ndarray) -> None:
        """
        Обратное распространение ошибки
        :param calc_result: Расчитанный результат
        :param goal: Ожидаемый резульат
        :return:
        """
        # np_diff_func = np.vectorize(self.differenciate_func)

        alpha = 0.1

        error = (goal-calc_result)**2

        for layer_num in range(self.layers):

            # diff = np_diff_func(error)

            delta_weights = alpha * (self.layer_inputs[layer_num] * (calc_result - goal))

            self.weight[layer_num] -= delta_weights.transpose()


        return error



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

    def activate(self, layer_result: np.ndarray) -> np.ndarray:
        """
        Активировать
        :param layer_result:
        :return:
        """
        np_activate = np.vectorize(self.activation_func)
        return np_activate(layer_result)

    @property
    def activation_func(self):
        """
        Вернуть функцию активации
        :return:
        """
        return self._activation_func or ActivationFunc.sigmoid

    @property
    def differenciate_func(self):
        """
        Вернуть функцию активации
        :return:
        """
        return self._differenciate_func or DifferensiateFunc.sigmoid

    def __repr__(self):
        return f'Network({self.neurons_counts}, {"ActivationFunc.sigmoid"})'


net = Network([3, 1], ActivationFunc.sigmoid)
print(net.weights)

data = np.array([[0, 0, 0],
                 [0, 0, 1],
                 [0, 1, 0],
                 [0, 1, 1],
                 [1, 0, 0],
                 [1, 0, 1],
                 [1, 1, 0],
                 [1, 1, 1],
                 ])

expectation = np.array([0, 0, 1, 1, 0, 0, 1, 1])

for iterations in range(100):
    common_error = 0
    for i in range(len(expectation)):
        result = net.forward(data[i])
        error = net.back_propagation(result, expectation[i])

        common_error += error
    print(error)

print(net.weights)
print(result)
