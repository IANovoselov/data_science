from typing import List
from random import random
import numpy as np


class ActivationFunc:
    """
    Названий функций активации
    """

    @classmethod
    def sigmoid(cls, value):
        """
        Сигмоидальная функция активации
        :param value:
        :return:
        """
        return 1 / (1 + np.exp(-value))


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

        self._activation_func = activation_func

        self._weights = []
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
        result = np.array([inputs])

        for layer_num in range(self.layers):
            layer_result = result.dot(self.weights[layer_num])
            result = self.activate(layer_result)

        return result

    @property
    def weights(self):
        """
        Вернуть веса
        :return:
        """
        return self._weights

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
        return self._activation_func

    def __repr__(self):
        return f'Network({self.neurons_counts}, {"ActivationFunc.sigmoid"})'


net = Network([2, 2, 4,  1], ActivationFunc.sigmoid)
print(net.weights)
result = net.forward([1, 1])
print(result)
