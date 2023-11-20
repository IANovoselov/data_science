from typing import List
from random import random
import numpy as np


class ActivationFunc:
    """
    Названий функций активации
    """
    SIGMOID = 'sigmoid'


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

        activation_func_dict = {None: self.sigmoid,
                                ActivationFunc.SIGMOID: self.sigmoid,
                                }

        self._activation_func = activation_func_dict.get(activation_func, self.sigmoid)

        self._weights = []
        self.layers = len(neurons_counts) - 1  # Количество слоёв, за исключением входа
        self.neurons_counts = neurons_counts[1:]

        # Заполнить веса для i-ого слоя
        for i, neuron_count in enumerate(neurons_counts[1:]):

            # Количество входов для i-ого слоя
            # i - индекс предыдущего слоя
            inputs_num = neurons_counts[i]

            self._weights.append([np.array([random() for _ in range(inputs_num)]) for _ in range(neuron_count)])

    def forward(self, inputs: List[int]) -> float:
        """
        Прямое распространение
        :param inputs:
        :return:
        """
        result = np.array(inputs)

        for l in range(self.layers):
            new_result = []
            for n in range(self.neurons_counts[l]):
                layer_result_by_neuron = result.dot(self.weights[l][n])
                new_result.append(self.activation(layer_result_by_neuron))

            result = np.array(new_result)

        return result

    @staticmethod
    def sigmoid(value):
        """
        Сигмоидальная функция активации
        :param value:
        :return:
        """
        return 1 / (1 + np.exp(-value))

    @property
    def weights(self):
        """
        Вернуть веса
        :return:
        """
        return self._weights

    @property
    def activation(self):
        """
        Вернуть функцию активации
        :return:
        """
        return self._activation_func

    def __repr__(self):
        return f'Network({self.neurons_counts}, {"sigmoid"})'


net = Network([2, 2, 1])
print(net.weights)
result = net.forward([1, 1])
print(result)







