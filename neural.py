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
    def relu(cls, value):
        if value > 0:
            return value
        else:
            return 0.0

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

    @classmethod
    def relu(cls, value):
        if value >= 0:
            return 1.0
        else:
            return 0.0


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

        self._activation_func = None
        self._differenciate_func = None

        self._weights = []
        self.layer_inputs = []
        self.activations = []
        self.layers = len(neurons_counts)
        self.neurons_counts = neurons_counts

        self._weights = [np.random.randn(y, x)
                         for x, y in zip(neurons_counts[:-1], neurons_counts[1:])]

    def forward(self, inputs: List[int]) -> np.ndarray:
        """
        Прямое распространение
        :param inputs:
        :return:
        """

        result = np.array([inputs]).T
        self.layer_inputs.append(result)
        self.activations.append(result)

        for layer_num in range(self.layers-1):

            # Запомнить результат выисления на каждом слое
            layer_result = np.dot(self.weights[layer_num], result)
            self.layer_inputs.append(layer_result)

            result = self.activate(layer_result)
            self.activations.append(result)

        return result

    def back_propagation(self, calc_result: np.ndarray, goal: np.ndarray) -> None:
        """
        Обратное распространение ошибки
        :param calc_result: Расчитанный результат
        :param goal: Ожидаемый резульат
        :return:
        """
        np_diff_func = np.vectorize(self.differenciate_func)

        alpha = 0.1

        error = (calc_result - goal)**2

        delta_output = (calc_result - goal) * np_diff_func(calc_result)
        delta_weights_output = alpha * np.dot(delta_output, self.activations[-2].T)

        weights_old = self.weight[-1].copy()
        self.weight[-1] -= delta_weights_output

        for layer_num in range(2, self.layers):

            i = -layer_num

            delta_hidden = np.dot(weights_old.T, delta_output) * np_diff_func(self.layer_inputs[i])
            delta_weights_hidden = alpha * np.dot(delta_hidden, self.activations[i-1].T)

            weights_old = self.weight[i].copy()
            self.weight[i] -= delta_weights_hidden

            delta_output = delta_hidden

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

    @activation_func.setter
    def activation_func(self, func):
        """

        :param value:
        :return:
        """
        self._activation_func = func

    @property
    def differenciate_func(self):
        """
        Вернуть функцию активации
        :return:
        """
        return self._differenciate_func or DifferensiateFunc.sigmoid

    @differenciate_func.setter
    def differenciate_func(self, func):
        """

        :param value:
        :return:
        """
        self._differenciate_func = func

    def __repr__(self):
        return f'Network({self.neurons_counts})'
