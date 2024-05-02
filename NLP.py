"""
Обработка естественного языка
"""

import sys
import csv
import codecs
import re

import numpy as np
from neural import ActivationFunc, DerivativeFunc, Network

vocabulary = set()

reviews = []
ratings = []

class PureText:

    def __init__(self, text):
        self.text = text

    def remove_numbers(self):
        self.text = re.sub('[0-9]+', '', self.text)
        return self

    def remove_html_tags(self):
        self.text = re.sub(r'<[^>]+>', '', self.text)
        return self

    def remove_symbols(self):
        self.text = re.sub("[-\?;,!@#$%^&*(){}£\/'']",'', self.text).replace('"', '').replace('...', ' ')\
            .replace('.', ' ').replace('..', ' ').replace('--', ' ')
        self.text = re.sub("( \?\(!@#$%^&*()_+=-'\:;|/`~.,{})",'', self.text)
        return self

with codecs.open("IMDB Dataset.csv", "r", "utf_8_sig" ) as file:
    for i, line in enumerate(file):
        line = line.split('",')

        if not line or len(line) < 2:
            continue

        review = PureText(line[0].strip()).remove_numbers().remove_html_tags().remove_symbols().text.strip()
        reviews.append(review)
        ratings.append(line[1].strip())

        words = set(review.split())
        words = set(word.lower() for word in words)
        vocabulary.update(words)

        if i == 24999:
            break

indexed_vocabulary = {}
for i, word in enumerate(vocabulary):
    indexed_vocabulary[word] = i

input_data_set = []
for review in reviews:
    review_indexes = []
    for word in review.split():
        review_indexes.append(indexed_vocabulary.get(word.lower()))

    input_data_set.append(review_indexes)


target_data_set = []
for rating in ratings:
    if rating == 'positive':
        target_data_set.append(1)
    else:
        target_data_set.append(0)

x = 4


net = Network([len(vocabulary), 100, 1])

net.weights = [0.2 * np.random.random((len(vocabulary), 100)) - 0.1,
               0.2 * np.random.random((100, 1)) - 0.1]

net.activation_func = [ActivationFunc.sigmoid, ActivationFunc.sigmoid]
net.derivative_func = [DerivativeFunc.sigmoid, DerivativeFunc.sigmoid]
net.alpha = 0.01
net.batch_size = 100
net.need_dropout = False

correct, total = 0, 0
for itration in range(2):
    for i in range(len(input_data_set) - 1000):

        x, y = (input_data_set[i], target_data_set[i])

        layer_1 = net.activation_func[0](np.sum(net.weights[0][x], axis=0))
        layer_2 = net.activation_func[1](np.dot(layer_1, net.weights[1]))

        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta.dot(net.weights[1].T)

        net.weights[0][x] -= layer_1_delta * net.alpha
        net.weights[1] -= np.outer(layer_1, layer_2_delta) * net.alpha

        if np.abs(layer_2_delta) < 0.5:
            correct += 1
        total += 1

        if i % 10 == 9:
            print('correct:', correct / total)


