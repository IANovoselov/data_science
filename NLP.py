"""
Обработка естественного языка
"""

import sys
import csv
import codecs
import re
import pandas as pd

import numpy as np
from neural import ActivationFunc, DerivativeFunc, Network
from collections import Counter
import math

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

raw_data = pd.read_csv("IMDB Dataset.csv", header=None)

raw_reviews = list(raw_data[0][1:25001])
ratings = list(raw_data[1][1:25001])


for i in range(25000):

    review = raw_reviews[i]
    review = PureText(review.strip()).remove_numbers().remove_html_tags().remove_symbols().text.strip()
    reviews.append(review)

    words = set(review.split())
    words = set(word.lower() for word in words)
    vocabulary.update(words)

with open('reviews.txt', 'w+') as file:
    for line in reviews:
        file.write(f"{line}\n")

with open('labels.txt', 'w+') as file:
    for line in ratings:
        file.write(f"{line}\n")


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
net.alpha = 0.03
net.batch_size = 100
net.need_dropout = False

correct, total = 0, 0
for itration in range(5):
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

def similar(net, indexed_vocabulary, target='beautiful'):
    target_index = indexed_vocabulary.get(target)

    if not target_index:
        return None

    scores = Counter()

    for word, index in indexed_vocabulary.items():
        raw_difference = net.weights[0][index] - net.weights[0][target_index]
        sq_difference = raw_difference ** 2
        scores[word] = -math.sqrt(sum(sq_difference))

    return scores.most_common(10)

print(similar(net, indexed_vocabulary))

print(similar(net, indexed_vocabulary))
