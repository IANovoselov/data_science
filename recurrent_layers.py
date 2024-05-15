import sys, random, math
from collections import Counter
import numpy as np
from neural import ActivationFunc, DerivativeFunc, Network

np.random.seed(1)
random.seed(1)
f = open('reviews.txt')
raw_reviews = f.readlines()
f.close()

f = open('labels.txt')
raw_labels = f.readlines()
f.close()

tokens = list(map(lambda x:(x.split(" ")), raw_reviews))

vocab = set()
for sent in tokens:
    for word in sent:
        if len(word) > 0:
            vocab.add(word)
vocab = list(vocab)

indexed_vocabulary = {}
for i, word in enumerate(vocab):
    indexed_vocabulary[word] = i

input_data_set = []
for sent in tokens:
    review_indexes = []
    for word in sent:
        if word in indexed_vocabulary:
            review_indexes.append(indexed_vocabulary.get(word))

    input_data_set.append(list(set(review_indexes)))

target_data_set = []
for rating in raw_labels:
    if rating == 'positive\n':
        target_data_set.append(1)
    else:
        target_data_set.append(0)

net = Network([len(vocab), 100, 1])

net.weights = [0.2 * np.random.random((len(vocab), 100)) - 0.1,
               0.2 * np.random.random((100, 1)) - 0.1]

net.activation_func = [ActivationFunc.sigmoid, ActivationFunc.sigmoid]
net.derivative_func = [DerivativeFunc.sigmoid, DerivativeFunc.sigmoid]
net.alpha = 0.03
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

norms = np.sum(net.weights[0] * net.weights[0], axis=1)
norms.resize(norms.shape[0], 1)
normed_weights = net.weights[0] * norms

def make_sent_vect(words):
    indices = list(map(lambda x: indexed_vocabulary[x], filter(lambda x: x in indexed_vocabulary, words)))

    return np.mean(normed_weights[indices], axis=0)

reviews2vector = list()
for review in tokens:
    reviews2vector.append(make_sent_vect(review))
reviews2vector = np.array(reviews2vector)

def most_similar_reviews(review):
    v = make_sent_vect(review)
    scores = Counter()
    for i, val in enumerate(reviews2vector.dot(v)):
        scores[i] = val

    most_similar = []

    for idx, score in scores.most_common(3):
        most_similar.append(raw_reviews[idx][0:100])

    return most_similar

most_similar_reviews(['boring', 'awful'])