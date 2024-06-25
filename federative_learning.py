import numpy as np
from collections import Counter
import random
import sys
from tensor import Tensor, SGD, Sequential, Linear, MSELoss, Tanh, Sigmoid, Embedding, CrossEntropyLoss, RNNCell
np.random.seed(12345)

# dataset from http://www2.aueb.gr/users/ion/data/enron-spam/

import codecs
with codecs.open('spam.txt', "r",encoding='utf-8', errors='ignore') as fdata:
    raw = fdata.readlines()

vocab = set()

spam = list()
for row in raw:
    spam.append(set(row[:-2].split(" ")))
    for word in spam[-1]:
        vocab.add(word)

with codecs.open('ham.txt', "r",encoding='utf-8', errors='ignore') as fdata:
    raw = fdata.readlines()

ham = list()
for row in raw:
    ham.append(set(row[:-2].split(" ")))
    for word in ham[-1]:
        vocab.add(word)

vocab.add("<unk>")

vocab = list(vocab)
w2i = {}
for i,w in enumerate(vocab):
    w2i[w] = i

def to_indices(input, l=500):
    indices = list()
    for line in input:
        if(len(line) < l):
            line = list(line) + ["<unk>"] * (l - len(line))
            idxs = list()
            for word in line:
                idxs.append(w2i[word])
            indices.append(idxs)
    return indices

spam_idx = to_indices(spam)
ham_idx = to_indices(ham)

train_spam_idx = spam_idx[0:-1000]
train_ham_idx = ham_idx[0:-1000]

test_spam_idx = spam_idx[-1000:]
test_ham_idx = ham_idx[-1000:]

train_data = list()
train_target = list()

test_data = list()
test_target = list()

for i in range(max(len(train_spam_idx),len(train_ham_idx))):
    train_data.append(train_spam_idx[i%len(train_spam_idx)])
    train_target.append([1])

    train_data.append(train_ham_idx[i%len(train_ham_idx)])
    train_target.append([0])

for i in range(max(len(test_spam_idx),len(test_ham_idx))):
    test_data.append(test_spam_idx[i%len(test_spam_idx)])
    test_target.append([1])

    test_data.append(test_ham_idx[i%len(test_ham_idx)])
    test_target.append([0])

def train(model, input_data, target_data, batch_size=500, iterations=5):

    criterion = MSELoss()
    optim = SGD(parameters=model.get_parameters(), alpha=0.01)

    n_batches = int(len(input_data) / batch_size)
    for iter in range(iterations):
        iter_loss = 0
        for b_i in range(n_batches):

            # padding token should stay at 0
            model.weight.data[w2i['<unk>']] *= 0
            input = Tensor(input_data[b_i*batch_size:(b_i+1)*batch_size], autograd=True)
            target = Tensor(target_data[b_i*batch_size:(b_i+1)*batch_size], autograd=True)

            pred = model.forward(input).sum(1).sigmoid()
            loss = criterion.forward(pred,target)
            loss.backward()
            optim.step()

            iter_loss += loss.data[0] / batch_size

            sys.stdout.write("\r\tLoss:" + str(iter_loss / (b_i+1)))
        print()

def test(model, test_input, test_output):

    model.weight.data[w2i['<unk>']] *= 0

    input = Tensor(test_input, autograd=True)
    target = Tensor(test_output, autograd=True)

    pred = model.forward(input).sum(1).sigmoid()
    return ((pred.data > 0.5) == target.data).mean()

model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0

# for i in range(3):
#     train(model, train_data, train_target, iterations=1)
#     print("% Correct on Test Set: " + str(test(model, test_data, test_target)*100))


bob = (train_data[0:1000], train_target[0:1000])
alice = (train_data[1000:2000], train_target[1000:2000])
sue = (train_data[2000:], train_target[2000:])


import phe
import copy

public_key, private_key = phe.generate_paillier_keypair(n_length=1024)

# encrypt the number "5"
x = public_key.encrypt(5)

# encrypt the number "3"
y = public_key.encrypt(3)

# add the two encrypted values
z = x + y

# decrypt the result
z_ = private_key.decrypt(z)
print("The Answer: " + str(z_))

model = Embedding(vocab_size=len(vocab), dim=1)
model.weight.data *= 0

# note that in production the n_length should be at least 1024
public_key, private_key = phe.generate_paillier_keypair(n_length=128)

def train_and_encrypt(model, input, target, pubkey):
    train(copy.deepcopy(model), input, target, iterations=1)

    encrypted_weights = list()
    for val in model.weight.data[:,0]:
        encrypted_weights.append(public_key.encrypt(val))
    ew = np.array(encrypted_weights).reshape(model.weight.data.shape)

    return ew

for i in range(3):
    print("\nStarting Training Round...")
    print("\tStep 1: send the model to Bob")
    bob_encrypted_model = train_and_encrypt(copy.deepcopy(model),
                                            bob[0], bob[1], public_key)

    print("\n\tStep 2: send the model to Alice")
    alice_encrypted_model = train_and_encrypt(copy.deepcopy(model),
                                              alice[0], alice[1], public_key)

    print("\n\tStep 3: Send the model to Sue")
    sue_encrypted_model = train_and_encrypt(copy.deepcopy(model),
                                            sue[0], sue[1], public_key)

    print("\n\tStep 4: Bob, Alice, and Sue send their")
    print("\tencrypted models to each other.")
    aggregated_model = bob_encrypted_model + \
                       alice_encrypted_model + \
                       sue_encrypted_model

    print("\n\tStep 5: only the aggregated model")
    print("\tis sent back to the model owner who")
    print("\t can decrypt it.")
    raw_values = list()
    for val in sue_encrypted_model.flatten():
        raw_values.append(private_key.decrypt(val))
    model.weight.data = np.array(raw_values).reshape(model.weight.data.shape)/3

    print("\t% Correct on Test Set: " + \
          str(test(model, test_data, test_target)*100))