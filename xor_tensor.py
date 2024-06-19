import numpy as np
from tensor import Tensor, SGD, Sequential, Linear, MSELoss, Tanh, Sigmoid, Embedding, CrossEntropyLoss

np.random.seed(0)

data = Tensor(np.array([1,2,1,2]), autograd=True)
target = Tensor(np.array([0,1,0,1]), autograd=True)


model = Sequential([Embedding(3,3), Tanh(), Linear(3,4)])
optim = SGD(parameters=model.get_parameters(), alpha=3)

for i in range(10):
    pred = model.forward(data)

    loss = CrossEntropyLoss().forward(pred, target)

    loss.backward()
    optim.step()
    print(loss)

print(pred)
