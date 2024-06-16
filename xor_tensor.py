import numpy as np
from tensor import Tensor, SGD, Sequential, Linear, MSELoss, Tanh, Sigmoid

np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)


model = Sequential(layers=[Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
optim = SGD(parameters=model.get_parameters(), alpha=3)

for i in range(10):
    pred = model.forward(data)

    loss = MSELoss().forward(pred, target)

    loss.backward()
    optim.step()
    print(loss)

print(pred)
