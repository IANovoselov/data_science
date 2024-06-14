import numpy as np
from tensor import Tensor, SGD, Sequential, Linear, MSELoss

np.random.seed(1)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)


model = Sequential(layers=[Linear(2, 3), Linear(3, 1)])
optim = SGD(parameters=model.get_parameters(), alpha=0.05)

for i in range(10):
    pred = model.forward(data)

    loss = MSELoss().forward(pred, target)

    loss.backward()

    optim.step()

    print(loss)

pred = model.forward(data)
print(pred)