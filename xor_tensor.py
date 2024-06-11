import numpy as np
from tensor import Tensor, SGD

np.random.seed(1)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

w = []
w.append(Tensor(np.random.rand(2, 3), autograd=True))
w.append(Tensor(np.random.rand(3, 1), autograd=True))

optim = SGD(parameters=w, alpha=0.1)

for i in range(10):
     pred = data.mm(w[0]).mm(w[1])

     loss = ((pred-target)*(pred-target)).sum(0)

     loss.backward(Tensor(np.ones_like(loss.data)))

     optim.step()

     print(loss)

pred = data.mm(w[0]).mm(w[1])