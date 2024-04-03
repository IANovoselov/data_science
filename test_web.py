import numpy as np
from neural import ActivationFunc, DifferensiateFunc, Network


net = Network([3, 5, 1])
net.activation_func = [ActivationFunc.sigmoid, ActivationFunc.sigmoid]
net.derivative_func = [DifferensiateFunc.sigmoid, DifferensiateFunc.sigmoid]
net.alpha = 5
data = np.array([[0, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 ])

expectation = np.array([0, 1, 1, 0])

net.batch_size = 1

net.train(data, expectation, iterations_num=300)

print(net.forward(data[0]))
print(net.forward(data[1]))
print(net.forward(data[2]))
print(net.forward(data[3]))