import numpy as np
from neural import ActivationFunc, DerivativeFunc, Network


net = Network([3, 5, 1])
net.build_wieghts()
net.activation_func = [ActivationFunc.tanh, ActivationFunc.sigmoid]
net.derivative_func = [DerivativeFunc.tanh, DerivativeFunc.sigmoid]
net.alpha = 7
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