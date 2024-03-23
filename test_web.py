import numpy as np
from neural import ActivationFunc, DifferensiateFunc, Network


net = Network([3, 6, 1])
net.activation_func = np.vectorize(ActivationFunc.sigmoid)
net.differenciate_func = np.vectorize(DifferensiateFunc.sigmoid)
net.alpha = 5
data = np.array([[0, 0, 1],
                 [0, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 ])

expectation = np.array([0, 1, 1, 0])

for iterations in range(1000):
    common_error = 0
    for i in range(len(expectation)):

        data_input = data[i]
        goal = expectation[i]

        result = net.forward(data_input)
        error = net.back_propagation(result, goal)

        common_error += error
print(common_error / len(expectation))
print(net.weights)