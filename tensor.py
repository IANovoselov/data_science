import numpy as np


class Tensor:
    """
    Тензор
    """
    def __init__(self, data, autograd=False, creators=None, creation_op=None, _id=None):
        self.data = data
        self.creators = creators
        self.creation_op = creation_op
        self.grad = None
        self.autograd = autograd
        self.children = {}  # Количество градиентов от каждого потомка
        self.id = np.random.randint(0, 100000) if _id is None else None

        if creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        """
        Обраное распространение
        :param grad:
        :param grad_origin:
        :return:
        """
        if not self.autograd:
            return None

        if grad_origin is not None:
            if self.children[grad_origin.id] == 0:
                raise Exception('Уже было обраное распространение')
            else:
                self.children[grad_origin.id] -= 1

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if(self.creators is not None and
                (self.all_children_grads_accounted_for() or
                 grad_origin is None)):

            if(self.creation_op == "add"):
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)

            if(self.creation_op == "sub"):
                self.creators[0].backward(Tensor(self.grad.data), self)
                self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

            if(self.creation_op == "mul"):
                new = self.grad * self.creators[1]
                self.creators[0].backward(new , self)
                new = self.grad * self.creators[0]
                self.creators[1].backward(new, self)

            if(self.creation_op == "mm"):
                c0 = self.creators[0]
                c1 = self.creators[1]
                new = self.grad.mm(c1.transpose())
                c0.backward(new)
                new = self.grad.transpose().mm(c0).transpose()
                c1.backward(new)

            if(self.creation_op == "transpose"):
                self.creators[0].backward(self.grad.transpose())

            if("sum" in self.creation_op):
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.expand(dim,
                                                           self.creators[0].data.shape[dim]))

            if("expand" in self.creation_op):
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.sum(dim))

            if(self.creation_op == "neg"):
                self.creators[0].backward(self.grad.__neg__())

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, autograd=True, creators=[self, other], creation_op='add')
        return Tensor(self.data + other.data)

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1, autograd=True, creators=[self], creation_op='neg')
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):

        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)

        if(self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)

    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")

        return Tensor(self.data.transpose())

    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class SGD:
    """
    Спуск по градиенту
    """
    def __init__(self, parameters, alpha = 0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha

            if zero:
                p.grad.data *= 0


class Layer:

    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.randint(n_inputs, n_outputs)*np.sqrt(2.0/n_inputs)
        self.weights = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weights)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weights) + self.bias.expand(0, len(input.data))


