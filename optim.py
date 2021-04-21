class SGD():
    
    def __init__(self, parameters, lr, momentum):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.v = [0] * len(self.parameters)
    
    def zero_grad(self):
        for w, g in self.parameters:
            g.fill_(0)
    
    def step(self):
        for i, (w, g) in enumerate(self.parameters):
            self.v[i] = self.momentum * self.v[i] + self.lr * g
            w += -self.v[i]
