class SGD():
    
    def __init__(self, parameters, lr, momentum):
        ''' Initializes the SGD optimizer.

        Parameters
        ----------
        parameters: torch.tensor
            Models parameters

        lr: float
            Learning rage

        momentum: float
            Momentum
        '''
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.v = [0] * len(self.parameters)
    
    def zero_grad(self):
        ''' Sets the gradients of the parameters to 0 '''
        for w, g in self.parameters:
            g.fill_(0)
    
    def step(self):
        ''' Updates the model parameters '''
        for i, (w, g) in enumerate(self.parameters):
            self.v[i] = self.momentum * self.v[i] + self.lr * g
            w += -self.v[i]
