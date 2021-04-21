import math
import torch


class Module:
    
    def forward(self, *input):
        raise NotImplementedError
    
    def backward(self, *output):
        raise NotImplementedError
    
    def parameters(self):
        raise NotImplementedError


class Sequential(Module):
    
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, output):
        for layer in reversed(self.layers):
            output = layer.backward(output)
        return output
    
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters


class Linear(Module):

    def __accumulate_gradients(self, input, delta):
        self.gW += torch.matmul(input.T, delta) / input.shape[0]
        self.gb += delta.sum(dim=0) / input.shape[0]
    
    def __init__(self, d_in, d_out):
        self.context = {}
        
        init_bound = 1 / math.sqrt(d_in)
        self.W = torch.empty((d_in, d_out)).uniform_(-init_bound, init_bound)
        self.b = torch.empty((1, d_out)).uniform_(-init_bound, init_bound)
        
        self.gW = torch.empty(self.W.shape).fill_(0)
        self.gb = torch.empty(self.b.shape).fill_(0)

    def forward(self, input):
        self.context['input'] = input
        return torch.matmul(input, self.W) + self.b
    
    def backward(self, output):
        input = self.context['input']
        self.__accumulate_gradients(input, output)
        return torch.matmul(output, self.W.T)

    def parameters(self):
        return [[self.W, self.gW], [self.b, self.gb]]


class Sigmoid(Module):

    def __sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
    
    def __sigmoid_prime(self, z):
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))
    
    def __init__(self):
        self.context = {}

    def forward(self, input):
        self.context['input'] = input
        return self.__sigmoid(input)
    
    def backward(self, output):
        input = self.context['input']
        return output * self.__sigmoid_prime(input)
    
    def parameters(self):
        return []


class Tanh(Module):

    def __tanh(self, z):
        return (torch.exp(2 * z) - 1) / (torch.exp(2 * z) + 1)
    
    def __tanh_prime(self, z):
        return (1 - self.__tanh(z)) ** 2
    
    def __init__(self):
        self.context = {}

    def forward(self, input):
        self.context['input'] = input
        return self.__tanh(input)
    
    def backward(self, output):
        input = self.context['input']
        return output * self.__tanh_prime(input)
    
    def parameters(self):
        return []


class Relu(Module):
    
    def __relu(self, z):
        return torch.max(z, torch.empty(z.shape).fill_(0))
    
    def __relu_prime(self, z):
        return (z > 0).float()

    def __init__(self):
        self.context = {}
    
    def forward(self, input):
        self.context['input'] = input
        return self.__relu(input)

    def backward(self, output):
        input = self.context['input']
        return output * self.__relu_prime(input)
    
    def parameters(self):
        return []


class LossMSE(Module):

    def __mse(self, target, prediction):
        return (target - prediction).pow(2).mean(dim=0)

    def __mse_prime(self, target, prediction):
        return -2 * (target - prediction)
    
    def __init__(self):
        self.context = {}
    
    def forward(self, target, prediction):
        self.context['target'] = target
        self.context['prediction'] = prediction
        return self.__mse(target, prediction)
    
    def backward(self):
        target = self.context['target']
        prediction = self.context['prediction']
        return self.__mse_prime(target, prediction)
    
    def parameters(self):
        []


class LossBCE(Module):

    def __bce(self, target, prediction):
        return (target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction)).mean()
    
    def __bce_prime(self, target, prediction):
        return - (target / prediction - (1 - target) / (1 - prediction))

    def __init__(self):
        self.context = {}
    
    def forward(self, target, prediction):
        self.context['target'] = target
        self.context['prediction'] = prediction
        return self.__bce(target, prediction)
    
    def backward(self):
        target = self.context['target']
        prediction = self.context['prediction']
        return self.__bce_prime(target, prediction)
    
    def parameters(self):
        []
