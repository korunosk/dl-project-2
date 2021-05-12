import math
import torch


# Base classs

class Module:
    
    def forward(self, *input):
        ''' Implements the forward pass '''
        raise NotImplementedError
    
    def backward(self, *output):
        ''' Implements the backward pass '''
        raise NotImplementedError
    
    def parameters(self):
        ''' Returns parameters list '''
        raise NotImplementedError


# Modules

class Sequential(Module):
    
    def __init__(self, *layers):
        ''' Initializes the sequential module.

        Parameters
        ----------
        list (Module)
            List of modules, ex: (Linear, Relu, Linear, Sigmoid)
        '''
        self.layers = layers
    
    def forward(self, input):
        ''' Implements the forward pass through the module.
        
        Parameters
        ----------
        input: torch.tensor
            Sample or batch of samples
        
        Returns
        -------
        torch.tensor
            The output of the module
        '''
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def backward(self, output):
        ''' Implements the backward pass through the module.

        Parameters
        ----------
        output: torch.tensor
            The output of the module
        '''
        for layer in reversed(self.layers):
            output = layer.backward(output)
        return output
    
    def parameters(self):
        ''' Returns parameters list of the module.

        Returns
        -------
        list
            Parameter list
        '''
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters


# Layers

class Linear(Module):

    def __accumulate_gradients(self, input, delta):
        ''' Helper function that accumulates the gradients
        
        Parameters
        ----------
        input: torch.tensor
            The input to the layer

        delta: torch.tensor
            The error
        '''
        self.gW += torch.matmul(input.T, delta) / input.shape[0]
        self.gb += delta.sum(dim=0) / input.shape[0]
    
    def __init__(self, d_in, d_out):
        ''' Initializes the linear layer.
        
        Parameters
        ----------
        d_in: int
            The input size

        d_out: int
            The otput size
        '''
        self.context = {}
        
        # To avoid vanishing gradients
        init_bound = 1 / math.sqrt(d_in)

        self.W = torch.empty((d_in, d_out)).uniform_(-init_bound, init_bound)
        self.b = torch.empty((1, d_out)).uniform_(-init_bound, init_bound)
        
        self.gW = torch.empty(self.W.shape).fill_(0)
        self.gb = torch.empty(self.b.shape).fill_(0)

    def forward(self, input):
        ''' Implements the forward pass of the linear layer.

        Parameters
        ----------
        input: torch.tensor
            The input to the layer

        Returns
        -------
        torch.tensor
            The result of: W * input + b
        '''
        self.context['input'] = input
        return torch.matmul(input, self.W) + self.b
    
    def backward(self, output):
        ''' Implements the backward pass of the linear layer.
        
        Also, it accumulates the gradients.

        Parameters
        ----------
        output: torch.tensor
            The output of the next layer.
        
        Returns
        -------
        torch.tenosr
            The result of: W^T * output
        '''
        input = self.context['input']
        self.__accumulate_gradients(input, output)
        return torch.matmul(output, self.W.T)

    def parameters(self):
        ''' Returns the parameter list of the linear layer.
        
        Returns
        -------
        list
            Parameter list
        '''
        return [[self.W, self.gW], [self.b, self.gb]]


# Activation functions

class Sigmoid(Module):

    def __sigmoid(self, z):
        ''' Implements the Sigmoid function '''
        return 1 / (1 + torch.exp(-z))
    
    def __sigmoid_prime(self, z):
        ''' Implements the derivative of the Sigmoid function '''
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))
    
    def __init__(self):
        self.context = {}

    def forward(self, input):
        ''' Implements the forward pass of the Sigmoid layer.

        Parameters
        ----------
        input: torch.tensor
            The input to the layer

        Returns
        -------
        torch.tensor
            The result of: Sigmoid(input)
        '''
        self.context['input'] = input
        return self.__sigmoid(input)
    
    def backward(self, output):
        ''' Implements the backward pass of the Sigmoid layer.
        
        Parameters
        ----------
        output: torch.tensor
            The output of the next layer.
        
        Returns
        -------
        torch.tenosr
            The result of: output * Sigmoid'(input)
        '''
        input = self.context['input']
        return output * self.__sigmoid_prime(input)
    
    def parameters(self):
        ''' Returns the parameter list of the Sigmoid layer.

        The list is empty for this layer.
        
        Returns
        -------
        list
            Parameter list
        '''
        return []


class Tanh(Module):

    def __tanh(self, z):
        ''' Implements the hyperbolic tangent function '''
        return (torch.exp(2 * z) - 1) / (torch.exp(2 * z) + 1)
    
    def __tanh_prime(self, z):
        ''' Implements the derivative of the hyperbolic tangent function '''
        return (1 - self.__tanh(z)) ** 2
    
    def __init__(self):
        self.context = {}

    def forward(self, input):
        ''' Implements the forward pass of the Tanh layer.

        Parameters
        ----------
        input: torch.tensor
            The input to the layer

        Returns
        -------
        torch.tensor
            The result of: Tanh(input)
        '''
        self.context['input'] = input
        return self.__tanh(input)
    
    def backward(self, output):
        ''' Implements the backward pass of the Tanh layer.
        
        Parameters
        ----------
        output: torch.tensor
            The output of the next layer.
        
        Returns
        -------
        torch.tenosr
            The result of: output * Tanh'(input)
        '''
        input = self.context['input']
        return output * self.__tanh_prime(input)
    
    def parameters(self):
        ''' Returns the parameter list of the Tanh layer.

        The list is empty for this layer.
        
        Returns
        -------
        list
            Parameter list
        '''
        return []


class Relu(Module):
    
    def __relu(self, z):
        ''' Implements the rectified linear unit function '''
        return torch.max(z, torch.empty(z.shape).fill_(0))
    
    def __relu_prime(self, z):
        ''' Implements the derivative of the rectified linear unit function '''
        return (z > 0).float()

    def __init__(self):
        self.context = {}
    
    def forward(self, input):
        ''' Implements the forward pass of the ReLU layer.

        Parameters
        ----------
        input: torch.tensor
            The input to the layer

        Returns
        -------
        torch.tensor
            The result of: ReLU(input)
        '''
        self.context['input'] = input
        return self.__relu(input)

    def backward(self, output):
        ''' Implements the backward pass of the ReLU layer.
        
        Parameters
        ----------
        output: torch.tensor
            The output of the next layer.
        
        Returns
        -------
        torch.tenosr
            The result of: output * ReLU'(input)
        '''
        input = self.context['input']
        return output * self.__relu_prime(input)
    
    def parameters(self):
        ''' Returns the parameter list of the ReLU layer.

        The list is empty for this layer.
        
        Returns
        -------
        list
            Parameter list
        '''
        return []


# Losses

class LossMSE(Module):

    def __mse(self, target, prediction):
        ''' Implements the mean squared error loss function '''
        return (target - prediction).pow(2).mean(dim=0)

    def __mse_prime(self, target, prediction):
        ''' Implements the derivative of the mean squared error loss function '''
        return -2 * (target - prediction)
    
    def __init__(self):
        self.context = {}
    
    def forward(self, target, prediction):
        ''' Implements the forward pass of the MSE loss function.

        Parameters
        ----------
        target: torch.tensor
            The target value
        prediction: torch.tensor
            The predicted value

        Returns
        -------
        torch.tensor
            The result of: MSE(target, prediction)
        '''
        self.context['target'] = target
        self.context['prediction'] = prediction
        return self.__mse(target, prediction)
    
    def backward(self):
        ''' Implements the backward pass of the MSE loss function.
        
        Returns
        -------
        torch.tenosr
            The result of: MSE'(target, prediction)
        '''
        target = self.context['target']
        prediction = self.context['prediction']
        return self.__mse_prime(target, prediction)
    
    def parameters(self):
        ''' Returns the parameter list of the MSE loss function.

        The list is empty for this function.
        
        Returns
        -------
        list
            Parameter list
        '''
        []


class LossBCE(Module):

    def __bce(self, target, prediction):
        ''' Implements the binary cross-entropy loss function '''
        return (target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction)).mean(dim=0)
    
    def __bce_prime(self, target, prediction):
        ''' Implements the derivative of the binary cross-entropy loss function '''
        return - (target / prediction - (1 - target) / (1 - prediction))

    def __init__(self):
        self.context = {}
    
    def forward(self, target, prediction):
        ''' Implements the forward pass of the BCE loss function.

        Parameters
        ----------
        target: torch.tensor
            The target value
        prediction: torch.tensor
            The predicted value

        Returns
        -------
        torch.tensor
            The result of: MSE(target, prediction)
        '''
        self.context['target'] = target
        self.context['prediction'] = prediction
        return self.__bce(target, prediction)
    
    def backward(self):
        ''' Implements the backward pass of the BCE loss function.
        
        Returns
        -------
        torch.tenosr
            The result of: BCE'(target, prediction)
        '''
        target = self.context['target']
        prediction = self.context['prediction']
        return self.__bce_prime(target, prediction)
    
    def parameters(self):
        ''' Returns the parameter list of the BCE loss function.

        The list is empty for this function.
        
        Returns
        -------
        list
            Parameter list
        '''
        []
