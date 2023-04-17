import math
import torch
import torch.nn as nn


class SolutionMLP(nn.Module):
    def __init__(self, layers, activation='tanh', mapping=None):
        super().__init__()
        if isinstance(layers, str):
            layers = [int(item) for item in layers.split(',')]

        self.in_dim = layers[0]
        self.n_states = layers[-1]
        
        self.depth = len(layers) - 1
        layer_list = []
        for i in range(self.depth):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        
        self.mlp = nn.ModuleList(layer_list)
        self.activation = choose_activation(activation)
        self.mapping = mapping

    def forward(self, t):
        n = len(self.mlp)
        if self.mapping is not None:
            t = self.mapping(t)
            
        for (i, layer) in enumerate(self.mlp):
            t = layer(t)
            if i != n-1:
                t = self.activation(t)
        return t


class LinearMult(nn.Module):
    def __init__(self, in_dim, out_dim, n_nets, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_nets = n_nets
        self.weights = nn.Parameter(torch.FloatTensor(n_nets, out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_nets, 1, out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.init_weights()
   
    def init_weights(self):
        # initialize weights and biases
        # the same as for linear layer
        # from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        fan_in = self.in_dim
        gain = nn.init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(self.weights, -bound, bound)
        
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def forward(self, x):
        if len(x.shape) < 3:
            y = torch.einsum('bi,noi->nbo', x, self.weights) + self.bias
        else:
            y = torch.einsum('nbi,noi->nbo', x, self.weights) + self.bias
        return y


class SolutionMLPMult(nn.Module):
    def __init__(self, layers, activation='tanh', n_nets=1, mapping=None):
        super().__init__()
        if isinstance(layers, str):
            layers = [int(item) for item in layers.split(',')]

        self.in_dim = layers[0]
        self.n_nets = n_nets
        self.n_states = layers[-1]
        
        self.depth = len(layers) - 1
        layer_list = []
        for i in range(self.depth):
            layer_list.append(LinearMult(layers[i], layers[i+1], n_nets))
                
        self.mlp = nn.ModuleList(layer_list)
        self.activation = choose_activation(activation)
        self.mapping = mapping

    def forward(self, t):
        if self.mapping is not None:
            t = self.mapping(t)           
        t1 = t
        
        n = len(self.mlp)
        for (i, layer) in enumerate(self.mlp):
            t1 = layer(t1)
            if i != n-1:
                t1 = self.activation(t1)

        return t1


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
    def forward(self, x):
        return (x - self.mean) / self.std


def choose_activation(name):
    if name == 'tanh':
        return torch.tanh
    elif name == 'sin':
        return torch.sin
    else:
        raise ValueError(f"Unknown non-linearity {name}")
