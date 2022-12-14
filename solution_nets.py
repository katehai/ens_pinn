import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import choose_nonlinearity


class CompositeMapping(nn.Module):
    def __init__(self, mappings):
        super().__init__()
        self.mappings = nn.Sequential(*mappings)
        
    def forward(self, x):
        y = self.mappings(x)
        return y


class FourierMapping(nn.Module):
    def __init__(self, in_dim=1, n_feat=16, sigma_b=1.):
        super().__init__()
        self.n_feat = n_feat
        self.sigma_b = sigma_b
        b = sigma_b * torch.randn(n_feat, in_dim)
        self.register_buffer('b', b)
        
    def forward(self, v):
        pe = torch.zeros(v.shape[0], 2 * self.n_feat, device=v.device)
        f = 2 * math.pi * torch.mm(v, self.b.T)
        pe[:, 0::2] = torch.sin(f)
        pe[:, 1::2] = torch.cos(f)
        return pe
    
    
class SolutionMLP(nn.Module):
    def __init__(self, layers, nonlin='tanh', mapping=None):
        super().__init__()
        self.in_dim = layers[0]
        self.n_states = layers[-1]
        
        self.depth = len(layers) - 1
        layer_list = []
        for i in range(self.depth):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        
        self.mlp = nn.ModuleList(layer_list)
        self.nonlin = choose_nonlinearity(nonlin)
        self.mapping = mapping

    def forward(self, t):
        n = len(self.mlp)
        if self.mapping is not None:
            t = self.mapping(t)
            
        for (i, layer) in enumerate(self.mlp):
            t = layer(t)
            if i != n-1:
                t = self.nonlin(t)
        return t


class LinearSiren(nn.Module):
    def __init__(self, in_dim, out_dim, w0, w0_mult, bias=True, is_first=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_first = is_first
        self.w0 = w0
        self.w0_mult = w0_mult
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)        
        self.init_weights()
            
    def init_weights(self):
        # w0 division is used to scale the gradient and to improve the training speed
        fan_in = self.in_dim
        
        if self.is_first:
            bound = 1. / fan_in
        else:
            bound = math.sqrt(6. / fan_in) / self.w0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        
        # bias initialization is the same as in pythorch linear
        
    def forward(self, x):
        return self.w0_mult * self.linear(x)


class SolutionMLPsiren(nn.Module):
    def __init__(self, n_states=2, h_dim=50, in_dim=1,
                 mapping=None, w0=30., w0_last=30.):
        super().__init__()
        n = h_dim
        self.mlp = nn.ModuleList([
            LinearSiren(in_dim, n, w0=w0, w0_mult=w0, is_first=True),
            LinearSiren(n, n, w0=w0, w0_mult=w0),
            LinearSiren(n, n, w0=w0, w0_mult=w0),
            LinearSiren(n, n, w0=w0, w0_mult=w0),
            LinearSiren(n, n_states, w0=w0, w0_mult=w0_last)] 
        )
        self.nonlin = torch.sin
        self.mapping = mapping

    def forward(self, t):
        n = len(self.mlp)
        if self.mapping is not None:
            t = self.mapping(t)
            
        for (i, layer) in enumerate(self.mlp):
            t = layer(t)
            if i != n-1:
                t = self.nonlin(t)
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
            print("Init uniform")
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)  # bias init
        
    def forward(self, x):
        if len(x.shape) < 3:
            y = torch.einsum('bi,noi->nbo', x, self.weights) + self.bias
        else:
            y = torch.einsum('nbi,noi->nbo', x, self.weights) + self.bias
        return y


class SolutionMLPMult(nn.Module):
    def __init__(self, layers, nonlin='tanh', n_nets=1, mapping=None):
        super().__init__()
        self.in_dim = layers[0]
        self.n_nets = n_nets
        self.n_states = layers[-1]
        
        self.depth = len(layers) - 1
        layer_list = []
        for i in range(self.depth):
            layer_list.append(LinearMult(layers[i], layers[i+1], n_nets))
                
        self.mlp = nn.ModuleList(layer_list)
        self.nonlin = choose_nonlinearity(nonlin)
        self.mapping = mapping
        
    def forward(self, t):
        if self.mapping is not None:
            t = self.mapping(t)           
        t1 = t
        
        n = len(self.mlp)
        for (i, layer) in enumerate(self.mlp):
            t1 = layer(t1)
            if i != n-1:
                t1 = self.nonlin(t1)

        return t1
    
class SolutionMLPDropout(nn.Module):
    def __init__(self, layers, nonlin='tanh', n_nets=1, mapping=None):
        super().__init__()
        self.in_dim = layers[0]
        self.n_nets = n_nets
        self.n_states = layers[-1]
        
        self.depth = len(layers) - 1
        layer_list = []
        dropout_list = []
        for i in range(self.depth):  
            # add just linear layer
            layer = nn.Linear(layers[i], layers[i+1])            
            layer_list.append(layer)
            if i != self.depth - 1:
                dropout_list.append(nn.Dropout(p=0.01))
                
        self.mlp = nn.ModuleList(layer_list)
        self.dropout = nn.ModuleList(dropout_list)
        self.nonlin = choose_nonlinearity(nonlin)
        self.mapping = mapping
        
    def forward(self, t):
        # we make n_nets predictions simulating predictions made by an ensemble
        # print("Training mode is ", self.training)
        if self.training:
            in_size = t.shape[-1]
            t = t.repeat(self.n_nets, 1, 1).reshape(-1, in_size) # increase batch size 5 times
        if self.mapping is not None:
            t = self.mapping(t)  
        t1 = t
        
        n = len(self.mlp)
        for (i, layer) in enumerate(self.mlp):
            t1 = layer(t1)
            if i != n-1:
                t1 = self.dropout[i](t1)
                t1 = self.nonlin(t1)

        # reshape results back
        if self.training:
            t1 = t1.reshape(self.n_nets, -1, self.n_states)
        return t1


class LinearMultSiren(LinearMult):
    def __init__(self, in_dim, out_dim, n_nets, bias=True, w0=30., w0_mult=30,
                 is_first=False):
        self.is_first = is_first
        self.w0 = w0
        self.w0_mult = w0_mult
        super().__init__(in_dim, out_dim, n_nets, bias)
   
    def init_weights(self):
        # initialize weights and biases
        fan_in = self.in_dim
        
        if self.is_first:
            bound = 1. / fan_in
        else:
            bound = 1. / self.w0 * math.sqrt(6. / fan_in) 
        nn.init.uniform_(self.weights, -bound, bound)
        
        if self.bias is not None:
            bound = 1. / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        if len(x.shape) < 3:
            y = torch.einsum('bi,noi->nbo', x, self.weights) + self.bias
        else:
            y = torch.einsum('nbi,noi->nbo', x, self.weights) + self.bias
            
        y = self.w0_mult * y
        return y
            
        
class SolutionMLPMultSiren(nn.Module):
    def __init__(self, n_states=2, h_dim=50, in_dim=1, n_nets=1, w0=30.,
                 w0_last=30., mapping=None):
        super().__init__()
        n = h_dim
        self.in_dim = in_dim
        self.n_nets = n_nets
        self.n_states = n_states
        self.mlp = nn.ModuleList([
            LinearMultSiren(in_dim, n, n_nets, w0=w0, w0_mult=w0, is_first=True),
            LinearMultSiren(n, n, n_nets, w0=w0, w0_mult=w0),
            LinearMultSiren(n, n, n_nets, w0=w0, w0_mult=w0),
            LinearMultSiren(n, n, n_nets, w0=w0, w0_mult=w0),
            LinearMultSiren(n, n_states, n_nets, w0=w0, w0_mult=w0_last)] 
        )
        self.nonlin = torch.sin
        self.mapping = mapping

    def forward(self, t):
        if self.mapping is not None:
            t = self.mapping(t)
            
        t1 = t
        n = len(self.mlp)
        for (i, layer) in enumerate(self.mlp):
            t1 = layer(t1)
            if i != n-1:
                t1 = self.nonlin(t1)
        
        return t1

    
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
    def forward(self, x):
        return (x - self.mean) / self.std
