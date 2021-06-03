import torch.nn as nn
import torch
from models import register
import numpy as np
import torch.nn.functional as F

def exists(val):
    return val is not None
    
# sine层
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        # pytorch中的nn.Linear是用来设置网络中的全连接层
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.cuda()
        self.init_weights()

    def init_weights(self):
        # torch.no_grad()是一个上下文管理器，被该语句wrap的部分不会被track梯度
        # 对应的操作，看论文，即对权重进行一个操作，是第一层怎样，不是第一层怎样
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input, gamma=None, beta=None):

        out = self.linear(input)
        # FiLM modulation
        if exists(gamma):
            out = out * gamma
        if exists(beta):
            out = out + beta

        out = self.omega_0 * out

        return torch.sin(out)

# 参数生成器
def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=0.1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.weight.cuda()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
            self.bias.cuda()

        self.lr_mul = lr_mul

    def forward(self, input):
        y = F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        return y


class MappingNetwork(nn.Module):
    def __init__(self, dim, dim_out, depth=3, lr_mul=0.1):
        super().__init__()
        layers = []
        # 这里用的是激活函数是leaky_relu()
        for i in range(depth):
            layers.extend([EqualLinear(dim, dim, lr_mul), leaky_relu()])
        self.net = nn.Sequential(*layers)

        self.to_gamma = nn.Linear(dim, dim_out)
        self.to_beta = nn.Linear(dim, dim_out)

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        x = self.net(x)
        return self.to_gamma(x), self.to_beta(x)

# siren网络
class SirenNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list,first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        # 生成siren层
        layers = []
        lastv = in_dim
        for idx, hidden in enumerate(hidden_list):
            # 初始化过程
            layers.append(SineLayer(lastv, hidden, is_first=idx==0, omega_0=hidden_omega_0))
            lastv = hidden
        self.layers = nn.Sequential(*layers)
        self.last_layer = nn.Linear(lastv, out_dim)
        # self.layers = nn.Sequential(*layers)

    def forward(self, x, gamma, beta):
        for layer in self.layers:
            x = layer(x, gamma, beta)
        return self.last_layer(x)

# 总网络
@register('siren_with_condition')
class SIREN_with_condition(nn.Module):
    def __init__(self, in_dim, out_dim, condition_dim, hidden_list):
        super().__init__()
        # 生成参数生成器
        self.mapping = MappingNetwork(dim = condition_dim, dim_out = hidden_list[0])
        # 生成SirenNet
        self.siren = SirenNet(in_dim = in_dim, hidden_list = hidden_list, out_dim = out_dim)

    def forward(self, latent, coors):
        # 这里生成了siren中所用的参数
        # 参数生成器
        
        gamma, beta = self.mapping(latent.view(-1, latent.shape[-1]))
        shape = coors.shape[:-1]
        x = self.siren(coors.view(-1, coors.shape[-1]),gamma,beta)
        return x.view(*shape, -1)

