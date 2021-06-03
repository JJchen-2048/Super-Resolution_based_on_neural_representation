import torch.nn as nn
import torch
from models import register
import numpy as np

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.
    # omega_0是超参数

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    # 有关于class的用法，SineLayer里面有nn.Module，相当于是继承nn.Module
    # super().__init__()是继承nn.Module的初始化，__init__是初始化，在刚创建class时候运行

    # 有关于nn.Module的使用，一般把具有可学习参数的层放在构造函数__init__()里
    # 不具有可学习的层(比如说是ReLU)放在构造函数中
    # 但是这里，只是写了个Sine层
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        # pytorch中的nn.Linear是用来设置网络中的全连接层
        self.linear = nn.Linear(in_features, out_features, bias=bias)

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

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))



@register('siren1')
class SIREN1(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list,first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        layers = []
        layers.append(SineLayer(in_dim, hidden_list[0], is_first=False, omega_0=first_omega_0))
        lastv = hidden_list[0]
        for hidden in hidden_list:
            # 初始化过程
            layers.append(nn.Linear(lastv, hidden))
            # 换一下
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)