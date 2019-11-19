# -*- coding:gbk -*-
# 设置源文件编码格式为:gbk
# -*- coding:utf-8 -*-
# 设置源文件编码格式为:UTF-8

import torch
import torch.nn as nn
import numpy as np
import matplotlib
from torch import optim
from torch.autograd import Variable
from torch.version import cuda


def make_feature(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1,4)], 1)

W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.FloatTensor([0.9])

def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]
# mm表示做矩阵乘法
def get_batch(batch_size = 32):
    """Builds a batch i.e. (x,f(x)) pair"""
    random = torch.randn(batch_size)
    x = make_feature(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)


# Define model
class ploy_model(nn.Module):
    def __init__(self):
        super(ploy_model, self).__init__()
        self.ploy = nn.Linear(3,1)

    def forward(self, x):
        out = self.ploy(x)
        return out

if torch.cuda.is_available():
    model = ploy_model().cuda()
else:
    model = ploy_model()

# Define loss function and opt
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

epoch = 0
while True:
    # Get data
    batch_x, batch_y = get_batch()
    # Forward pass
    output = model(batch_x)
    loss = criterion(output, batch_y)
    # print_loss = loss.data[0]
    print_loss = loss.item()
    # Reset gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # update parameters
    optimizer.step()
    epoch += 1
    print(print_loss)
    if print_loss< 1e-3:
        break
