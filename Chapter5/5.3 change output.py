# -*- coding: utf-8 -*—
# Date: 2022/3/18 0018
# Time: 21:45
# Author: HQR
import torch
import torchvision.models as models
from collections import OrderedDict
import torch.nn as nn

net = models.resnet50()
print(net)


'''
1. 将模型（net）最后名称为“fc”的层替换成了名称为“classifier”的结构
'''
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                                        ('relu1', nn.ReLU()),
                                        ('dropout1', nn.Dropout(0.5)),
                                        ('fc2', nn.Linear(128, 10)),
                                        ('output', nn.Softmax(dim=1))
                                        ]))

net.fc = classifier
print(net)


'''
2. 通过torch.cat实现了tensor的拼接;
修改forward函数（配套定义一些层），先将2048维的tensor通过激活函数层和dropout层，
再和外部输入变量"add_variable"拼接，最后通过全连接层映射到指定的输出维度10。
'''
class Model(nn.Module):

    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)

    def forward(self, x, add_variable):
        x = self.net(x)
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)), 1)
        x = self.fc_add(x)
        x = self.output(x)
        return x


model = Model(net).cuda()
# input 和 add_var 是需要自己进行设置
# outputs = model(inputs, add_var)
