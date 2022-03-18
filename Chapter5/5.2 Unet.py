# 5.2 Unet

## 5.2.1 U-net简介

- U-Net是分割 (Segmentation) 模型的杰作，在以医学影像为代表的诸多领域有着广泛的应用。U-Net模型结构如下图所示，通过残差连接结构解决了模型学习中的退化问题，使得神经网络的深度能够不断扩展。

- 不难发现U-Net模型具有非常好的对称性。模型从上到下分为若干层，每层由左侧和右侧两个模型块组成，每侧的模型块与其上下模型块之间有连接；同时位于同一层左右两侧的模型块之间也有连接，称为“Skip-connection”。此外还有输入和输出处理等其他组成部分。由于模型的形状非常像英文字母的“U”，因此被命名为“U-Net”。

- 组成U-Net的模型块主要有如下几个部分：

1）每个子块内部的两次卷积（Double Convolution）

2）左侧模型块之间的下采样连接，即最大池化（Max pooling）

3）右侧模型块之间的上采样连接（Up sampling）

4）输出层的处理

除模型块外，还有模型块之间的横向连接，输入和U-Net底部的连接等计算，这些单独的操作可以通过forward函数来实现。

下面我们用PyTorch先实现上述的模型块，然后再利用定义好的模型块构建U-Net模型，用于U-net的配准。

```python
# -*- coding: utf-8 -*—
# Date: 2022/3/16 0016
# Time: 23:07
# Author: HQR
""" Parts of the U-Net model, name:unet_model.py """

import torch
from torch.nn import functional as F

class CNNLayer(torch.nn.Module):
    '''
    卷积层
    :param C_in:
    :param C_out:
    '''
    def __init__(self, C_in, C_out):

        super(CNNLayer, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C_in, C_out, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(C_out, C_out, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            torch.nn.BatchNorm2d(C_out),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(torch.nn.Module):
    '''
    下采样
    :param C:
    '''
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(C, C,kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):

        return self.layer(x)


class UpSampling(torch.nn.Module):
    '''
    上采样
    :param C:
    '''
    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.C = torch.nn.Conv2d(C, C // 2, kernel_size=(1,1), stride=(1,1))

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.C(up)
        return torch.cat((x, r), 1)
```

```python
# -*- coding: utf-8 -*—
# Date: 2022/3/16 0016
# Time: 22:38
# Author: HQR
import torch
from unet_model import CNNLayer, DownSampling, UpSampling

class Unet(torch.nn.Module):

    def __init__(self):
        super(Unet, self).__init__()

        self.C1 = CNNLayer(3, 64)
        self.D1 = DownSampling(64)
        self.C2 = CNNLayer(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = CNNLayer(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = CNNLayer(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = CNNLayer(512, 1024)
        self.U1 = UpSampling(1024)
        self.C6 = CNNLayer(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = CNNLayer(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = CNNLayer(256, 128)
        self.U4 = UpSampling(128)


        self.C9 = CNNLayer(128, 64)
        self.pre = torch.nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        '''
        U型结构
        :param x:
        :return:
        '''
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        return self.sigmoid(self.pre(O4))


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256) #.cuda()
    net = Unet() #.cuda()
    print(net(a).shape)
```

```
torch.Size([2, 3, 256, 256])

Process finished with exit code 0
```

