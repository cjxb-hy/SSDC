import os
import random
import numpy as np
import torch
from torch import nn
multi_channel = 1


class ConvIn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvIn, self).__init__()

        self.convin = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding),
            # nn.BatchNorm2d(out_channels)
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, input):

        output = self.convin(input)

        return output


class Resblk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblk, self).__init__()

        self.conv0 = ConvIn(in_channels, out_channels)
        self.conv1 = ConvIn(out_channels, out_channels)
        self.conv2 = ConvIn(out_channels, out_channels)

        self.extra = nn.Sequential()
        if in_channels != out_channels:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, padding=0)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):

        x = self.conv0(input)
        x = self.conv1(x)
        x = self.conv2(x)

        y = self.extra(input)
        output = torch.add(x, y)

        output = self.relu(output)

        return output


class DomUni(nn.Module):
    def __init__(self, n_channels):
        super(DomUni, self).__init__()

        self.stem = nn.Sequential(
            ConvIn(n_channels, 16),
            nn.ReLU(inplace=True),
        )

        self.resblk1 = Resblk(in_channels=16, out_channels=32)
        self.resblk2 = Resblk(in_channels=32, out_channels=64)

        self.oput = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=n_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):

        x = self.stem(input)
        x = self.resblk1(x)
        x = self.resblk2(x)
        x = self.oput(x)

        output = torch.mul(input, x)

        return output


class DomUniS(nn.Module):
    def __init__(self, n_channels):
        super(DomUniS, self).__init__()

        self.stem = nn.Sequential(
            ConvIn(n_channels, 16),
            nn.ReLU(inplace=True),
        )

        self.resblk1 = Resblk(in_channels=16, out_channels=32)

        self.oput = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=n_channels,
                      kernel_size=1, stride=1, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):

        x = self.stem(input)
        x = self.resblk1(x)
        output = self.oput(x)

        # output = torch.mul(input, x)

        return output


if __name__ == '__main__':

    net = DomUni(1)
    # print(net)
    x = torch.randn([2, 1, 512, 512])
    output = net(x)
    print(output.shape)
