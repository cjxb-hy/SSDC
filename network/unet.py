from time import time
import torch
from torch import nn


class WeightLearning(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WeightLearning, self).__init__()

        self.conv1 = nn.Conv2d(2 * in_channels, out_channels,
                               kernel_size=3, stride=2, padding=1)

        # self.in1 = nn.InstanceNorm2d(out_channels)

        # self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=2, padding=1)

        self.adapool = nn.AdaptiveAvgPool2d([1, 1])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        # x = self.in1(x)
        # x = self.relu1(x)
        x = self.conv2(x)
        x = self.adapool(x)
        x = self.sigmoid(x)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.weights1 = WeightLearning(out_channels, out_channels)
        self.weights2 = WeightLearning(out_channels, out_channels)

        self.loss = Cycle_Consistency()

    def forward(self, x1, x2):

        x1, x2 = self.conv1(x1), self.conv1(x2)
        w1 = self.weights1(x1, x2)
        x1, x2 = w1*x1, w1*x2
        loss1 = self.loss(x1, x2)
        x1, x2 = self.bn1(x1), self.bn1(x2)
        x1, x2 = self.relu1(x1), self.relu1(x2)

        x1, x2 = self.conv2(x1), self.conv2(x2)
        w2 = self.weights2(x1, x2)
        x1, x2 = w2*x1, w2*x2
        loss2 = self.loss(x1, x2)
        x1, x2 = self.bn2(x1), self.bn2(x2)
        x1, x2 = self.relu2(x1), self.relu2(x2)

        return x1, x2, (loss1 + loss2) / 2


class Cycle_Consistency(nn.Module):
    def __init__(self):
        super(Cycle_Consistency, self).__init__()

    def cycle_consistency(self, ori, cyc):

        loss = torch.mean(torch.abs(ori-cyc))

        return loss

    def forward(self, ori, cyc):
        return self.cycle_consistency(ori, cyc)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dconv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1, x2 = self.pool(x1), self.pool(x2)
        x1, x2, lossd = self.dconv(x1, x2)

        return x1, x2, lossd


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.dconv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, x3, x4):

        x1, x3 = self.up(x1), self.up(x3)

        x12 = torch.cat([x1, x2], dim=1)
        x34 = torch.cat([x3, x4], dim=1)

        x12, x34, lossd = self.dconv(x12, x34)

        return x12, x34, lossd


class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Output, self).__init__()

        self.output = nn.Conv2d(in_channels, out_channels,
                                kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):

        x1 = self.output(x1)
        x2 = self.output(x2)

        return x1, x2


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, scale='small'):
        super(UNet, self).__init__()

        dict_ = {'small': 1, 'middle': 2, 'large': 4}

        self.scale = scale
        self.rate = dict_[self.scale]

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.input_conv = DoubleConv(n_channels, 16*self.rate)
        self.down1 = Down(16*self.rate, 32*self.rate)
        self.down2 = Down(32*self.rate, 64*self.rate)
        self.down3 = Down(64*self.rate, 128*self.rate)
        self.down4 = Down(128*self.rate, 256*self.rate)
        self.up1 = Up(256*self.rate, 128*self.rate)
        self.up2 = Up(128*self.rate, 64*self.rate)
        self.up3 = Up(64*self.rate, 32*self.rate)
        self.up4 = Up(32*self.rate, 16*self.rate)
        self.output = Output(16*self.rate, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, xo, xp):

        xo1, xp1, loss1 = self.input_conv(xo, xp)
        xo2, xp2, loss2 = self.down1(xo1, xp1)
        xo3, xp3, loss3 = self.down2(xo2, xp2)
        xo4, xp4, loss4 = self.down3(xo3, xp3)
        xo5, xp5, loss5 = self.down4(xo4, xp4)
        xo6, xp6, loss6 = self.up1(xo5, xo4, xp5, xp4)
        xo7, xp7, loss7 = self.up2(xo6, xo3, xp6, xp3)
        xo8, xp8, loss8 = self.up3(xo7, xo2, xp7, xp2)
        xo9, xp9, loss9 = self.up4(xo8, xo1, xp8, xp1)
        output1, output2 = self.output(xo9, xp9)
        loss = (loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9)/9

        return output1, output2, loss


if __name__ == "__main__":
    n_channels = 4
    n_classes = 3

    device = torch.device('cuda')

    net = UNet(n_channels, n_classes).to(device)
    x1 = torch.randn(2, 4, 128, 128).to(device)
    x2 = torch.randn(2, 4, 128, 128).to(device)

    x1, x2 = net(x1, x2)
    # x2 = x2.flatten(2)

    print('output shape:', x1.shape, x2.shape)
