import torch
from torch import nn


class Generalized_Dice(nn.Module):
    def __init__(self):
        super(Generalized_Dice, self).__init__()

    def generalized_dice_coeff(self, logits, label):

        dice = 0
        eps = 1e-7

        _, channel, _, _ = label.shape

        softmaxpred = nn.Softmax(dim=1)(logits)

        for i in range(channel):
            w = torch.sum(label[:, i, :, :], dim=[1, 2])
            w = 1 / (w**2 + eps)
            inse = w * torch.sum(softmaxpred[:, i, :, :]
                                 * label[:, i, :, :], dim=[1, 2])
            l = w * torch.sum(softmaxpred[:, i, :, :] *
                              softmaxpred[:, i, :, :], dim=[1, 2])
            r = w * torch.sum(label[:, i, :, :]
                              * label[:, i, :, :], dim=[1, 2])

            dice += 2.0 * inse / (l + r + eps)

        return torch.mean(dice) / channel

    def forward(self, logits, label):
        return 1 - self.generalized_dice_coeff(logits, label)


class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()

    def cross_entropy(self, logits, label):

        log_output = nn.LogSoftmax(dim=1)(logits)
        nll = - log_output * label.float()
        loss = torch.mean(torch.sum(nll, dim=1))

        return loss

    def forward(self, logits, label):
        return self.cross_entropy(logits, label)


class Dice_Loss(nn.Module):
    def __init__(self):
        super(Dice_Loss, self).__init__()

    def dice_loss(self, logits, label):

        dice = 0
        eps = 1e-7

        bsize, channel, height, width = label.shape
        softmaxpred = nn.Softmax(dim=1)(logits)

        for i in range(channel):
            inse = torch.sum(softmaxpred[:, i, :, :] * label[:, i, :, :])
            l = torch.sum(softmaxpred[:, i, :, :] * softmaxpred[:, i, :, :])
            r = torch.sum(label[:, i, :, :] * label[:, i, :, :])
            dice += (2.0 * inse + eps) / (l + r + eps)

        return torch.mean(dice) / channel

    def forward(self, logits, label):
        return 1 - self.dice_loss(logits, label)


class Cycle_Consistency(nn.Module):
    def __init__(self):
        super(Cycle_Consistency, self).__init__()

    def cycle_consistency(self, ori, cyc):

        loss = torch.mean(torch.abs(ori-cyc))

        return loss

    def forward(self, ori, cyc):
        return self.cycle_consistency(ori, cyc)


class Dis_Loss(nn.Module):
    def __init__(self):
        super(Dis_Loss, self).__init__()

    def discrimintor_loss(self, ori, cyc):

        loss = torch.mean(torch.square(ori-cyc))

        return loss

    def forward(self, ori, cyc):
        return self.discrimintor_loss(ori, cyc)