import numpy as np
import torch
from torch import nn
import binary as mmb


def dice_eval(logits, label):

    eps = 1e-7
    bsize, channel, height, width = label.shape

    dice_list = [0 for i in range(channel-1)]
    count_list = [0 for i in range(channel-1)]

    if torch.sum(label[:, 0, :, :]) == height * width:
        return dice_list, count_list
    else:
        predicter = nn.Softmax(dim=1)(logits)
        predicter = predicter.permute(0, 2, 3, 1)
        compact_pred = torch.argmax(predicter, dim=3)
        pred = torch.nn.functional.one_hot(compact_pred, channel)
        pred = pred.permute(0, 3, 1, 2)
        for i in range(1, channel):
            dice = 0
            inse = torch.sum(pred[:, i, :, :] * label[:, i, :, :])
            u_p = torch.sum(pred[:, i, :, :])
            u_l = torch.sum(label[:, i, :, :])
            if u_l != 0:
                count_list[i-1] = 1
                dice = 2.0 * inse / (u_p + u_l + eps)
            dice_list[i-1] = dice

    return dice_list, count_list


def _assd(logits, label):

    if np.sum(logits) > 0 and np.sum(label) > 0:
        return mmb.assd(logits, label)
        # return mmb.hd(binary_segmentation, binary_gt_label)
    else:
        return np.nan

def assd_eval(logits, label):

    assd_list = [0 for i in range(channel-1)]
    count_list = [0 for i in range(channel-1)]

    bsize, channel, _, _ = label.shape

    predicter = nn.Softmax(dim=1)(logits)
    predicter = predicter.permute(0, 2, 3, 1)
    compact_pred = torch.argmax(predicter, dim=3)
    pred = torch.nn.functional.one_hot(compact_pred, channel)
    pred = pred.permute(0, 3, 1, 2)

    pred = np.asarray(pred)
    label = np.asarray(label)

    for i in range(bsize):
        for j in range(1, channel):
            assd_tmp = _assd(pred[i, j, ...], label[i, j, ...])
            if not np.isnan(assd_tmp):
                count_list[j-1] += 1.0
                assd_list[j-1] += assd_tmp

    return [a / c for a, c in zip(assd_list,count_list)]
