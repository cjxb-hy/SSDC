import torch
from torch import nn


def loop_iterable(iterable):

    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):

    for param in model.parameters():
        param.requires_grad = requires_grad


def load_model(net, model_path):

    save_model = torch.load(model_path)
    model_dict = net.state_dict()
    state_dict = {k: v for k, v in save_model.items()
                  if k in model_dict.keys()}
    model_dict.update(state_dict)
    return model_dict


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
