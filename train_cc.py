import os
import random
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from network.unet import UNet
from network.style import DomUni, DomUniS
from utlis.datasetbmp import Data_Loader
from utlis.datasetmulti import Multi_Data_Loader
from utlis.loss import Generalized_Dice, Cycle_Consistency
from utlis.lib import load_model, dice_eval, set_requires_grad


def main():

    lr = 1e-4
    num_cls = 4
    epochs = 50
    in_channels = 1
    batch_size = 4
    device = torch.device('cuda:2')

    train_path = ['/home/niusijie/drdu_cycle/fastmri/Cirrus/brain', '/home/niusijie/drdu_cycle/heart/Cirrus/ct',
                  '/home/niusijie/drdu_cycle/heart/Cirrus/mr', '/home/niusijie/drdu_cycle/luna/Cirrus/lung_1',
                  '/home/niusijie/drdu_cycle/luna/Cirrus/lung_2', '/home/niusijie/all_data/Cirrus_train/oct']

    ori_train_path = '/home/niusijie/all_data/Cirrus_train'

    train_set = DataLoader(dataset=Multi_Data_Loader(train_path, ori_train_path),
                           batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    seg_net = UNet(in_channels, num_cls, 'middle').to(device)
    style_net = DomUniS(in_channels).to(device)

    criterion = Generalized_Dice().to(device)
    cycle_criterion = Cycle_Consistency().to(device)
    optimizer = optim.Adam([{'params': seg_net.parameters()}, {
                           'params': style_net.parameters()}], lr=lr, betas=(0.9, 0.99))

    best_dice = 0.2

    for epoch in range(1, epochs+1):

        seg_net.train()
        style_net.train()

        # train
        for batch, (_, ori_image, image, label) in enumerate(train_set):

            optimizer.zero_grad()

            image, label = image.to(device), label.to(device)
            ori_image = ori_image.to(device)

            du_image = style_net(image)
            output, outputp, _ = seg_net(ori_image, du_image)

            loss = 0.5 * criterion(output, label) + 0.5 * \
                criterion(outputp, label)
            cycle_loss = cycle_criterion(du_image, ori_image)

            total_loss = loss + 10 * cycle_loss

            total_loss.backward()
            optimizer.step()

            if batch % 20 == 0 and batch != 0:
                print('Epoch: {:3}, batch: {:3}, c_loss: {:.10f}, cycle_loss: {:.10f}'.format(
                    epoch, batch, loss.item(), cycle_loss.item()))

        # save model
        path = './models/Cirrus/Cirrus_unet_middle.pth'
        torch.save(seg_net.state_dict(), path)
        print('Epoch: {:3}, save model:{}'.format(epoch, path))

        path = './models/Cirrus/Cirrus_unet_middle_style.pth'
        torch.save(style_net.state_dict(), path)
        print('Epoch: {:3}, save model:{}'.format(epoch, path))


if __name__ == "__main__":

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main()
