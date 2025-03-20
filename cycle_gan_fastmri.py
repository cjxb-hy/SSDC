import os
import random
import numpy as np

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from utlis.dataset import Data_Loader
from utlis.loss import Cycle_Consistency
from utlis.lib import set_requires_grad, loop_iterable, show_img

from network.generator import Im_Generator
from network.reconstructor import Im_Reconstructor
from network.discriminator import Image_Discriminator


def seed_(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():

    seed_(0)

    g_lr = 1e-4
    d_lr = 2.5e-5
    epochs = 100
    iter = 200
    channels = 1
    batch_size = 16
    load_model_flag = False
    device = torch.device('cuda:1')

    model_fold = './models/Cirrus/fastmri/'
    
    source = DataLoader(dataset=Data_Loader('/mnt/sda/xhli/all_data/Cirrus_train/oct'),
                        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    target = DataLoader(dataset=Data_Loader('/mnt/sda/xhli/drdu_data/fastmri/brain_bmp'),
                        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    save_fold = '/mnt/sda/xhli/drdu_cycle/fastmri/Cirrus/result_c'

    recon_net = Im_Reconstructor(channels).to(device)
    generator = Im_Generator(channels).to(device)
    st_im_discriminator = Image_Discriminator(channels).to(device)
    ts_im_discriminator = Image_Discriminator(channels).to(device)

    criterion_c = nn.BCEWithLogitsLoss().to(device)
    cycle_criterion = Cycle_Consistency().to(device)

    recon_optimizer = optim.Adam(recon_net.parameters(), lr=g_lr)
    generator_optimizer = optim.Adam(generator.parameters(), lr=g_lr)
    st_im_discriminator_optimizer = optim.Adam(
        st_im_discriminator.parameters(), lr=d_lr)
    ts_im_discriminator_optimizer = optim.Adam(
        ts_im_discriminator.parameters(), lr=d_lr)

    if load_model_flag:
        # s_model_dict = load_model(seg_net, model_path=seg_model_path)
        # seg_net.load_state_dict(s_model_dict)
        pass

    for epoch in range(1, epochs+1):

        s_iter_train = loop_iterable(source)
        t_iter_train = loop_iterable(target)

        recon_net.train()
        generator.train()
        st_im_discriminator.train()
        ts_im_discriminator.train()

        for iter_ in range(1, iter+1):

            recon_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            st_im_discriminator_optimizer.zero_grad()
            ts_im_discriminator_optimizer.zero_grad()

            set_requires_grad(st_im_discriminator, False)
            set_requires_grad(ts_im_discriminator, False)

            s_name, s_image = next(s_iter_train)
            t_name, t_image = next(t_iter_train)
            s_image, t_image = s_image.to(device), t_image.to(device)

            # s -> t
            fake_t = generator(s_image)
            cycle_s = recon_net(fake_t)
            fake_t_is_real = st_im_discriminator(fake_t)

            s_cycle_loss = cycle_criterion(s_image, cycle_s)

            s2t_adv_loss = criterion_c(
                fake_t_is_real, torch.ones_like(fake_t_is_real))

            # same_t = generator(t_image)
            # t_identity_loss = cycle_criterion(same_t, t_image)

            loss = 10 * s_cycle_loss + 0.01 * s2t_adv_loss
            loss.backward()

            set_requires_grad(st_im_discriminator, True)
            fake_t = fake_t.detach()

            id_fake_t = st_im_discriminator(fake_t)
            id_real_t = st_im_discriminator(t_image)

            s2t_dis_loss = 0.5 * criterion_c(id_fake_t, torch.zeros_like(id_fake_t)) + \
                0.5*criterion_c(id_real_t, torch.ones_like(id_real_t))

            s2t_dis_loss.backward()

            # t -> s
            fake_s = recon_net(t_image)
            cycle_t = generator(fake_s)
            fake_s_is_real = ts_im_discriminator(fake_s)

            t_cycle_loss = cycle_criterion(t_image, cycle_t)

            t2s_adv_loss = criterion_c(
                fake_s_is_real, torch.ones_like(fake_s_is_real))

            # same_s = recon_net(s_image)
            # s_identity_loss = cycle_criterion(same_s, s_image)

            loss = 10*t_cycle_loss + 0.01 * t2s_adv_loss
            loss.backward()

            set_requires_grad(ts_im_discriminator, True)
            fake_s = fake_s.detach()

            id_fake_s = ts_im_discriminator(fake_s)
            id_real_s = ts_im_discriminator(s_image)

            t2s_dis_loss = 0.5 * criterion_c(id_fake_s, torch.zeros_like(id_fake_s)) + \
                0.5*criterion_c(id_real_s, torch.ones_like(id_real_s))

            t2s_dis_loss.backward()

            recon_optimizer.step()
            generator_optimizer.step()
            st_im_discriminator_optimizer.step()
            ts_im_discriminator_optimizer.step()

            if iter_ % 10 == 0:
                print('Epoch: {}, iter: {}, s_cycle_loss: {}, s2t_advloss: {}, s2t_dis_loss: {}, t_cycle_loss: {}, t2s_advloss: {}, t2s_dis_loss: {}'.format(
                    epoch, iter_, s_cycle_loss.item(), s2t_adv_loss.item(), s2t_dis_loss.item(), t_cycle_loss.item(), t2s_adv_loss.item(), t2s_dis_loss.item()))
                if iter_ % 50 == 0:
                    show_img(epoch, iter_, [s_name, t_name], [
                             s_image, t_image, fake_s, fake_t, cycle_s, cycle_t], save_fold)

        path = model_fold + 'reconstructor.pth'
        torch.save(recon_net.state_dict(), path)

        path = model_fold + 'generator.pth'
        torch.save(generator.state_dict(), path)

        path = model_fold + 'st_im_discriminator.pth'
        torch.save(st_im_discriminator.state_dict(), path)

        path = model_fold + 'ts_im_discriminator.pth'
        torch.save(ts_im_discriminator.state_dict(), path)

        print('Epoch: {}, models are saved in :{}'.format(epoch, model_fold))


if __name__ == "__main__":
    main()
