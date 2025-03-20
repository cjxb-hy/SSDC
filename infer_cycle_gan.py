import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from utlis.dataset import Data_Loader
from utlis.lib import load_model, save_img

from network.generator import Im_Generator


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    seed_torch()

    channels = 1
    batch_size = 16
    load_model_flag = True
    device = torch.device('cuda:1')

    model_path = './models/Cirrus/fastmri/generator.pth'
    
    save_fold = '/mnt/sda/xhli/drdu_cycle/fastmri/Cirrus/brain'

    source = DataLoader(dataset=Data_Loader('/mnt/sda/xhli/all_data/Cirrus_train/oct'),
                        batch_size=batch_size, shuffle=False)

    generator = Im_Generator(channels).to(device)

    if load_model_flag:
        model_dict = load_model(generator, model_path=model_path)
        generator.load_state_dict(model_dict)

    generator.eval()

    for name, image in source:

        image = image.to(device)

        fake_t = generator(image)

        b, _, _, _ = fake_t.shape
        fake_t = fake_t.detach()

        for i in range(b):
            save_img(name[i], fake_t[i, :, :, :], save_fold)


if __name__ == "__main__":
    main()
