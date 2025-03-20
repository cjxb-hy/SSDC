import os
import glob
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Data_Loader(Dataset):
    def __init__(self, data_path):

        self.data_path = data_path

        self.img_path = glob.glob(data_path + '/*.bmp')

    def __getitem__(self, index):

        image_path = self.img_path[index]

        image = Image.open(image_path)

        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

        image = tf(image)

        return image_path, image

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = Data_Loader("F:/drdu_data/chaos/ct_bmp")

    print("数据个数：", len(dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=4, shuffle=True)

    for batch, [name, x] in enumerate(train_loader):

        print(batch, name, x.shape)
        break
