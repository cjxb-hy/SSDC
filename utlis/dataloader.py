import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


class Data_Loader(Dataset):
    def __init__(self, data_path, num_class, data_type='train'):

        self.data_path = data_path
        self.num_class = num_class
        self.data_type = data_type

        self.img_list = glob.glob(self.data_path + '/*.bmp')

    def label_decomp(self, label_vol, num_cls, data_type='train'):

        ratio = 64 if data_type == 'train' else 85

        _vol = np.zeros(label_vol.shape)
        _vol[label_vol == 0] = 1

        for i in range(1, num_cls):
            _n_slice = np.zeros(label_vol.shape)
            _n_slice[label_vol == i * ratio] = 1
            _vol = np.concatenate((_vol, _n_slice), axis=2)

        return np.float32(_vol)

    # 'drdu_cycle/luna/Spectralis/lung_1'
    # 'drdu_cycle/fastmri/Spectralis/brain'
    def __getitem__(self, index):

        img_path = self.img_list[index]
        if self.data_type == 'train':
            imgp_path = img_path.replace(
                'all_data/Spectralis_train/oct', 'drdu_cycle/luna/Spectralis/lung_2')
        else:
            imgp_path = img_path
        lbl_path = img_path.replace('oct', 'refer')

        image = Image.open(img_path)
        imagep = Image.open(imgp_path)
        label = Image.open(lbl_path)

        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

        label = np.array(label)
        label = label[..., np.newaxis]
        label = self.label_decomp(label, self.num_class, self.data_type)

        image, imagep, label = tf(image), tf(imagep), tf(label)

        return img_path, image, imagep, label

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':

    train_set = DataLoader(dataset=Data_Loader('/home/niusijie/all_data/Spectralis_train/oct', 4, 'train'),
                           batch_size=4, shuffle=True, num_workers=2, drop_last=True)

    print(len(train_set))

    # for name, namep, img, imgp, lbl in train_set:
    #     print(name[0])
    #     print(namep[0])
    #     break
