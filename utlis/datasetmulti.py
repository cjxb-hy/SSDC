import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _label_decomp(num_cls, label_vol):

    _batch_shape = list(label_vol.shape)
    _vol = np.zeros(_batch_shape)
    _vol[label_vol == 0] = 1
    for i in range(num_cls):
        if i == 0:
            continue
        _n_slice = np.zeros(label_vol.shape)
        _n_slice[label_vol == i*64] = 1
        _vol = np.concatenate((_vol, _n_slice), axis=2)
    return np.float32(_vol)


class Multi_Data_Loader(Dataset):
    def __init__(self, data_path, ori_path):

        self.data_path = data_path
        self.ori_path = ori_path
        self.img_path = []
        for d_path in self.data_path:

            sub_list = glob.glob(os.path.join(d_path, '*.bmp'))

            self.img_path += sub_list

    def __getitem__(self, index):

        image_path = self.img_path[index]
        # /mnt/sda/xhli/drdu_data/fastmri/brain_bmp/1.bmp
        name = image_path.split('/')[-1]
        ori_img_path = os.path.join(self.ori_path, 'oct', name)
        label_path = os.path.join(self.ori_path, 'refer', name)

        image = Image.open(image_path)
        ori_img = Image.open(ori_img_path)
        label = Image.open(label_path)

        label = np.array(label)
        label = label[..., np.newaxis]
        label = _label_decomp(4, label)

        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

        # tf_dr = transforms.Compose([
        #     transforms.ColorJitter(brightness=(0.1, 2), contrast=(0.1, 2)),
        #     transforms.ToTensor(),
        # ])

        image, label = tf(image), tf(label)
        ori_img = tf(ori_img)

        return image_path, ori_img, image, label

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = Multi_Data_Loader(['/mnt/sda/xhli/drdu_cycle/fastmri/brain', '/mnt/sda/xhli/drdu_cycle/heart/ct',
                                 '/mnt/sda/xhli/drdu_cycle/heart/mr', '/mnt/sda/xhli/drdu_cycle/luna/lung_1',
                                 '/mnt/sda/xhli/drdu_cycle/luna/lung_2'],
                                '/mnt/sda/xhli/all_data/Spectralis_train')
    print("数据个数：", len(dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=4, shuffle=True)

    for batch, [name, x, ori_x, y] in enumerate(train_loader):

        # np_data = np.array(s.squeeze())
        print(batch, name)
        break

        # path2 = name[0].replace('oct', 'sdf_pred')
        # path2 = path2.replace('bmp', 'npy')

        # # for i in range(3):
        # #     np_data[:, :, i] = np_data[:, :, i]*(i+1)*64

        # # cv_data = np.sum(np_data, axis=2)

        # # cv2.imwrite(path2, np_data)
        # np.save(path2, np_data)

    # pass
