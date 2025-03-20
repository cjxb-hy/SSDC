import os
import glob
import numpy as np
from PIL import Image
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _label_decomp(num_cls, label_vol, data_type='train'):

    if data_type == 'train':
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

    elif data_type == 'val':
        _batch_shape = list(label_vol.shape)
        _vol = np.zeros(_batch_shape)
        _vol[label_vol == 0] = 1
        for i in range(num_cls):
            if i == 0:
                continue
            _n_slice = np.zeros(label_vol.shape)
            _n_slice[label_vol == i*85] = 1
            _vol = np.concatenate((_vol, _n_slice), axis=2)
        return np.float32(_vol)


class Data_Loader(Dataset):
    def __init__(self, data_path, data_type='train', fold='oct'):

        self.data_path = data_path
        self.fold = fold
        self.img_path = glob.glob(os.path.join(
            data_path, self.fold+'/*.bmp'))
        self.data_type = data_type

    def __getitem__(self, index):

        image_path = self.img_path[index]
        label_path = image_path.replace(self.fold, 'refer')

        image = Image.open(image_path)
        label = Image.open(label_path)

        label = np.array(label)
        label = label[..., np.newaxis]
        label = _label_decomp(4, label, self.data_type)

        tf = transforms.Compose([
            transforms.ToTensor(),
        ])

        image, label = tf(image), tf(label)

        return image_path, image, label

    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":

    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = Data_Loader(
        "/mnt/sda/xhli/all_data/Spectralis_val", 'val')
    print("数据个数：", len(dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=16, shuffle=True)

    for batch, [name, x, y] in enumerate(train_loader):

        # np_data = np.array(s.squeeze())
        print(batch, len(name))

        # path2 = name[0].replace('oct', 'sdf_pred')
        # path2 = path2.replace('bmp', 'npy')

        # # for i in range(3):
        # #     np_data[:, :, i] = np_data[:, :, i]*(i+1)*64

        # # cv_data = np.sum(np_data, axis=2)

        # # cv2.imwrite(path2, np_data)
        # np.save(path2, np_data)

    # pass
