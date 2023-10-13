import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import pandas as pd


class Val_Dataset(dataset):
    def __init__(self, arg):
        self.arg = arg
        self.filename_list = pd.read_csv(arg, dtype=str, header=None).values.tolist()

    def __getitem__(self, index):
        data = np.load(self.filename_list[index][0])
        image = data['data'][0]
        label = data['data'][1]
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        sample = {'image': image, 'label': label}
        sample['case_name'] = self.filename_list[index][0].split('/')[-1][:-4]
        return sample

    def __len__(self):
        return len(self.filename_list)


def to_one_hot_3d(tensor, n_classes=3):
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot
