import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image

class LOLDataset(Dataset):
    def __init__(self, train , path = '../../dataset/Combined', transform = None, p_rot90 = 0, p_flipud = 0, p_fliplr = 0, patch_mode = 0):
        super(LOLDataset, self).__init__()
        self.patch_mode = patch_mode
        self.p_rot90 = p_rot90
        self.p_fliplr = p_fliplr
        self.p_flipud = p_flipud
        self.train = train
        self.transform = transform
        if self.train == 'train':
            self.datapath = os.path.join(path, 'train')
        elif self.train == 'test':
            self.datapath = os.path.join(path, 'test')
        elif self.train == 'unknown':
            self.datapath = os.path.join(path, 'unknown')
        else:
            print("Unknown mode for 'train'")
            assert(0)
        self.highpath = os.path.join(self.datapath, 'high')
        self.lowpath = os.path.join(self.datapath, 'low')
        self.high_names = os.listdir(self.highpath)
        self.low_names = os.listdir(self.lowpath)
        assert(self.high_names == self.low_names)


    def __getitem__(self, i):
        patch_size = 120
        low_data = torch.tensor(np.transpose(plt.imread(os.path.join(self.lowpath, self.low_names[i])),(2,0,1)))
        high_data = torch.tensor(np.transpose(plt.imread(os.path.join(self.highpath, self.high_names[i])),(2,0,1)))
        if self.transform is not None:
            low_data = self.transform(low_data)
            high_data = self.transform(high_data)
        if self.cointoss(self.p_fliplr):
            low_data = torch.flip(low_data, [2])
            high_data = torch.flip(high_data, [2])
        if self.cointoss(self.p_flipud):
            low_data = torch.flip(low_data, [1])
            high_data = torch.flip(high_data, [1])
        if self.cointoss(self.p_rot90):
            low_data = low_data.permute(0,2,1)
            high_data = high_data.permute(0,2,1)
        if self.patch_mode:
            _,h,w = low_data.shape
            h_left, w_left = h-patch_size, w-patch_size
            h_start = np.random.randint(0,h_left)
            w_start = np.random.randint(0,w_left)
            return low_data[:,h_start:h_start+patch_size,w_start:w_start+patch_size], high_data[:,h_start:h_start+patch_size,w_start:w_start+patch_size]
        return low_data, high_data

    def __len__(self):
        return len(self.high_names)

    def cointoss(self, p):
        return torch.bernoulli(torch.tensor(p*1.0)) == 1