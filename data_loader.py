'''
adapted from https://github.com/fungtion/DANN_py3/blob/master/data_loader.py
by James Kim
May 25, 2023
'''

import torch.utils.data as data
from PIL import Image
import os

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform = None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r') # NOTE: consider using 'with'
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform: # if not None
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels
    
    def __len__(self):
        return self.n_data