from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from Packages.augmentation import *

class FLOWER102DataLoader:

    class_names = [f"{i}" for i in range(0,102)]

    def __init__(self, model_transform, batch_size=32, data_dir= './data', shuffle=True, nworkers=4, pin_memory=True):
        self.data_dir = data_dir

        self.train_set = datasets.Flowers102(
            self.data_dir,
            split = 'train',
            download=True,
            transform=model_transform.build_transforms(train=True)
        )

        self.test_set = datasets.Flowers102(
            self.data_dir,
            split = 'val',
            download=True,
            transform=model_transform.build_transforms(train=False)
        )

        self.init_kwargs = {
            'shuffle': shuffle,
            'batch_size': batch_size,
            'num_workers': nworkers,
            'pin_memory': pin_memory
        }

    def get_loaders(self):
        return DataLoader(self.train_set, **self.init_kwargs), DataLoader(self.test_set, **self.init_kwargs)