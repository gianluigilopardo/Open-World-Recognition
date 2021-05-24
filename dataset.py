import numpy as np
import torch
from PIL import Image

from owr import utils
from owr import params


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indexes, transform):
        """
        :param dataset: the whole dataset
        :param indices: indices to take and put in the subset
        """
        self.dataset = dataset
        self.indexes = indexes
        self.transform = transform

    def __getitem__(self, idx):
        image, labels, _ = self.dataset[self.indexes[idx]]
        return self.transform(Image.fromarray(np.transpose(image))), labels, idx

    def __len__(self):
        return len(self.indexes)
