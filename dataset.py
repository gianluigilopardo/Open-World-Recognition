import numpy as np
import torch
from PIL import Image

import utils
import params


class Dataset(torch.utils.data.Dataset):  # extends pytorch class Dataset
    """
    The class Dataset define methods and attributes to manage the dataset
    Attributes:
        -_train: Bool, default value = True
        -_dataset:
        -_data: list of images, each represented by a [32]x[32]x[3] vector that define pixels
        -_targets: list of labels for each image
    """

    def __init__(self, dataset, train=True, transform=None, target_transform=None):
        """
        :param dataset: pytorch dataset used, e.g.: datasets.cifar.CIFAR100
        :param train: Boolean
        :param transform: pytorch transformers
        :param target_transform: pytorch target_transform
        """
        self._train = train
        self._dataset = dataset('data', train=train, download=True, transform=transform,
                                target_transform=target_transform)
        self._targets = np.array(self._dataset.targets)
        self._data = np.array(self._dataset.data)
        self.splits = utils.splitter()

    def get_classes_names(self):
        """
        :return: list mapping the classes of the dataset into text labels
        """
        names = list(self._dataset.class_to_idx.keys())
        return names

    def get_indexes_groups(self, current_task=0):
        """
        :param current_task: int
        :return:
        """
        # This method returns a list containing the indexes of all the images
        # belonging to the classes [current_task, current_task + TASK_SIZE]
        indexes = []
        current_task = int(current_task/params.TASK_SIZE)
        # print(current_task)
        # print(self.splits)
        self.searched_classes = self.splits[current_task]
        i = 0
        for el in self._targets:
            if el in self.searched_classes:
                indexes.append(i)
            i += 1
        return indexes

    def __getitem__(self, idx):
        """
        :param idx: index of an image
        :return: image and its class label
        """
        image = np.transpose(self._data[idx])
        label = self._targets[idx]
        return image, label, idx

    def append(self, images, labels):
        self._data = np.concatenate((self._data, images), axis=0)
        self._targets = np.concatenate((self._targets, np.array(labels)), axis=0)

    def __len__(self):
        return len(self._targets)


class Subset(Dataset):
    """
    serve per prendere dei sottoinsiemi dell'insieme originale a seconda del task in cui ci troviamo
    quindi anzichè fare il training su tutto il dataset lo faccio sulle immagini le cui classi sono
    nel task in cui ci troviamo

    Noi dobbiamo riaddestrare tutta la nostra rete, non su tutto il dataset ma su dei task dei dataset,
    dei batch di dataset. Ad esempio se andiamo di 10 in 10, il primo task ha dimensione 10e il primo
    subset sarà un sottoinsieme di 10 classi del dataset

    """
    def __init__(self, dataset, indices, transform):
        """
        :param dataset: the whole dataset
        :param indices: indices to take and put in the subset
        :param transform: transformers che gli passiamo, serve quando creaimo il dataset se voglio usare transformer
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        """
        dato un insieme di indici lui mi restituisce le immagini e le labels delle immagini
        :param idx:
        :return:
        """
        image, labels, _ = self.dataset[self.indices[idx]]
        return self.transform(Image.fromarray(np.transpose(image))), labels, idx

    def __len__(self):
        return len(self.indices)
