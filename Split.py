
import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
#from DatasetCIFAR import params
import random

#random.seed(params.SEED)
from PIL import Image
import os
import os.path
import sys



class Dataset(torch.utils.data.Dataset):
    '''
    The class Dataset define methods and attributes to manage a CIFAR100 dataset
    Attributes:
        train = Bool, default value = True
        Transform
        Target_Transform

        _dataset = contains the pythorch dataset CIFAR100 defined by hyperparameters passes by input
        _targets = contains a lisf of 60000 elements, and each one referred to an image of the dataset. The value of each element is an integer in [0,99] that explicit the label for that image
                    E.g. _targets[100] get the label of the 100th image in the dataset
        _data = contains a list of 60000 imagest, each one represented by a [32]x[32]x[3] vector that define pixels
        _labelNames = contains a list of 100 elements, each one represent a class; it maps integer indexes to human readable labels

    '''

    def returnSplits(SEED = 0):
        el = np.linspace(0, 99, 100)  # el è il vettore [0,1,...99]
        splits = [None] * 10  # creo la lista di 10 elementi inizialmente vuoti

        for i in range(0, 10):
            random.seed(SEED)  # inizializzo a ogni nuovo inizio del for il seed con cui estrarre randomicamente i numeri dal vettore
            n = random.sample(set(el),k=10)  # estraggo 10 float dalla popolazione casualmente, ho bisogno di usare set() perchè richiede un set come input
            splits[i] = n  # riempio la lista
            el = list(set(el) - set(n))  # s - t = nuovo insieme con elementi in s ma non in t

        # quindi praticamente io inizio con un vettore [0:99] e ogni volta ne estraggo 10 casualmente, ma alla iterazione
        # successiva non potrò prendere i 10 che ho preso in precedenza, infatti alla fine del for, set(el) - set(n) = 0

        # splits è una lista di 10 liste, ciascuna con numeri diversi, tutte le 10 liste sono state create casualmente
        return splits

#     def __getClassesNames__(self):
#         # This method returns a list mapping the 100 classes into a human readable label. E.g. names[0] is the label that maps the class 0
#         names = []
#         classi = list(self._dataset.class_to_idx.keys())
#         return classi
#
#     def __init__(self, train=True, transform=None, target_transform=None):
#         self._train = train
#         self._dataset = datasets.cifar.CIFAR100('data', train=train, download=True, transform=transform,
#                                                 target_transform=target_transform)
#         self._targets = np.array(
#             self._dataset.targets)  # targets è np array di 50 K di numeri interi, ovvero credo dica per ogni immagine del training setqual è la classe di appartenenzaEssendo CIFAR100 una sottoclasse di CIFAR10, qui fa riferimento a quell'implementazione.
#         self._data = np.array(self._dataset.data)
#         self.splits = params.returnSplits()
#
#     def __getIndexesGroups__(self, index=0):  # index in  realtà è  task
#         # This method returns a list containing the indexes of all the images belonging to classes [starIndex, startIndex + 10]
#         indexes = []
#         self.searched_classes = self.splits[
#             int(index / 10)]  # index = 0 sono nella prima lista delle 10 liste di immagini casuali
#         # self.searched_classes è la lista di 10 elementi, dove ogni elemento è un numero tra 0 e 100 che indica una certa classe
#         i = 0
#         for el in self._targets:  # scansiono il vettore di 50K e ciascuna cella contiene un intero corrispondente alla classe a cui apparteiene quell'immagiine
#
#             # sel.targets[5] = 79 -> la 5 immagine del training set appartiene alla classe 79
#
#             if (el in self.searched_classes):
#                 # se l'immagine di training corrente appartiene a una delle classi tra le 10 in self.searched_classes
#                 indexes.append(i)
#             i += 1
#         return indexes  # è una lista contenente gli indici delle immagini di training appartenenti alle 10 classi di self.searched_classes
#
#     def __getitem__(self, idx):
#         # Given an index, this method return the image and the class corresponding to that index
#         image = np.transpose(self._data[idx])
#         label = self._targets[idx]
#         return image, label, idx
#
#     def append(self, images, labels):
#         self._data = np.concatenate((self._data, images), axis=0)
#         self._targets = np.concatenate((self._targets, np.array(labels)), axis=0)
#
#     def __len__(self):
#         return len(self._targets)
#
#
# class Subset(Dataset):
#     r"""
#     Subset of a dataset at specified indices.
#     Arguments:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#     """
#
#     def __init__(self, dataset, indices, transform):
#         self.dataset = dataset
#         self.indices = indices
#         self.transform = transform
#
#     def __getitem__(self, idx):
#         im, labels, _ = self.dataset[self.indices[idx]]
#         return self.transform(Image.fromarray(np.transpose(im))), labels, idx
#
#     def __len__(self):
#         return len(self.indices)