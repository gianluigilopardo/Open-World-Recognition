import os
import logging
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import pandas as pd
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sn

#COLAB:

# if not os.path.isdir('./owr'):
#   !git clone -b icarl https://andrerubeis:Ruby199711@github.com/gianluigilopardo/Open-World-Recognition.git
#   !mv 'Open-World-Recognition' 'owr'

# from owr import ResNet
# from owr import params
# from owr import utils
# from owr.dataset import *
# from owr import icarl

#PYCHARM:

import ResNet
import params
import utils
import dataset


# transformers
train_transformer = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])

# dataset
cifar = datasets.cifar.CIFAR100
train_dataset = cifar('data', train=True, download=True, transform=train_transformer)
test_dataset = cifar('data', train=False, download=True, transform=test_transformer)

# splits
splits = utils.splitter()
print('splits: ' + str(splits))

# model
model = ResNet.resnet32(num_classes=params.NUM_CLASSES).to(params.DEVICE) #.to() Performs Tensor dtype and/or device conversion. A torch.dtype and torch.device are inferred from the arguments of self.to(*args, **kwargs).
# simplification: we initialize the network with all the classes
optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM, #SGD with momentum renders some speed to the optimization and also helps escape local minima better. Adam is great, it's much faster than SGD, the default hyperparameters usually works fine, but it has its own pitfall too. Many accused Adam has convergence problems that often SGD + momentum can converge better with longer training time. We often see a lot of papers in 2018 and 2019 were still using SGD.
                            weight_decay=params.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,gamma=params.GAMMA) #mi serve per quando allenerò la rete, all'epoca 49 e 63 decrementa il lr di un fattore gamma

"""torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs.
    lr_scheduler.MultiStepLR: Decays the learning rate of each parameter group by gamma once the number of epoch reaches
                              one of the milestones."""
# Run
test_indexes=[]
metrics = [None] * params.NUM_TASKS  # se non inizializzo con None da errore

exemplars = [None] * params.NUM_CLASSES

test_indexes = []
accs = []

for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):
    train_indexes = utils.get_task_indexes(train_dataset, task) #salvo gli indici delle immagini di training appartennti alle 10 classi della task corrente
    test_indexes = test_indexes + utils.get_task_indexes(test_dataset, task) #stessa cosa ma con test images

    train_subset = Subset(train_dataset, train_indexes, transform=train_transformer) #prendo un sottoinsieme del training set specificando le immagini che voglio tramite indici
    test_subset = Subset(test_dataset, test_indexes, transform=test_transformer) #stessa cosa per il test

    train_loader = DataLoader(train_subset, num_workers=params.NUM_WORKERS,
                          batch_size=params.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, num_workers=params.NUM_WORKERS,
                         batch_size=params.BATCH_SIZE, shuffle=True)

    if (task == 0):
      torch.save(model, 'resNet_task{0}.pt'.format(task)) #sono nella prima task e prima di allenare la rete salvo il modello di feature extractor come file Resnet_task{0{

    models.trainLWF(task, train_loader, splits)
    loss_accuracy = models.testLWF(task, test_loader, splits)
    metrics[int(task / 10)] = loss_accuracy  # pars_task[i] = (accuracy, loss) at i-th task
