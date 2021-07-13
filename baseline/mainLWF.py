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


from owr.baseline import ResNet
from owr.baseline import params
from owr.baseline import utils
from owr.baseline import models
from owr.baseline.dataset import *
from owr.baseline import icarl

#PYCHARM:

# import ResNet
# import params
# import utils
# import dataset

#Here we address the catastrophic forgetting by adding distillation loss during the training that aims to preserve the knowledge on old classes as implemented in iCaRL paper

# preprocessing
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

# model: simplification: we initialize the network with all the classes
model = ResNet.resnet32(num_classes=params.NUM_CLASSES).to(params.DEVICE) #.to() Performs Tensor dtype and/or device conversion. 
optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM, 
                            weight_decay=params.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,gamma=params.GAMMA) #change learning rate at epoch 49 and 63 with a factor gamma


# Run
test_indexes=[]
metrics = [None] * params.NUM_TASKS  # se non inizializzo con None da errore

exemplars = [None] * params.NUM_CLASSES #not used in LWF

test_indexes = []
accs = []

for task in range(0, params.NUM_CLASSES, params.TASK_SIZE): #incremental train
  
    train_indexes = utils.get_task_indexes(train_dataset, task) #store the indexes of training images of classes that belong to the current task
    test_indexes = test_indexes + utils.get_task_indexes(test_dataset, task) #same reasoning for test data but in this case we accumulate the indexes since we test on all classes seen so far

    train_subset = Subset(train_dataset, train_indexes, transform=train_transformer) #create incremental training and testing datasets 
    test_subset = Subset(test_dataset, test_indexes, transform=test_transformer)

    train_loader = DataLoader(train_subset, num_workers=params.NUM_WORKERS,
                          batch_size=params.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, num_workers=params.NUM_WORKERS,
                         batch_size=params.BATCH_SIZE, shuffle=True)

    if (task == 0):
      torch.save(model, 'resNet_task{0}.pt'.format(task)) #in the first task we store the model as Resnet_task{0}

    models.trainLWF(task, train_loader, splits)
    loss_accuracy = models.testLWF(task, test_loader, splits)
    metrics[int(task / 10)] = loss_accuracy  # pars_task[i] = (accuracy, loss) at i-th task

