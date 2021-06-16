import os
import logging
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import torch.nn as nn
import ResNet
import models
import params
import utils
from dataset import *
import icarl
import ModelRoutines

print(f"learning rate : {params.LR}")
print(f"learning rate schedule epochs: {params.STEP_SIZE}")

cifar = datasets.cifar.CIFAR100
# Initializations
# transformers
train_transformer = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])
train_dataset = cifar('data', train=True, download=True, transform=train_transformer)
test_dataset = cifar('data', train=False, download=True, transform=test_transformer)
# splits
splits = utils.splitter()
print(f"The splits are {splits}")
# model
model = ResNet.resnet32(num_classes=params.NUM_CLASSES).to(params.DEVICE)
# simplification: we initialize the network with all the classes
optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                            weight_decay=params.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                 gamma=params.GAMMA)  # allow to change the LR at predefined epochs
# loss
loss_function = nn.BCEWithLogitsLoss()  # CrossEntropyLoss() #
binary = 1 # da eliminare
# Run
exemplars = [None] * params.NUM_CLASSES # da eliminare

test_indexes = []
# vectors for accuracy curves
train_accs = []
test_accs = []
for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):

    classes = utils.get_classes(splits, task)

    classes = []
    for i, x in enumerate(splits[:int(task / params.TASK_SIZE) + 1]):
        v = np.array(x)
        classes = np.concatenate((classes, v), axis=None)
        classes = classes.astype(int)

    print(f"\n Task : {task}, Classes : {classes}, len vector : {len(classes)} \n ")

# plt.plot([1, 2, 3], 'go-', label='line 1', linewidth=2)
# plt.plot([1, 4, 9], 'rs-', label='line 2')
# plt.legend()
# plt.show()