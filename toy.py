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
import params
import utils
from dataset import *
import icarl
import ModelRoutines

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
# SEED = 42
splits = utils.splitter()
print('splits: ' + str(splits))
rearranged_labels = []
for split in splits:
    for s in split:
        rearranged_labels.append(s)
print(rearranged_labels)
# print(len(set(rearranged_labels)))
# model
model = ResNet.resnet32(num_classes=params.TASK_SIZE).to(params.DEVICE)
# simplification: we initialize the network with all the classes
optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                            weight_decay=params.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                 gamma=params.GAMMA)  # allow to change the LR at predefined epochs
# loss
loss_function = nn.BCEWithLogitsLoss() # CrossEntropyLoss() #
binary = 1
# Run
exemplars = [None] * params.NUM_CLASSES

test_indexes = []
accs = []
# for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):
#   train_indexes = utils.get_task_indexes(train_dataset, task)
#   # print(f"The number of training samples is : {len(train_indexes)}")
#   test_indexes = test_indexes + utils.get_task_indexes(test_dataset, task)
#   # print(f"The number of testing samples is : {len(test_indexes)}")
#
#   train_subset = Subset(train_dataset, train_indexes, transform=train_transformer)
#   test_subset = Subset(test_dataset, test_indexes, transform=test_transformer)
#
#   train_loader = DataLoader(train_subset, num_workers=params.NUM_WORKERS,
#                             batch_size=params.BATCH_SIZE, shuffle=True)
#   test_loader = DataLoader(test_subset, num_workers=params.NUM_WORKERS,
#                             batch_size=params.BATCH_SIZE, shuffle=True)
#
#   model = ModelRoutines.train_model(model, loss_function, optimizer,
#                                     scheduler, train_loader,params.DEVICE,
#                                     params.NUM_EPOCHS, binary, rearranged_labels)
#
#
#   ACC = ModelRoutines.evaluate_model(model, test_loader, params.DEVICE)

# l = torch.nn.Linear(in_features=2,out_features=4)
# l.weight[:2, :2] = torch.zeros(2,2)
# print(l.weight)

plt.plot([1, 2, 3], 'go-', label='line 1', linewidth=2)
plt.plot([1, 4, 9], 'rs-', label='line 2')
plt.legend()
plt.show()