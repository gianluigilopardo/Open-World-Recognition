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

from owr.baseline import ResNet
from owr.baseline import models
from owr.baseline import params
from owr.baseline import utils
from owr.baseline.dataset import *

###### This script is the main for running finetuning on CIFAR 100 dataset
# remeber to set lr = 2 in params before running.
print(f"learning rate : {params.LR}")
print(f"learning rate schedule epochs: {params.STEP_SIZE}")

############################################################
#################### DATA MANAGEMENT #######################

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

###########################################################################
##################### instantiate BiC object###############################

model = ResNet.resnet32(num_classes=params.NUM_CLASSES).to(params.DEVICE)
# simplification: we initialize the network with all the classes

#########################################################################################
##################### Loss Function (Only classification) ###############################
loss_function = nn.BCEWithLogitsLoss()  # CrossEntropyLoss() #

############################################################################
# lists for the evaluation phase
test_indexes = []
# vectors for accuracy curves
train_accs = []
test_accs = []

#############################################################################
##################  RUN THE INCREMENTAL TRAINING ############################

for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):
    #########################################################################
    ######## MANAGE THE DATA FOR THE CURRENT TASK ###########################

    train_indexes = utils.get_task_indexes(train_dataset, task)
    test_indexes = test_indexes + utils.get_task_indexes(test_dataset, task)

    train_subset = Subset(train_dataset, train_indexes, transform=train_transformer)
    test_subset = Subset(test_dataset, test_indexes, transform=test_transformer)

    train_loader = DataLoader(train_subset, num_workers=params.NUM_WORKERS,
                              batch_size=params.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, num_workers=params.NUM_WORKERS,
                             batch_size=params.BATCH_SIZE, shuffle=True)
    # instantiate optimizer and scheduler for this task
    optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                                weight_decay=params.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                     gamma=params.GAMMA)  # allow to change the LR at predefined epochs
    classes = utils.get_classes(splits, task) # classes until now

    ################################### Incremental Training ##########################################
    # train the model without preventing catastrofic forgetting
    model = models.train_network_lower_bound(classes, model, optimizer, train_loader, scheduler, task)

    ########################## EVALUATION OF THE CURRENT INCREMENTAL STEP #############################

    print('\n EVALUATION \n')
    classes = []
    for i, x in enumerate(splits[:int(task / params.TASK_SIZE) + 1]):
        v = np.array(x)
        classes = np.concatenate((classes, v), axis=None)
        classes = classes.astype(int)
    total = 0.0
    running_corrects = 0.0
    for img, lbl, _ in train_loader:
        img = img.float().to(params.DEVICE)
        outputs = model(img)
        cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
        _, preds = torch.max(cut_outputs.data, 1)
        preds = preds.to(params.DEVICE)
        labels = utils.map_splits(lbl, classes).to(params.DEVICE)
        total += len(lbl)
        running_corrects += torch.sum(preds == labels.data).data.item()
    accuracy = float(running_corrects / total)
    print(f'task: {task}', f'train accuracy = {accuracy}')
    train_accs.append(accuracy)

    total = 0.0
    running_corrects = 0.0
    tot_preds = []
    tot_lab = []
    for img, lbl, _ in test_loader:
        img = img.float().to(params.DEVICE)
        outputs = model(img)
        cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
        _, preds = torch.max(cut_outputs.data, 1)
        preds = preds.to(params.DEVICE)
        labels = utils.map_splits(lbl, classes).to(params.DEVICE)

        tot_preds = np.concatenate((tot_preds, preds.data.cpu().numpy()))
        tot_lab = np.concatenate((tot_lab, labels.data.cpu().numpy()))

        total += len(lbl)
        running_corrects += torch.sum(preds == labels.data).data.item()
        # print('running_corrects: ' + str(running_corrects))
    accuracy = float(running_corrects / total)
    test_accs.append(accuracy)
    print(f'task: {task}', f'test accuracy = {accuracy}')
    # cf = confusion_matrix(tot_lab, tot_preds)
    # df_cm = pd.DataFrame(cf, range(task + params.TASK_SIZE), range(task + params.TASK_SIZE))

######################################################################################
#################### PLOT THE WHOLE ACCURACY CURVES ##################################

print(f"Training accuracies : {train_accs}")
print(f"Testing accuracies : {test_accs}")
plt.plot(train_accs, 'go-', label='training accuracies', linewidth=2)
plt.plot(test_accs, 'rs-', label='testing accuracies')
plt.legend()
plt.show()
