import copy
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sn
import os
import logging
import sys
import torch.nn as nn

from owr.open_world import BiC
from owr.open_world import ResNet
from owr.open_world import models
from owr.open_world import params
from owr.open_world import utils
from owr.open_world.dataset import *
from owr.open_world import icarl
from owr.open_world import ModelRoutines
from collections import defaultdict

# This script is the main for running the fixed-threshold rejection strategy both for
# iCaRL and BiC models.
# remeber to set lr = 0.1 in params before running BiC.

### Useful functions for compute evaluation metrics ###
def return0dot0():
    return 0.0
def returnList():
    return []
def harmonic_mean(a,b):
    return (2 * a * b) / (a + b)

soft_max = nn.Softmax(dim=1)

print("iCaRL/BiC open world recognition")
print(f"learning rate : {params.LR}")
print(f"learning rate schedule epochs: {params.STEP_SIZE}")

############################################################
#################### DATA MANAGEMENT #######################

cifar = datasets.cifar.CIFAR100
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
# get the incremental subdivision of classes - Inside the function there is a seed that can be changed
# in order to evaluate another class sequence
# splits
splits = utils.splitter()
print('splits: ' + str(splits))

# Get the open_world_test_indexes, i.e. test set for evaluate the capability of reject
# (last 50 classes of the current random splits)
open_world_test_indexes = [] # list of indexes
for task in range(params.NUM_CLASSES//2,params.NUM_CLASSES, params.TASK_SIZE):
    open_world_test_indexes = open_world_test_indexes + utils.get_task_indexes(test_dataset, task)
open_world_test_subset = Subset(test_dataset, open_world_test_indexes, transform=test_transformer)
open_word_test_loader = DataLoader(open_world_test_subset, num_workers=params.NUM_WORKERS,
                                     batch_size=params.BATCH_SIZE, shuffle=True)
# this test set contains only unknown classes for our purpose
closed_word_test_indexes = [] # test set for closed world with and without rejection

###################################################################################
##################### instantiate the model's object ##############################

# choose the model
# The followiing two for iCaRL
# model = ResNet.resnet32(num_classes=params.NUM_CLASSES).to(params.DEVICE)
# exemplars = [None] * params.NUM_CLASSES

# The following for BiC
model = BiC.BiC_method(num_classes=params.NUM_CLASSES).to(params.DEVICE)

############################################################################
##### lists for the evaluation phase
# They will store the accuracy curves
closed_word_without_rejection_accuracy = []
closed_word_with_rejection_accuracy = defaultdict(returnList)
open_set_accuracy = defaultdict(returnList)
open_world_aritmetic_mean = {} # defaultdict(returnList)
open_world_harmonic_mean = {} # defaultdict(returnList)

###################################################################################
##################### set a list of rejection threesholds #########################

rejection_global_treesholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# Set the value for the unknown label
unknown_label = -1

for task in range(0, params.NUM_CLASSES // 2, params.TASK_SIZE):
    not_known = 0 # for compute a statistics
    ################################################################################################
    ############################ Data employed in this task ########################################
    # Train and Test datasets for this tasks
    train_indexes = utils.get_task_indexes(train_dataset, task)
    closed_word_test_indexes = closed_word_test_indexes + utils.get_task_indexes(test_dataset, task)
    new_test_indexes = utils.get_task_indexes(test_dataset, task)

    train_subset = Subset(train_dataset, train_indexes, transform=train_transformer)
    test_subset = Subset(test_dataset, closed_word_test_indexes, transform=test_transformer)
    new_test_subset = Subset(test_dataset, new_test_indexes, transform=test_transformer)

    train_loader = DataLoader(train_subset, num_workers=params.NUM_WORKERS,
                              batch_size=params.BATCH_SIZE, shuffle=True)
    closed_word_test_loader = DataLoader(test_subset, num_workers=params.NUM_WORKERS,
                             batch_size=params.BATCH_SIZE, shuffle=True)
    new_test_loader = DataLoader(new_test_subset, num_workers=params.NUM_WORKERS,
                             batch_size=params.BATCH_SIZE, shuffle=True)

    ################################### Incremental Training ##########################################
    # for iCaRL
    # model, exemplars = icarl.incremental_train(train_dataset, model, exemplars, task, train_transformer)
    # for BiC
    _, _, _ = model.incremental_training(train_dataset, train_transformer, task, new_test_loader, closed_word_test_loader)

    ################################### Evaluation ####################################################
    # currently discovered classes
    classes = []
    for i, x in enumerate(splits[:int(task / params.TASK_SIZE) + 1]):
        v = np.array(x)
        classes = np.concatenate((classes, v), axis=None)
        classes = classes.astype(int)
    with torch.no_grad():
        ############## Closed word without rejection
        total = 0.0
        running_corrects = 0.0
        not_known = 0
        batch = 1

        for img, lbl, _ in closed_word_test_loader:
            img = img.float().to(params.DEVICE)
            # compute the models outputs
            # for iCaRL
            # outputs = model(img).to(params.DEVICE)
            # for BiC
            outputs = model(img, task).to(params.DEVICE)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            probs = soft_max(cut_outputs)

            _, preds = torch.max(probs.data, 1)

            labels = utils.map_splits(lbl, classes).to(params.DEVICE)
            total += len(lbl)

            running_corrects += torch.sum(preds == labels.data).data.item()
            batch = batch + 1

        accuracy = float(running_corrects / total)
        closed_word_without_rejection_accuracy.append(accuracy)
        print(f'task: {task}', f'closed_word_without_rejection_accuracy = {accuracy}')

        ############## Closed word with rejection
        total = 0.0
        running_corrects = 0.0
        not_known = 0
        batch = 1
        running_corrects_dict = defaultdict(return0dot0)

        for img, lbl, _ in closed_word_test_loader:
            img = img.float().to(params.DEVICE)

            # for iCaRL
            # outputs = model(img).to(params.DEVICE)
            # for BiC
            outputs = model(img, task).to(params.DEVICE)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            probs = soft_max(cut_outputs)

            maxPred, preds = torch.max(probs.data, 1)
            labels = utils.map_splits(lbl, classes).to(params.DEVICE)
            total += len(lbl)
            ####################################################################################################
            ##################### APPLY THE REJECTION STRATEGY FOR EACH THRESHOLDS #############################
            # scan 'maxPred', if the maximum probability is below the threshold predict as unknown otherwise with
            # the corresponding 'preds' label.
            for threeshold in rejection_global_treesholds:
                preds_tmp = copy.deepcopy(preds)
                # Apply the threeshold in a naive way, this can be upgraded
                for i, p in enumerate(maxPred):
                    if p <= threeshold:
                        preds_tmp[i] = unknown_label
                        not_known = not_known + 1
                running_corrects_dict[str(threeshold)] += torch.sum(preds_tmp == labels.data).data.item()
            batch = batch + 1
        for threeshold in rejection_global_treesholds:
            accuracy = float(running_corrects_dict[str(threeshold)] / total)
            closed_word_with_rejection_accuracy[str(threeshold)].append(accuracy)
            print(f'task: {task}' f'threeshold : {threeshold}, closed_word_with_rejection_accuracy = {accuracy}')

        ############################### Open Word Scenario
        total = 0.0
        running_corrects = 0.0
        not_known = 0
        batch = 1
        running_corrects_dict = defaultdict(return0dot0)

        for img, _ , _ in open_word_test_loader:
            # we do not need labels, this are only unknown
            img = img.float().to(params.DEVICE)

            # for iCaRL
            # outputs = model(img).to(params.DEVICE)
            # for BiC
            outputs = model(img, task).to(params.DEVICE)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            probs = soft_max(cut_outputs)

            maxPred, preds = torch.max(probs.data, 1)
            total += len(preds)
            ####################################################################################################
            ##################### APPLY THE REJECTION STRATEGY FOR EACH THRESHOLDS #############################
            # scan 'maxPred', if the maximum probability is below the threshold predict as unknown otherwise with
            # the corresponding 'preds' label.
            for threeshold in rejection_global_treesholds:
                preds_tmp = copy.deepcopy(preds)
                # Apply the threeshold in a naive way, this can be upgraded
                for i, p in enumerate(maxPred):
                    if p <= threeshold:
                        preds_tmp[i] = unknown_label
                        not_known = not_known + 1
                running_corrects_dict[str(threeshold)] += torch.sum(preds_tmp == unknown_label).data.item()
            batch = batch + 1
        for threeshold in rejection_global_treesholds:
            accuracy = float(running_corrects_dict[str(threeshold)] / total)
            open_set_accuracy[str(threeshold)].append(accuracy)
            print(f'task: {task}' f'threeshold : {threeshold}, open_set_accuracy = {accuracy}')

#### THE INCREMENTAL TRAINING HAS ENDED ####
############## Compute the Open World Harmonic and Aritmetic Mean
for threeshold in rejection_global_treesholds:
    open_world_aritmetic_mean[str(threeshold)] = [(a + b)/2 for a,b in zip(closed_word_with_rejection_accuracy[str(threeshold)], open_set_accuracy[str(threeshold)])]
    open_world_harmonic_mean[str(threeshold)] = [harmonic_mean(a,b) for a,b in zip(closed_word_with_rejection_accuracy[str(threeshold)], open_set_accuracy[str(threeshold)])]

##########################################################################################
############################ PRINT FINAL RESULTS #########################################
print("\n RESULTS : ")

print("\n Closed world without rejection accuracy")
print(closed_word_without_rejection_accuracy)

print("\n Closed world with rejection accuracy")
print(closed_word_with_rejection_accuracy)

print("\n Open world accuracy")
print(open_set_accuracy)

print("\n Open world aritmetic mean")
print(open_world_aritmetic_mean)

print("\n Open world harmonic mean")
print(open_world_harmonic_mean)
