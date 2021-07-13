import copy
import os
import logging
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sn
import torch.nn as nn

if not os.path.isdir('./owr'):
   !git clone https://gianluigilopardo/Open-World-Recognition.git
   !mv 'Open-World-Recognition' 'owr'

from owr import BiC
from owr import ResNet
from owr import models
from owr import params
from owr import utils
from owr.dataset import *
from owr import icarl
from owr import ThresholdsLearner #va caricato nel branch main
from collections import defaultdict

# This script is the main for running the class-specific learnd rejection strategy for BiC method.
# remeber to set lr = 0.1 in params before running BiC.

### Useful functions for compute evaluation metrics ###

def return0dot0():
    return 0.0
def returnList():
    return []
def harmonic_mean(a,b):
    return (2 * a * b) / (a + b)

soft_max = nn.Softmax(dim=1)

print("BiC owr - Data Learned Thresholds")
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

# The following for BiC
model = BiC.BiC_method(num_classes=params.NUM_CLASSES).to(params.DEVICE)

############################################################################
##### lists for the evaluation phase
# They will store the accuracy curves
# we do not need dict---> There is only a (vector of) threshold => a list for saving results is enough
closed_word_without_rejection_accuracy = []
closed_word_with_rejection_accuracy = []
open_set_accuracy = []
open_world_aritmetic_mean = []
open_world_harmonic_mean = []

unknown_label = -1

for task in range(0, params.NUM_CLASSES // 2, params.TASK_SIZE):
    not_known = 0 ## for compute a statistics
    ################################################################################################
    ############### INSTANTIATE THE THRESHOLD PARAMETER FOR THIS TASK ##############################
    threshold = nn.parameter.Parameter(torch.ones(task + params.TASK_SIZE, device=params.DEVICE))

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
    ############################################################################################
    ############################# INCREMENTAL TRAINING  #######################################
    _, _, threshold_dataset_loader = model.incremental_training(train_dataset, train_transformer, task, new_test_loader,
                                                                closed_word_test_loader, with_rejection= True)
    ############################################################################################
    ############################# LEARNING THE THRESHOLDS ######################################
    # currently discovered classes
    classes = []
    for i, x in enumerate(splits[:int(task / params.TASK_SIZE) + 1]):
        v = np.array(x)
        classes = np.concatenate((classes, v), axis=None)
        classes = classes.astype(int)
    threshold = ThresholdsLearner.learn_thresholds(threshold, threshold_dataset_loader, classes, model, task)
    print('\n The learned threeshold are : ')
    print(threshold)
    ################################### Evaluation ####################################################
    with torch.no_grad():
        ############## Closed word without rejection
        total = 0.0
        running_corrects = 0.0
        not_known = 0
        batch = 1

        for img, lbl, _ in closed_word_test_loader:
            img = img.float().to(params.DEVICE)

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
        print(f'\n task: {task}', f'closed_word_without_rejection_accuracy = {accuracy}')

        ############## Closed word with rejection
        # use the rejection
        total = 0.0
        running_corrects = 0.0
        not_known = 0
        batch = 1
        running_corrects_dict = 0.0

        for img, lbl, _ in closed_word_test_loader:
            img = img.float().to(params.DEVICE)

            outputs = model(img, task).to(params.DEVICE)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            probs = soft_max(cut_outputs)

            # maxPred, preds = torch.max(probs.data, 1)
            labels = utils.map_splits(lbl, classes).to(params.DEVICE)
            total += len(lbl)
            # Apply the rejection strategy with a method
            preds = ThresholdsLearner.rejection(probs, threshold)
            running_corrects += torch.sum(preds == labels.data).data.item()
            batch = batch + 1
        accuracy = float(running_corrects / total)
        closed_word_with_rejection_accuracy.append(accuracy)
        print(f'task: {task}' f'closed_word_with_rejection_accuracy = {accuracy}')

        ############################### Open Word Scenario
        # use the rejection
        total = 0.0
        running_corrects = 0.0
        not_known = 0
        batch = 1
        running_corrects_dict = 0.0

        for img, _ , _ in open_word_test_loader:
            # we do not need labels, this are only unknown
            img = img.float().to(params.DEVICE)

            outputs = model(img, task).to(params.DEVICE)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            probs = soft_max(cut_outputs)

            # maxPred, preds = torch.max(probs.data, 1)
            total += len(preds)
            # Apply the rejection strategy with a method
            preds = ThresholdsLearner.rejection(probs, threshold)
            running_corrects += torch.sum(preds == unknown_label).data.item()
        accuracy = float(running_corrects / total)
        open_set_accuracy.append(accuracy)
        print(f'task: {task}' f', open_set_accuracy = {accuracy}')

#### THE INCREMENTAL TRAINING HAS ENDED ####
############## Compute Open World Harmonic and Aritmetic Mean
open_world_aritmetic_mean = [(a + b)/2 for a,b in zip(closed_word_with_rejection_accuracy, open_set_accuracy)]
open_world_harmonic_mean = [harmonic_mean(a,b) for a,b in zip(closed_word_with_rejection_accuracy, open_set_accuracy)]

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
