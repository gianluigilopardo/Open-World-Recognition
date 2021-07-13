import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os
import logging
import sys
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sn
import torch.nn as nn

# our modules
from owr import BiC
from owr.dataset import *

###### This script is the main for running BiC Method on CIFAR 100 dataset
# remeber to set lr = 0.1 in params before running BiC.
print("BiC Method running on CIFAR 100")
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
splits = utils.splitter()
###########################################################################
##################### instantiate BiC object###############################

model = BiC.BiC_method(num_classes=params.NUM_CLASSES).to(params.DEVICE)
# How many parameters to fit?
# print(len([par for par in model.parameters()])) # 101 tensors as parameters without bias corrcetion
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("Solver total trainable parameters : ", total_params) # Solver total trainable parameters :  472756 (before Bias correction)

############################################################################
# lists for the evaluation phase

test_indexes = []  # this list will store the test indexes for all seen classes
# vectors for accuracy curves
new_test_accs = []
test_accs = []

#############################################################################
##################  RUN THE INCREMENTAL TRAINING ############################

for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):

    #########################################################################
    ######## MANAGE THE DATA FOR THE CURRENT TASK ###########################

    # extract the correct indexes for the current task
    ###### INDEXES ######
    all_train_indexes, corrisponding_labels = utils.get_task_indexes_with_labels(train_dataset, task)
    test_indexes = test_indexes + utils.get_task_indexes(test_dataset, task)
    new_test_indexes = utils.get_task_indexes(test_dataset, task)

    ##### SUBSET #####
    all_train_subset = Subset(train_dataset, all_train_indexes, transform=train_transformer)
    test_subset = Subset(test_dataset, test_indexes, transform=test_transformer)
    new_test_subset = Subset(test_dataset, new_test_indexes, transform=test_transformer)

    #### LOADERS #####
    all_train_loader = DataLoader(all_train_subset, num_workers=params.NUM_WORKERS,
                                  batch_size=params.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, num_workers=params.NUM_WORKERS,
                             batch_size=params.BATCH_SIZE, shuffle=True)
    new_test_loader = DataLoader(new_test_subset, num_workers=params.NUM_WORKERS,
                                 batch_size=params.BATCH_SIZE, shuffle=True)
    ################################### Incremental Training ##########################################
    # in every task the model will see new classes: become able to classify them and stores a few for the following steps
    final_loss_curve, final_training_accs, _ = model.incremental_training(train_dataset, train_transformer, task,
                                                                          new_test_loader, test_loader)
    print(f"loss_curve = {final_loss_curve}")
    print(f"accuracy_curve = {final_training_accs}")

    # if you want plot it (it slow the training)
    # plt.plot(final_loss_curve, 'go-', label='loss curve')
    # plt.plot(final_training_accs, 'rs-', label='training accuracies')
    # plt.legend()
    # plt.title(f"Task : {task}")
    # plt.show()
    ############################## EVALUATION OF THE CURRENT INCREMENTAL STEP #########################

    with torch.no_grad():
        print('\n EVALUATION \n')
        ### Get all the discovered classes
        classes = []
        for i, x in enumerate(splits[:int(task / params.TASK_SIZE) + 1]):
            v = np.array(x)
            classes = np.concatenate((classes, v), axis=None)
            classes = classes.astype(int)
        ####  NEW TEST DATA ####
        total = 0.0
        running_corrects = 0.0
        for img, lbl, _ in new_test_loader:
            img = img.float().to(params.DEVICE)
            outputs = model(img, task)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            preds = preds.to(params.DEVICE)
            labels = utils.map_splits(lbl, classes).to(params.DEVICE)
            total += len(lbl)
            running_corrects += torch.sum(preds == labels.data).data.item()
        accuracy = running_corrects / float(total)
        new_test_accs.append(accuracy)
        print(f'task: {task}', f'test accuracy on only new classes = {accuracy}')

        ##### OLD & NEW TEST DATA ####
        total = 0.0
        running_corrects = 0.0
        tot_preds = []
        tot_lab = []
        for img, lbl, _ in test_loader:
            img = img.float().to(params.DEVICE)
            outputs = model(img, task)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            preds = preds.to(params.DEVICE)
            labels = utils.map_splits(lbl, classes).to(params.DEVICE)

            tot_preds = np.concatenate((tot_preds, preds.data.cpu().numpy()))
            tot_lab = np.concatenate((tot_lab, labels.data.cpu().numpy()))

            total += len(lbl)
            running_corrects += torch.sum(preds == labels.data).data.item()

        accuracy = running_corrects / float(total)
        test_accs.append(accuracy)
        print(f'task: {task}', f'test accuracy on old and new classes = {accuracy}')

######################################################################################
#################### PLOT THE WHOLE ACCURACY CURVES ##################################

print(f"New test accuracies : {new_test_accs}")
print(f"Test accuracies : {test_accs}")
plt.plot(new_test_accs, 'go-', label='new testing accuracies', linewidth=2)
plt.plot(test_accs, 'rs-', label='testing accuracies')
plt.legend()
plt.show()
