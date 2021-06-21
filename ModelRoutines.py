# This file cointains routines for training and evaluation of the incremental models
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from owr import params
from owr import utils
from owr.dataset import *


def train_model(model, loss_function, optimizer, scheduler, train_loader,device, num_epochs, binary):
    n_total_steps = len(train_loader)
    print(f"The total number of steps for each epoch will be {n_total_steps}")
    for epoch in range(num_epochs):
        for i, (images, labels, _) in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)
            labels_1h = F.one_hot(labels, num_classes = params.TASK_SIZE).float()
            if binary ==1 :
                labels = labels_1h

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i + 1) % (n_total_steps/4) == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
    return model

def evaluate_model(model, test_loader,device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        # n_class_correct = [0 for i in range(10)]
        # n_class_samples = [0 for i in range(10)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        acc = n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
    return acc

def train_model_incremental(model,loss_function,optimizer, scheduler, train_dataset, train_transformer,
                            test_dataset, test_transformer, splits):
    new_labels_order = [] # it is useful to corrcetly perform training and evaluation
    # from new to old
    new_labels_dictionary = {}
    # from old to new
    for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):
        # update new_labels_order
        for lbl in splits[task]:
            new_labels_order.append(lbl)

        train_indexes = utils.get_task_indexes(train_dataset, task)
        test_indexes = test_indexes + utils.get_task_indexes(test_dataset, task)

        train_subset = Subset(train_dataset, train_indexes, transform=train_transformer)
        test_subset = Subset(test_dataset, test_indexes, transform=test_transformer)

        train_loader = DataLoader(train_subset, num_workers=params.NUM_WORKERS,
                                  batch_size=params.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_subset, num_workers=params.NUM_WORKERS,
                                 batch_size=params.BATCH_SIZE, shuffle=True)

        # training phase
        binary = True
        model = train_model(model, loss_function, optimizer,
                                        scheduler, train_loader,params.DEVICE,
                                        params.NUM_EPOCHS, binary)
        # evaluation phase

        # on training samples

        # on test samples

        # if the incremental procedure will continue add to the classification layer new neurons
