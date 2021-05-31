# import torchvision
# import tarfile
# import os
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision import datasets
# import pandas as pd
# from sklearn.metrics import confusion_matrix
# import seaborn as sn
#
# import ResNet
# from dataset import *
# import params
# import icarl

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
import torchvision.transforms as tt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split,ConcatDataset


import params
import ResNet

#Before load the data we need to define apriori the transformation we want to apply, in order to avoid overfitting
#and have a better training

stats =((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32,padding=4,padding_mode="reflect"),
    tt.ToTensor(),
    #tt.Normalize(*stats)               se lo metto non stampa immagini
])

test_transform = tt.Compose([
    tt.ToTensor(),
    #tt.Normalize(*stats)               se lo metto non stampa immagini
])

"""
RandomHorizontalFlip randomly flips an image with a probability of 50%, and RandomCrop pads an image by 4 pixel
on each side then randomly crops 32x32 from the image after padding. We add such transformations to add noise
to the data and prevent our model from overfitting. There are also other transformations you can use such as
ColorJitter and RandomVerticalFlip,etc. but I found these to be sufficient for my purposes.

ToTensor simply converts the image to a Tensor. Since its a coloured image, it would have 3 channels (R,G,B)
 so the Tensor would be of size 3x32x32.
 
Normalize takes the mean and standard deviation for each channel of the entire dataset as input. Normalizing scales our data to a similar range of values to make sure that our gradients don’t go out of control.
"""

# dataset

train_dataset = CIFAR100(download=True,root="./data",transform=train_transform)
test_dataset = CIFAR100(root="./data",train=False,transform=test_transform)

"""
train_dataset = è una collezione di 50 K tuple
train_dataset[i] = è la tupla i avente come primo elemento l'immagine convertita in tensor 3x32x32 e come secondo 
                elemento la label (intero) che va da 0 a 100, quindi tiene già conto delle fine_classes e non coarse
"""

# a,b = test_dataset[1]
# print(a.shape)
# print(b)

# for image,label in train_dataset:
#     print("Image shape: ",image.shape)
#     print("Image tensor: ", image)
#     print("Label: ", label)
#     break
#
# train_classes_items = dict() #creo dizionario con chiavi classi e le conto
#
# for train_item in train_dataset:
#     label = train_dataset.classes[train_item[1]]
#     if label not in train_classes_items:
#         train_classes_items[label] = 1
#     else:
#         train_classes_items[label] += 1



def show_example(img, label):
    print('Label: ', train_dataset.classes[label], "("+str(label)+")")
    print(img.shape)
    img = img.permute(1,2,0)
    print(img.shape)
    plt.imshow(img) #channel are at the last in matplotlib where it was at front in tensors
    #plt.show()

show_example(train_dataset[0][0], train_dataset[0][1])

train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE,  # num_workers=params.NUM_WORKERS,
                         pin_memory=True, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params.BATCH_SIZE,  # num_workers=params.NUM_WORKERS,
                         pin_memory=True, shuffle=True)

"""
We divide the data into batches because if we feed all the data to our model in one instance, our computer might not be able to handle it or it might take too long. Batch size also plays a part in optimizing our model.
Num_workers generates batches in parallel. It essentially prepares the next n batches after a batch has been used. Pin_memory helps speed up the transfer of data from the CPU to the GPU. Both of these simply speeds up our work.
Now the last thing, shuffle, simply makes it so that the data we feed our model is randomly ordered.
"""

def show_batch(dl):
    for batch in dl:
        images,labels = batch
        fig, ax = plt.subplots(figsize=(7.5,7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20],nrow=5).permute(1,2,0))
        #plt.show()
        break

show_batch(train_loader)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class ToDeviceLoader:
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.data)

device = get_device()
print(device)

train_dl = ToDeviceLoader(train_loader,device)
test_dl = ToDeviceLoader(test_loader,device)

#model
model = ResNet.resnet32(num_classes=params.NUM_CLASSES).to(params.DEVICE)

# simplification: we initialize the network with all the classes
optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                            weight_decay=params.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                 gamma=params.GAMMA)  # allow to change the LR at predefined epochs
# Run
exemplars = [None] * params.NUM_CLASSES #all'inizio None perchè solo finetuning

test_indexes = []
accs = []

for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):
    train_indexes = train_dataset.get_indexes_groups(task)
    test_indexes = test_indexes + test_dataset.get_indexes_groups(task)

    train_subset = Subset(train_dataset, train_indexes, transform=train_transformer)
    test_subset = Subset(test_dataset, test_indexes, transform=test_transformer)

    train_loader = DataLoader(train_subset,  # num_workers=params.NUM_WORKERS,
                              batch_size=params.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset,  # num_workers=params.NUM_WORKERS,
                             batch_size=params.BATCH_SIZE, shuffle=True)

    model, exemplars = icarl.incremental_train(train_dataset, model, exemplars, task, train_transformer)

    col = []
    for i, x in enumerate(train_splits[:int(task / params.NUM_CLASSES * params.TASK_SIZE)+1]):
        v = np.array(x)
        col = np.concatenate((col, v), axis=None)
        col = col.astype(int)
    mean = None
    total = 0.0
    running_corrects = 0.0
    for img, lbl, _ in train_loader:
        img = img.float().to(params.DEVICE)
        preds, mean = icarl.classify(img, exemplars, model, task, train_dataset, mean)
        preds = preds.to(params.DEVICE)
        labels = utils.map_splits(lbl, col).to(params.DEVICE)

        total += len(lbl)
        running_corrects += torch.sum(preds == labels.data).data.item()

    accuracy = float(running_corrects / total)
    print(f'task: {task}', f'train accuracy = {accuracy}')
    accs.append(accuracy)

    total = 0.0
    running_corrects = 0.0
    tot_preds = []
    tot_lab = []
    for img, lbl, _ in test_loader:
        img = img.float().to(params.DEVICE)
        preds, _ = icarl.classify(img, exemplars, model, task, train_dataset, mean)
        preds = preds.to(params.DEVICE)
        labels = utils.map_splits(lbl, col).to(params.DEVICE)

        tot_preds = np.concatenate((tot_preds, preds.data.cpu().numpy()))
        tot_lab = np.concatenate((tot_lab, labels.data.cpu().numpy()))

        total += len(lbl)
        running_corrects += torch.sum(preds == labels.data).data.item()

    accuracy = float(running_corrects / total)
    print(f'task: {task}', f'test accuracy = {accuracy}')
    cf = confusion_matrix(tot_lab, tot_preds)
    df_cm = pd.DataFrame(cf, range(task + params.TASK_SIZE), range(task + params.TASK_SIZE))

# dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"
# torchvision.datasets.utils.download_url(dataset_url,'.')
# print("ciao")
# # Extract from archive
# with tarfile.open('./cifar100.tgz', 'r:gz') as tar:
#     tar.extractall(path='./data')
# print("ciao")
#
# """The dataset is extracted to the directory data/cifar100. It contains 2 folders train and test, containing
# the training set (50000 images) and test set (10000 images) respectively. Each of them contains 10 folders,
# one for each class of images. Let's verify this using os.listdir"""
#
# data_dir = "./data/cifar-100-python"
# print(os.listdir(data_dir))
# hjihd
# cifar100 = torchvision.datasets.CIFAR100("C:/Users\andre\Documents\DATA SCIENCE ENGINEERING\I ANNO\DATA SCIENCE LAB\Python\Open-World-Recognition")
# train_dataset = Dataset(dataset=cifar, train=True)
# print(type(train_dataset))
#
# test_dataset = Dataset(dataset=cifar, train=False)
#
# # splits
# train_splits = train_dataset.splits
# test_splits = test_dataset.splits
#
# # transformers
# train_transformer = transforms.Compose([transforms.RandomCrop(size=32, padding=4),
#                                         transforms.RandomHorizontalFlip(),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                         ])
#
# test_transformer = transforms.Compose([transforms.ToTensor(),
#                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                                        ])
#
# # data loaders
# # train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE,  # num_workers=params.NUM_WORKERS,
# #                          shuffle=True)
# # test_loader = DataLoader(test_dataset, batch_size=params.BATCH_SIZE,  # num_workers=params.NUM_WORKERS,
# #                          shuffle=True)
#
# # model
# model = ResNet.resnet32(num_classes=params.NUM_CLASSES).to(params.DEVICE)
# # simplification: we initialize the network with all the classes
# optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM,
#                             weight_decay=params.WEIGHT_DECAY)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
#                                                  gamma=params.GAMMA)  # allow to change the LR at predefined epochs
#
# # Run
# exemplars = [None] * params.NUM_CLASSES
#
# test_indexes = []
# accs = []
# for task in range(0, params.NUM_CLASSES, params.TASK_SIZE):
#     train_indexes = train_dataset.get_indexes_groups(task)
#     test_indexes = test_indexes + test_dataset.get_indexes_groups(task)
#
#     train_subset = Subset(train_dataset, train_indexes, transform=train_transformer)
#     test_subset = Subset(test_dataset, test_indexes, transform=test_transformer)
#
#     train_loader = DataLoader(train_subset,  # num_workers=params.NUM_WORKERS,
#                               batch_size=params.BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_subset,  # num_workers=params.NUM_WORKERS,
#                              batch_size=params.BATCH_SIZE, shuffle=True)
#
#     model, exemplars = icarl.incremental_train(train_dataset, model, exemplars, task, train_transformer)
#
#     col = []
#     for i, x in enumerate(train_splits[:int(task / params.NUM_CLASSES * params.TASK_SIZE)+1]):
#         v = np.array(x)
#         col = np.concatenate((col, v), axis=None)
#         col = col.astype(int)
#     mean = None
#     total = 0.0
#     running_corrects = 0.0
#     for img, lbl, _ in train_loader:
#         img = img.float().to(params.DEVICE)
#         preds, mean = icarl.classify(img, exemplars, model, task, train_dataset, mean)
#         preds = preds.to(params.DEVICE)
#         labels = utils.map_splits(lbl, col).to(params.DEVICE)
#
#         total += len(lbl)
#         running_corrects += torch.sum(preds == labels.data).data.item()
#
#     accuracy = float(running_corrects / total)
#     print(f'task: {task}', f'train accuracy = {accuracy}')
#     accs.append(accuracy)
#
#     total = 0.0
#     running_corrects = 0.0
#     tot_preds = []
#     tot_lab = []
#     for img, lbl, _ in test_loader:
#         img = img.float().to(params.DEVICE)
#         preds, _ = icarl.classify(img, exemplars, model, task, train_dataset, mean)
#         preds = preds.to(params.DEVICE)
#         labels = utils.map_splits(lbl, col).to(params.DEVICE)
#
#         tot_preds = np.concatenate((tot_preds, preds.data.cpu().numpy()))
#         tot_lab = np.concatenate((tot_lab, labels.data.cpu().numpy()))
#
#         total += len(lbl)
#         running_corrects += torch.sum(preds == labels.data).data.item()
#
#     accuracy = float(running_corrects / total)
#     print(f'task: {task}', f'test accuracy = {accuracy}')
#     cf = confusion_matrix(tot_lab, tot_preds)
#     df_cm = pd.DataFrame(cf, range(task + params.TASK_SIZE), range(task + params.TASK_SIZE))
#     # sn.set(font_scale=.5)  # for label size
#     # sn.heatmap(df_cm, annot=False)
#     # plt.show()
