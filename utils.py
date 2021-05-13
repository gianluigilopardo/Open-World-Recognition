import numpy as np
import torch

import params
from params import *


def splitter():
    el = range(NUM_CLASSES)
    splits = [None] * TASK_SIZE
    for i in range(0, TASK_SIZE):
        # random.seed(SEED)
        n = random.sample(set(el), k=int(NUM_CLASSES / TASK_SIZE))
        splits[i] = n
        el = list(set(el) - set(n))
    return splits


def map_splits(labels, splits):
    mapped_labels = []
    splits = list(splits)
    for label in labels:
        mapped_labels.append(splits.index(label))
    return torch.LongTensor(mapped_labels).to(params.DEVICE)


def compute_loss(outputs, old_outputs, onehot_labels, task, train_splits):
    criterion = torch.nn.BCEWithLogitsLoss
    m = torch.nn.Sigmoid()
    outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), \
                                          onehot_labels.to(params.DEVICE)
    col = []
    for i, x in enumerate(train_splits[:int(task / params.TASK_SIZE) + 1]):
        x = np.array(x)
        col = np.concatenate((col, x), axis=None)
    col = np.array(col).astype(int)
    if task == 0:
        loss = criterion(outputs, onehot_labels)
    if task > 0:
        target = onehot_labels.clone().to(params.DEVICE)
        target[:, col] = m(old_outputs[:, col]).to(params.DEVICE)
        loss = criterion(input=outputs, target=target)
    return loss


def get_classes(train_splits, task):
    classes = []
    for i, x_split in enumerate(train_splits[:int(task / params.TASK_SIZE) + 1]):
        x_split = np.array(x_split)  # classes in the current split
        classes = np.concatenate((classes, x_split), axis=None)  # classes in all splits up to now
    return classes.astype(int)


def get_indexes(train_indexes, exemplars):
    data_idx = np.array(train_indexes)
    for image_class in exemplars:
        if image_class is not None:
            data_idx = np.concatenate((data_idx, image_class))
    return data_idx
