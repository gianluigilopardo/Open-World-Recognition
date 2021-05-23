import numpy as np
import torch
import random

from owr import params
SEED = 42


def splitter():
    el = range(params.NUM_CLASSES)
    splits = [None] * int(params.NUM_CLASSES / params.TASK_SIZE)
    for i in range(0, int(params.NUM_CLASSES / params.TASK_SIZE)):
        random.seed(SEED)
        n = random.sample(set(el), k=int(params.NUM_CLASSES / params.TASK_SIZE))
        splits[i] = n
        el = list(set(el) - set(n))
    return splits


def map_splits(labels, splits):
    mapped_labels = []
    splits_list = list(splits)
    for label in labels:
        mapped_labels.append(splits_list.index(label))
    return torch.LongTensor(mapped_labels).to(params.DEVICE)


def get_classes(train_splits, task):
    classes = []
    for i, x_split in enumerate(train_splits[:int(task / params.NUM_CLASSES * params.TASK_SIZE) + 1]):
        x_split = np.array(x_split)  # classes in the current split
        classes = np.concatenate((classes, x_split), axis=None)  # classes in all splits up to now
    return classes.astype(int)


def get_indexes(train_indexes, exemplars):
    data_idx = np.array(train_indexes)
    for image_class in exemplars:
        if image_class is not None:
            data_idx = np.concatenate((data_idx, image_class))
    return data_idx
