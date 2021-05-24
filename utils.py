import numpy as np
import torch
import random

from owr import params
SEED = 42


def get_classes_names(dataset):
    return list(dataset.class_to_idx.keys())


def get_task_indexes(dataset, current_task=0):
    # This method returns a list containing the indexes of all the images
    # belonging to the classes in the current task: [current_task, current_task + TASK_SIZE]
    indexes = []
    current_task = int(current_task / params.TASK_SIZE)
    searched_classes = splitter()[current_task]
    for i in range(len(dataset.data)):
        if dataset.targets[i] in searched_classes:
            indexes.append(i)
    return indexes


def splitter():
    classes_idx = range(params.NUM_CLASSES)
    splits = [None] * int(params.NUM_TASKS)
    for i in range(int(params.NUM_TASKS)):
        random.seed(SEED)
        splits[i] = random.sample(set(classes_idx), k=int(params.TASK_SIZE))
        classes_idx = list(set(classes_idx) - set(splits[i]))
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
