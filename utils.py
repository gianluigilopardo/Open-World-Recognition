import numpy as np
import torch
import random
import params
SEED = 42

"""
    we have 100 classes and we have to divide them into tasks, it takes the #classes we have, the #number of tasks and 
    it divides the dataset in vectors of the desired size (10 or 50 in case of open set) and for the dataset retrieves
    the indices of the classes 
    
    
"""
def splitter(): #a
    el = range(params.NUM_CLASSES)
    splits = [None] * int(params.NUM_CLASSES / params.TASK_SIZE)
    for i in range(0, int(params.NUM_CLASSES / params.TASK_SIZE)):
        random.seed(SEED) #because the splits for the training set and test set should be the same so for this reason we put here the random seed
        n = random.sample(set(el), k=int(params.NUM_CLASSES / params.TASK_SIZE))
        splits[i] = n
        el = list(set(el) - set(n))
    return splits


def map_splits(labels, splits):
    """splits restituisce 10 vettori da 10 nel caso base, che poi sarebbero numtask vettori di tasksize, e map splits mi mappa
    le classi dei vettori
    """
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


def get_indexes(train_indexes, exemplars): #indexes of the task
    data_idx = np.array(train_indexes)
    for image_class in exemplars:
        if image_class is not None:
            data_idx = np.concatenate((data_idx, image_class))
    return data_idx
