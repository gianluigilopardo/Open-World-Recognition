import numpy as np
import torch
import random
import math
from owr import params
from sklearn.model_selection import train_test_split

SEED = 42


# seeds : 42, 24, 1993

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


def get_task_indexes_with_labels(dataset, current_task=0):
    # This method returns a list containing the indexes of all the images
    # belonging to the classes in the current task: [current_task, current_task + TASK_SIZE]
    indexes = []
    corrisponding_labels = []
    current_task = int(current_task / params.TASK_SIZE)
    searched_classes = splitter()[current_task]
    for i in range(len(dataset.data)):
        if dataset.targets[i] in searched_classes:
            indexes.append(i)
            corrisponding_labels.append(dataset.targets[i])
    return indexes, corrisponding_labels


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
    # return the already seen classes at the current task
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


def get_train_val_indexes(train_indexes, corrisponding_training_labels, exemplars, proportion, task):
    if task == 0:
        train_idx, val_idx, _, _ = train_test_split(train_indexes, corrisponding_training_labels,
                                                    test_size=proportion, random_state=42,
                                                    stratify=corrisponding_training_labels)
    if task > 0:
        exemplars_indexes = []
        exemplars_labels = []
        # remember that task is the number of old classes at this time
        balanced_number_of_samples = math.floor(task * proportion * (params.K / task))
        for label, image_class in enumerate(exemplars):
            if image_class is not None:
                exemplars_indexes = exemplars_indexes + image_class
                exemplars_labels = exemplars_labels + [label for x in image_class]

        train_old_idx, val_old_idx, _, _ = train_test_split(train_indexes, corrisponding_training_labels,
                                                            test_size=int(balanced_number_of_samples), random_state=42,
                                                            stratify=corrisponding_training_labels)
        train_new_idx, val_new_idx, _, _ = train_test_split(exemplars_indexes, exemplars_labels,
                                                            test_size=int(balanced_number_of_samples), random_state=42,
                                                            stratify=exemplars_labels)
        train_idx = train_old_idx + train_new_idx
        val_idx = val_old_idx + val_new_idx
    return train_idx, val_idx

def get_train_val1_val2_indexes_for_rejection(train_indexes, corrisponding_training_labels, exemplars, proportion, task):
    if task == 0:
        train_idx, val_rejection_idx, _, _ = train_test_split(train_indexes, corrisponding_training_labels,
                                                                  test_size=proportion, random_state=42,
                                                                  stratify=corrisponding_training_labels)
    if task > 0:
        exemplars_indexes = []
        exemplars_labels = []
        # remember that task is the number of old classes at this time
        balanced_number_of_samples = math.floor(task * proportion * (params.K / task))
        for label, image_class in enumerate(exemplars):
            if image_class is not None:
                exemplars_indexes = exemplars_indexes + image_class
                exemplars_labels = exemplars_labels + [label for x in image_class]

        train_old_idx, val_old_idx, train_old_lbl, val_old_lbl = train_test_split(train_indexes, corrisponding_training_labels,
                                                                                  test_size=int(balanced_number_of_samples), random_state=42,
                                                                                  stratify=corrisponding_training_labels)
        # split half and half the (balanced) validation set for the new classes
        val_old_idx, val_old_rejection_idx, _, _ = train_test_split(val_old_idx, val_old_lbl,
                                                                    test_size=0.5, random_state=42,
                                                                    stratify=val_old_lbl)

        train_new_idx, val_new_idx, train_new_lbl, val_new_lbl = train_test_split(exemplars_indexes, exemplars_labels,
                                                                                  test_size=int(balanced_number_of_samples), random_state=42,
                                                                                  stratify=exemplars_labels)
        val_new_idx, val_new_rejection_idx, _, _ = train_test_split(val_new_idx, val_new_lbl,
                                                                    test_size=0.5, random_state=42,
                                                                    stratify=val_new_lbl)
        train_idx = train_old_idx + train_new_idx
        val_idx = val_old_idx + val_new_idx
        val_rejection_idx = val_old_rejection_idx + val_new_rejection_idx
    if task == 0:
        val_idx = []
    return train_idx, val_idx, val_rejection_idx
