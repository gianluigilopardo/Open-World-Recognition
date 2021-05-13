import copy
from torch.utils.data import DataLoader
from copy import deepcopy

import models
import params
from dataset import *


def incremental_train(train_data, model, exemplars, task, train_transformer, random_s=False):
    train_splits = train_data.splits  # indexes of the splits
    train_indexes = train_data.get_indexes_groups(task)
    classes = utils.get_classes(train_splits, task)
    model = update_representation(train_data, exemplars, model, task, train_indexes, train_splits, train_transformer)
    m = int(params.K / (task + params.TASK_SIZE) + 0.5)   # number of exemplars
    exemplars = reduce_exemplars(exemplars, m)
    exemplars = construct_exemplar_set(exemplars, m, classes[task:], train_data, train_indexes, model, random_s)
    return model, exemplars


def update_representation(train_data, exemplars, model, task, train_indexes, train_splits, train_transformer):
    classes = utils.get_classes(train_splits, task)
    # data_idx contains indexes of images in train_data (new classes) and in exemplars (old classes)
    data_idx = utils.get_indexes(train_indexes, exemplars)
    subset = Subset(train_data, data_idx, train_transformer)
    data_loader = DataLoader(subset, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                                weight_decay=params.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE, gamma=params.GAMMA)
    old_model = deepcopy(model)  # we keep the current (old) model
    old_model.train(False)  # = .test()
    model = models.train_network(classes, model, old_model, optimizer, data_loader, scheduler, task, train_splits)
    return model


def reduce_exemplars(exemplars, m):
    exemplars = copy.deepcopy(exemplars)
    for i, el in enumerate(exemplars):
        if el is not None:
            exemplars[i] = el[:m]
    return exemplars


