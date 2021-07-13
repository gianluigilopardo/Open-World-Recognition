from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import numpy as np
import pandas as pd
from numpy import random
import time

from owr import utils
from owr import params
from owr.dataset import *
from owr import models 

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import math


# import utils
# import params
# import models
# from dataset import *


# Algorithm 1 iCaRL CLASSIFY
def classify(images, exemplars, model, task, train_dataset, mean=None, classifier='nme'):
    if classifier == 'nme':
        preds, mean = classify_nme(images, exemplars, model, task, train_dataset, mean)
        return preds, mean
    else:
        preds, _ = classify_models(images, exemplars, model, task, train_dataset, mean, classifier)
        return preds, _

def classify_models(images, exemplars, model, task, train_dataset, mean, classifier):
    splits = utils.splitter()
    model.train(False)
    with torch.no_grad():
        images = images.float().to(params.DEVICE)
        x_test = model(images, features=True)
        x_test /= torch.norm(x_test, p=2)

    analyzed_classes = []
    for i in range(int(task / params.TASK_SIZE)+1):
        analyzed_classes = np.concatenate((analyzed_classes, splits[i]))
  
    l = []
    for k in range(len(analyzed_classes)):
        class_k = int(analyzed_classes[k])
        l.extend(exemplars[class_k])
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    subset = Subset(train_dataset, l, transformer)
    data_loader = DataLoader(subset, num_workers=params.NUM_WORKERS, batch_size=params.BATCH_SIZE)
  
    df = pd.DataFrame(columns = ['data', 'labels'])
    for images, labels, idxs in data_loader:
        with torch.no_grad():
            images = images.float().to(params.DEVICE)
            x = model(images, features=True)
            x /= torch.norm(x, p=2)
            tmp = pd.DataFrame()
            tmp['data'] = x
            tmp['labels'] = labels
            df = df.append(tmp)
    classifier.fit(list(df['data']),list(df['labels']) )
    preds = classifier.predict(x_test)
    return torch.tensor(preds), 0

def classify_nme(images, exemplars, model, task, train_dataset, mean):
    splits = utils.splitter()
    preds = []
    num_classes = task + params.TASK_SIZE
    means = torch.zeros((num_classes, 64)).to(params.DEVICE)
    model.train(False)
    images = images.float().to(params.DEVICE)
    phi_x = model(images, features=6)
    phi_x /= torch.norm(phi_x, p=2)
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    analyzed_classes = []
    if mean is None:
        for i in range(int(task / params.TASK_SIZE)+1):
            analyzed_classes = np.concatenate((analyzed_classes, splits[i]))
        for k in range(len(analyzed_classes)):
            counter = 0  # number of images
            class_k = int(analyzed_classes[k])
            ss = Subset(train_dataset, exemplars[class_k], transformer)
            data_loader = DataLoader(ss, num_workers=params.NUM_WORKERS,
                                     batch_size=params.BATCH_SIZE)
            for images, labels, idxs in data_loader:  # batches
                counter += len(images)
                with torch.no_grad():
                    images = images.float().to(params.DEVICE)
                    x = model(images, features=6)
                    x /= torch.norm(x, p=2)
                means[k] += torch.sum(x, dim=0)
            means[k] = means[k] / counter  # average
            means[k] = means[k] / means[k].norm()
    else:
        means = mean
    for data in phi_x:
        pred = np.argmin(np.sqrt(np.sum((data.data.cpu().numpy() - means.data.cpu().numpy()) ** 2, axis=1)))
        preds.append(pred)
    return torch.tensor(preds), means

# Algorithm 2 iCaRL INCREMENTAL TRAIN
def incremental_train(train_dataset, model, exemplars, task, train_transformer, random_s=False, loss_version='opt1'):
    train_splits = utils.splitter()  # indexes of the splits
    train_indexes = utils.get_task_indexes(train_dataset, task)
    classes = utils.get_classes(train_splits, task)
    secs1  = int(round(time.time()))
    model = update_representation(train_dataset, exemplars, model, task, train_indexes, train_splits, train_transformer, loss_version)
    secs2  = int(round(time.time()))
    m = int(params.K / (task + params.TASK_SIZE) + 0.5)  # number of exemplars
    exemplars = reduce_exemplars(exemplars, m)
    secs3  = int(round(time.time()))
    exemplars = construct_exemplar_set(exemplars, m, classes[task:], train_dataset, train_indexes, model, random_s)
    secs4  = int(round(time.time()))
    print('udpate representation: ', secs2-secs1, 'reduce exemplars:', secs3-secs2, 'costruct exemplars:', secs4-secs3)
    return model, exemplars


# Algorithm 3 iCaRL UPDATE REPRESENTATION
def update_representation(train_dataset, exemplars, model, task, train_indexes, train_splits, train_transformer, loss_version='opt1'):
    classes = utils.get_classes(train_splits, task)
    # data_idx contains indexes of images in train_data (new classes) and in exemplars (old classes)
    data_idx = utils.get_indexes(train_indexes, exemplars)
    subset = Subset(train_dataset, data_idx,
                    train_transformer)
    data_loader = DataLoader(subset, batch_size=params.BATCH_SIZE, num_workers=params.NUM_WORKERS,
                             shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                                weight_decay=params.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE, gamma=params.GAMMA)
    old_model = copy.deepcopy(model)  # we keep the current (old) model
    old_model.train(False)  # = .test()
    model.train(True)
    model = models.train_network(classes, model, old_model, optimizer, data_loader, scheduler, task, train_splits, loss_version)
    return model


# Algorithm 4 iCaRL CONSTRUCT EXEMPLAR SET
def construct_exemplar_set(exemplars, m, classes, train_dataset, train_indexes, model, random_set=False):
    # classes: current classes
    exemplars = copy.deepcopy(exemplars)
    for image_class in classes:
        images_idx = []
        for i in train_indexes:
            image, label = train_dataset.__getitem__(i)
            if label == image_class:
                images_idx.append(i)
        if random_set is not True:
            exemplars[image_class] = generate_new_exemplars(images_idx, m, model, train_dataset)
        else:
            exemplars[image_class] = random.sample(images_idx, m)
    return exemplars


# Algorithm 5 iCaRL REDUCE EXEMPLAR SET
def reduce_exemplars(exemplars, m):
    exemplars = copy.deepcopy(exemplars)
    for i, exemplar_class_i in enumerate(exemplars):
        if exemplar_class_i is not None:
            exemplars[i] = exemplar_class_i[:m]
    return exemplars


def generate_new_exemplars(images_idx, m, model, train_dataset):
    model = model.train(False)
    features = []
    with torch.no_grad():
        for idx in images_idx:
            image, label = train_dataset.__getitem__(idx)
            image = torch.tensor(image).unsqueeze(0).float()
            x = model(image.to(params.DEVICE), features=True).data.cpu().numpy()
            x = x / np.linalg.norm(x)
            features.append(x[0])
    features = np.array(features)
    means = np.mean(features, axis=0)
    means = means / np.linalg.norm(means)
    new_exemplars = []
    phi_new_exemplars = []
    images_idx = copy.deepcopy(images_idx)
    for k in range(0, m):
        phi_x = features  # features of all images in current class
        phi_P = np.sum(phi_new_exemplars, axis=0)  # sum on all existing exemplars
        mu = 1 / (k + 1) * (phi_x + phi_P)
        exemplar_idx = np.argmin(np.sqrt(np.sum(means - mu) ** 2),
                                 axis=0)  # idx of exemplar with min euclidean distance
        new_exemplars.append(images_idx[exemplar_idx])
        phi_new_exemplars.append(features[exemplar_idx])
        features = np.delete(features, exemplar_idx, axis=0)
        images_idx.pop(exemplar_idx)
    return new_exemplars
