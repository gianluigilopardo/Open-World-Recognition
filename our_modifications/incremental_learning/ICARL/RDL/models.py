import torch
import numpy as np
import math
from owr.our_modifications.incremental_learning.ICARL.RDL import params
from owr.our_modifications.incremental_learning.ICARL.RDL import utils
# import params
# import utils
from torch.nn import functional as F

def compute_loss(outputs, onehot_labels, task, train_splits):
    classes = utils.get_classes(train_splits, task-1)
    criterion = torch.nn.BCEWithLogitsLoss()
    outputs, onehot_labels = outputs.to(params.DEVICE), onehot_labels.to(params.DEVICE)
    loss = criterion(input=outputs, target=onehot_labels)
    return loss

def compute_loss_cosine(features, old_features):
    loss = torch.nn.CosineEmbeddingLoss()
    ys = torch.tensor([1]*len(features)).to(params.DEVICE)
    features, old_features = features.to(params.DEVICE), old_features.to(params.DEVICE)
    features = F.normalize(features, p=2, dim=1)
    old_features = F.normalize(old_features, p=2, dim=1)
    return loss(features, old_features, ys)


def train_network(classes, model, old_model, optimizer, data_loader, scheduler, task, train_splits,loss_version='opt1'):
    for epoch in range(params.NUM_EPOCHS):
        rand_level = np.random.randint(0,params.NETWORK_DEEP,1)[0]
        length = 0
        running_corrects = 0
        for images, labels, idx in data_loader:
            images = images.float().to(params.DEVICE)  #"""convert image into float vector"""
            labels = labels.long().to(params.DEVICE)  # .long() #convert in long
            onehot_labels = torch.eye(params.NUM_CLASSES)[labels].to(params.DEVICE) #serve per la BCELoss
            mapped_labels = utils.map_splits(labels, classes)
            optimizer.zero_grad() #azzero i gradienti
            outputs = model(images, features=0)
            loss = compute_loss(outputs, onehot_labels, task, train_splits)

            if task > 0:

              gamma = 1/5*math.sqrt(task/params.TASK_SIZE)
              features = model(images, features=rand_level)
              old_features = old_model(images, features=rand_level)
              conservative_loss = compute_loss_cosine(features, old_features)
              loss = loss + gamma*conservative_loss

            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            running_corrects += torch.sum(preds == mapped_labels.data).data.item()
            length += len(images)
            loss.backward()
            optimizer.step()
        accuracy = running_corrects / float(length)
        scheduler.step()
        print('Step: ' + str(task) + ", Epoch: " + str(epoch) + ", Loss: " +
              str(loss.item()) + ', Accuracy: ' + str(accuracy))
    return model
