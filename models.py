import torch
import numpy as np

import params
import utils


def compute_loss(outputs, old_outputs, onehot_labels, task, train_splits):
    criterion = torch.nn.BCEWithLogitsLoss()
    m = torch.nn.Sigmoid()
    outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), \
                                          onehot_labels.to(params.DEVICE)
    classes = utils.get_classes(train_splits, task)
    if task == 0:
        loss = criterion(input=outputs, target=onehot_labels)
    if task > 0:
        target = onehot_labels.clone().to(params.DEVICE)
        target[:, classes] = m(old_outputs[:, classes]).to(params.DEVICE)
        loss = criterion(input=outputs, target=target)
    return loss


def train_network(classes, model, old_model, optimizer, data_loader, scheduler, task, train_splits):
    for epoch in range(params.NUM_EPOCHS):
        length = 0
        running_corrects = 0
        for images, labels, idx in data_loader:
            images = images.float().to(params.DEVICE)  #"""convert image into float vector"""
            labels = labels.long().to(params.DEVICE)  # .long() #convert in long
            onehot_labels = torch.eye(params.NUM_CLASSES)[labels].to(params.DEVICE) #che serve?
            mapped_labels = utils.map_splits(labels, classes)
            optimizer.zero_grad()
            # features=False : use fully connected layer (see ResNet)
            old_outputs = old_model(images, features=False)  
            outputs = model(images, features=False)
            loss = compute_loss(outputs, old_outputs, onehot_labels, task, train_splits)
            # cut_outputs take only the first #task outputs: see simplification in main
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
