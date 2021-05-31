import torch
import numpy as np

from owr import params
from owr import utils


def compute_loss(outputs, old_outputs, onehot_labels, task, train_splits):
    criterion = torch.nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()
    outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), \
                                          onehot_labels.to(params.DEVICE)
    # print('outputs: ' + str(outputs))
    classes = utils.get_classes(train_splits, task)
    if task == 0:
        loss = criterion(input=outputs, target=onehot_labels)
    if task > 0:
        target = onehot_labels.clone().to(params.DEVICE)
        target[:, classes] = sigmoid(old_outputs[:, classes]).to(params.DEVICE)
        loss = criterion(input=outputs, target=target)
    return loss


def train_network(classes, model, old_model, optimizer, data_loader, scheduler, task, train_splits):
    for epoch in range(params.NUM_EPOCHS):
        length = 0
        running_corrects = 0
        for images, labels, _ in data_loader:
            images = images.float().to(params.DEVICE)
            labels = labels.long().to(params.DEVICE)  # .long()
            onehot_labels = torch.eye(params.NUM_CLASSES)[labels].to(params.DEVICE)
            mapped_labels = utils.map_splits(labels, classes)
            if( (task>0) and (epoch%10 == 0) ):
                print('mapped_labels: ' + str(mapped_labels))
            optimizer.zero_grad()
            # features=False : use fully connected layer (see ResNet)
            old_outputs = old_model(images, features=False)
            outputs = model(images, features=False)
            if((task>0) and (epoch%10 == 0)):
                print('outputs: ' + str(outputs))
            loss = compute_loss(outputs, old_outputs, onehot_labels, task, train_splits)
            # cut_outputs take only the first #task outputs: see simplification in main
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            if((task>0) and (epoch%10 == 0)):
                print('classes: ' + str(classes[None, :]))
                print('preds: ' + str(preds))
                print('cut_outputs: ', cut_outputs)
            running_corrects += torch.sum(preds == mapped_labels.data).data.item()
            length += len(images)
            loss.backward()
            optimizer.step()
        accuracy = running_corrects / float(length)
        scheduler.step()
        print('Step: ' + str(task) + ", Epoch: " + str(epoch) + ", Loss: " +
              str(loss.item()) + ', Accuracy: ' + str(accuracy))
    return model
