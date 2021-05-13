import torch
import numpy as np

import params
import utils


def train_network(classes, model, old_model, optimizer, data_loader, scheduler, task, train_splits):
    for epoch in range(params.NUM_EPOCHS):
        length = 0
        running_corrects = 0
        for images, labels, idx in data_loader:
            images = images.float().to(params.DEVICE)
            labels = labels.to(params.DEVICE)
            onehot_labels = torch.eye(params.NUM_CLASSES)
            mapped_labels = utils.map_splits(labels, classes,)
            optimizer.zero_grad()
            old_outputs = old_model(images, features=False)
            outputs = model(images, features=False)
            loss = utils.compute_loss(outputs, old_outputs, onehot_labels, task, train_splits)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None:], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            running_corrects += torch.sum(preds == mapped_labels.data).data.item()
            length += len(images)
            loss.backward()
            optimizer.step()
        accuracy = running_corrects / float(length)
        scheduler.step()
        print('Step: ' + str(task) + ", Epoch: " + str(epoch), ", Loss: " +
              str(loss.item()) + ', Accuracy: ' + str(accuracy))
        return model
