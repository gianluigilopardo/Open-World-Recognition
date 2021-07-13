import torch
import numpy as np

from owr.ablation_study.bic_method import utils
from owr.ablation_study.bic_method import params


def compute_loss(outputs, old_outputs, onehot_labels, task, train_splits):
    criterion = torch.nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()
    outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), \
                                          onehot_labels.to(params.DEVICE)
    # print('outputs: ' + str(outputs))
    classes = utils.get_classes(train_splits, task - 1) # old classes
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
        for images, labels, idx in data_loader:
            images = images.float().to(params.DEVICE)  #"""convert image into float vector"""
            labels = labels.long().to(params.DEVICE)  # .long() #convert in long
            onehot_labels = torch.eye(params.NUM_CLASSES)[labels].to(params.DEVICE) #serve per la BCELoss
            mapped_labels = utils.map_splits(labels, classes)
            optimizer.zero_grad() #azzero i gradienti
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


def train_network_lower_bound(classes, model, optimizer, data_loader, scheduler, task):
    for epoch in range(params.NUM_EPOCHS):
        length = 0
        running_corrects = 0
        for images, labels, _ in data_loader:
            images = images.float().to(params.DEVICE)
            labels = labels.long().to(params.DEVICE)  # .long()
            onehot_labels = torch.eye(params.NUM_CLASSES)[labels].to(params.DEVICE)
            mapped_labels = utils.map_splits(labels, classes)
            optimizer.zero_grad()
            # features=False : use fully connected layer (see ResNet)
            # old_outputs = old_model(images, features=False)
            outputs = model(images, features=False)
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(outputs, onehot_labels)
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

def evaluate_model(model, task, new_test_loader, test_loader):
    model.architecture.eval()
    splits = utils.splitter()
    with torch.no_grad():
        print('\n EVALUATION before Bias Correction\n')
        classes = []
        for i, x in enumerate(splits[:int(task / params.TASK_SIZE) + 1]):
            v = np.array(x)
            classes = np.concatenate((classes, v), axis=None)
            classes = classes.astype(int)
        # mean = None
        total = 0.0
        running_corrects = 0.0
        for img, lbl, _ in new_test_loader:
            img = img.float().to(params.DEVICE)
            outputs = model(img, task)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            preds = preds.to(params.DEVICE)
            labels = utils.map_splits(lbl, classes).to(params.DEVICE)
            total += len(lbl)
            running_corrects += torch.sum(preds == labels.data).data.item()
        accuracy = float(running_corrects / total)
        print(f'task: {task}', f'test accuracy on only new classes = {accuracy}')

        # ###### OLD #####
        # total = 0.0
        # running_corrects = 0.0
        # for img, lbl, _ in old_test_loader:
        #     img = img.float().to(params.DEVICE)
        #     outputs = model(img, task)
        #     cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
        #     _, preds = torch.max(cut_outputs.data, 1)
        #     preds = preds.to(params.DEVICE)
        #     labels = utils.map_splits(lbl, classes).to(params.DEVICE)
        #     total += len(lbl)
        #     running_corrects += torch.sum(preds == labels.data).data.item()
        # accuracy = float(running_corrects / total)
        # print(f'task: {task}', f'test accuracy on only old classes = {accuracy}')

        ##### OLD & NEW ####
        total = 0.0
        running_corrects = 0.0
        tot_preds = []
        tot_lab = []
        for img, lbl, _ in test_loader:
            img = img.float().to(params.DEVICE)
            outputs = model(img, task)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            preds = preds.to(params.DEVICE)
            labels = utils.map_splits(lbl, classes).to(params.DEVICE)

            tot_preds = np.concatenate((tot_preds, preds.data.cpu().numpy()))
            tot_lab = np.concatenate((tot_lab, labels.data.cpu().numpy()))

            total += len(lbl)
            running_corrects += torch.sum(preds == labels.data).data.item()
            # print('running_corrects: ' + str(running_corrects))
        accuracy = float(running_corrects / total)
        print(f'task: {task}', f'test accuracy on old and new classes = {accuracy}')
