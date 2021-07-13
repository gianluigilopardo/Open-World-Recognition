import math

import numpy as np
import torch.nn as nn
import torch
from owr.our_modifications.open_world import params
from owr.our_modifications.open_world import utils

soft_max = nn.Softmax(dim=1)
relu = nn.ReLU()
unknown_label = -1

def init_thresholds():
    # only 50 the relation between the indexes and the corrisponing real classes is determinated
    # by utils.map_splits(labels, classes)
    thr = nn.parameter.Parameter(torch.ones(params.NUM_CLASSES//2))
    return thr

def compute_thresholds_loss(probs_cut_outputs, mapped_labels, threshold, task):
    # the loss for learning the thresholds
    how_many_discovered_classes = int(task+params.TASK_SIZE)
    # cut_threshold= threshold[:how_many_discovered_classes]
    accumulator = torch.zeros(1, device=params.DEVICE)
    for i, x in enumerate(probs_cut_outputs):
        # 128 times x is a probability vector
        m = 3 * torch.ones(how_many_discovered_classes, device=params.DEVICE)
        m[mapped_labels[i]] = - 1
        accumulator += torch.sum(relu(m * (x - 0.99 * threshold)))
    loss = accumulator / params.BATCH_SIZE
    return loss

def learn_thresholds(threshold, data_loader, classes, model, task):
    # optimize the loss and learn thresholds
    print("\n Learning the thresholds : \n")
    loss_curve = []
    train_accs = [] # accuracy in term of rejection
    # how_many_discovered_classes = int(task+params.TASK_SIZE)
    optimizer = torch.optim.Adam([threshold], lr=params.REJECTION_LR, weight_decay=params.REJECTION_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.REJECTION_STEP_SIZE, gamma=params.REJECTION_GAMMA)
    for epoch in range(params.REJECTION_NUM_EPOCHS):
        ls = []
        length = 0
        running_corrects = 0
        for images, labels, _ in data_loader:
            images = images.float().to(params.DEVICE)
            labels = labels.long().to(params.DEVICE)  # .long()
            mapped_labels = utils.map_splits(labels, classes)
            optimizer.zero_grad()
            with torch.no_grad():
                outputs = model(images, task)
            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), classes[None, :], axis=1).to(params.DEVICE)
            probs_cut_outputs = soft_max(cut_outputs)
            # compute the loss
            loss = compute_thresholds_loss(probs_cut_outputs, mapped_labels, threshold, task)
            ls.append(loss.item())
            # cut_outputs take only the first #task outputs: see simplification in main
            # use the rejection
            preds = rejection(probs_cut_outputs, threshold)
            # if good should not reject
            running_corrects += torch.sum(preds == mapped_labels.data).data.item()
            length += len(images)
            loss.backward()
            optimizer.step()
        accuracy = running_corrects / float(length)
        train_accs.append(accuracy)
        averaged_loss = np.mean(ls)
        loss_curve.append(averaged_loss)
        scheduler.step()
        # print('\n The threesholds are : ')
        # print(threshold)
        print('Step: ' + str(task) + ", Epoch: " + str(epoch) + ", Loss: " +
              str(averaged_loss) + ', Accuracy with rejection: ' + str(accuracy))
    return threshold

def rejection(probs_cut_outputs, threshold):
    maxPred, preds = torch.max(probs_cut_outputs, 1)
    idx = 0
    for (prob,pred) in zip(maxPred, preds):
        if prob < threshold[pred]:
            preds[idx] = unknown_label
        idx += 1
    return preds


# def rejection(outputs, threshold, labels):
#     # labels: batch 128 eleemnti, ciascun elemento è un numero compreso tra 0 e 100 che identifica la classe
#     # outputs: 128x50
#     # treshold: vettore da 50
#
#     preds = soft(outputs)  # calcolo le probabilità
#
#     for label in labels:
#         # es. label = 0, labels[label] = 2 (ovvero il primo elemento del batch è la classe 2), se preds[label = 0] è minore o ugale alla soglia della classe 2 (treshold[labels[label = 0]= 2]]
#         if preds[label] <= treshold[labels[label]]
#             outputs[label] = -1
#
#     return preds
