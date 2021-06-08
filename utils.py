import numpy as np
import torch
import random
import params
import torch.nn as nn
#fkjjk
#from owr import params

SEED = 42


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


def trainLWF(task, train_loader, train_splits):
    print(f'task = {task} ')
    resNet = torch.load('resNet_task' + str(task) + '.pt').train(True)
    old_resNet = torch.load('resNet_task' + str(task) + '.pt').train(False)

    # Define the parameters for traininig:
    optimizer = torch.optim.SGD(resNet.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                                weight_decay=params.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                     gamma=params.GAMMA)  # allow to change the LR at predefined epochs
    current_step = 0
    # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones. Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    # GAMMA: Multiplicative factor of learning rate decay.

    col = np.array(train_splits[int(task / 10)]).astype(int)
    print("train col = ", col)
    print("train col = ", col[None, :])
    ##Train phase
    for epoch in range(params.NUM_EPOCHS):
        lenght = 0
        scheduler.step()  # update the learning rate
        running_corrects = 0

        for images, labels, _ in train_loader:
            images = images.float().to(params.DEVICE)
            labels = labels.to(params.DEVICE)
            mappedLabels = mapFunction(labels,col)  # mapped labels è una matrice di 128 elementi in cui ogni elemento corrisponde all'indice

            onehot_labels = torch.eye(100)[labels].to(
                params.DEVICE)  # it creates the one-hot-encoding list for the labels; needed for BCELoss

            optimizer.zero_grad()  # Zero-ing the gradients

            # Forward pass to the network
            old_outputs = old_resNet(images)  # Yo = CNN(Xn, teta0, tetas)
            outputs = resNet(images)  # Yo_pred
            loss = calculateLoss(outputs, old_outputs, onehot_labels, task, train_splits)

            # Get predictions

            cut_outputs = np.take_along_axis(outputs.to(params.DEVICE), col[None, :], axis=1).to(params.DEVICE)
            _, preds = torch.max(cut_outputs.data, 1)
            # print(preds)

            # Update Corrects
            running_corrects += torch.sum(preds == mappedLabels.data).data.item()
            loss.backward()  # backward pass: computes gradients
            optimizer.step()  # update weights based on accumulated gradients

            current_step += 1
            lenght += len(images)
        # Calculate Accuracy
        accuracy = running_corrects / float(lenght)
        print("At step ", str(task), " and at epoch = ", epoch, " the loss is = ", loss.item(), " and accuracy is = ",
              accuracy)
    torch.save(resNet, 'resNet_task{0}.pt'.format(task + 10))


def testLWF(task, test_loader, test_splits):
    criterion = torch.nn.BCEWithLogitsLoss()
    t_l = 0
    resNet = torch.load('resNet_task' + str(task + 10) + '.pt').eval()  # Set Network to evaluation mode
    running_corrects = 0

    col = []
    # in fase di test verifico su tutti le classi viste fino ad ora, quindi prendo da test splits gli indici dei gruppi da 0 a task
    for i, x in enumerate(test_splits[:int(task / 10) + 1]):
        v = np.array(x)
        col = np.concatenate((col, v), axis=None)
    col = col.astype(int)
    tot_preds = []
    tot_lab = []
    for images, labels, _ in test_loader:
        images = images.float().to(params.DEVICE)
        labels = labels.to(params.DEVICE)
        mappedLabels = mapFunction(labels, col)
        # M1 onehot_labels = torch.eye(task + params.TASK_SIZE)[mappedLabels].to(params.DEVICE) #it creates the one-hot-encoding list for the labels; neede for BCELoss
        onehot_labels = torch.eye(100)[labels].to(params.DEVICE)
        # Forward Pass
        outputs = resNet(images)
        # Get predictions
        outputs = outputs.to(params.DEVICE)

        cut_outputs = np.take_along_axis(outputs, col[None, :], axis=1)
        cut_outputs = cut_outputs.to(params.DEVICE)
        _, preds = torch.max(cut_outputs.data, 1)
        tot_preds = np.concatenate((tot_preds, preds.data.cpu().numpy()))
        tot_lab = np.concatenate((tot_lab, mappedLabels.data.cpu().numpy()))
        # Update Corrects
        running_corrects += torch.sum(preds == mappedLabels.data).data.item()
        print(len(images))
        t_l += len(images)
    # Calculate Accuracy
    accuracy = running_corrects / float(t_l)

    # Calculate Loss

    loss = criterion(outputs, onehot_labels)
    print('Test Loss: {} Test Accuracy : {}'.format(loss.item(), accuracy))
    # cf = confusion_matrix(tot_lab, tot_preds)
    # df_cm = pd.DataFrame(cf, range(task + params.TASK_SIZE), range(task + params.TASK_SIZE))
    # sn.set(font_scale=.4) # for label size
    # sn.heatmap(df_cm, annot=False)
    # plt.show()
    return (accuracy, loss.item())


def calculateLoss(outputs, old_outputs, onehot_labels, task, train_splits):
    criterion = torch.nn.BCEWithLogitsLoss()
    m = nn.Sigmoid()

    outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), onehot_labels.to(
        params.DEVICE)

    col = []
    for i, x in enumerate(train_splits[:int(task / 10)]):
        v = np.array(x)
        col = np.concatenate((col, v), axis=None)
    col = np.array(col).astype(int)

    if (task == 0):
        loss = criterion(outputs, onehot_labels)
    if (task > 0):
        target = onehot_labels.clone().to(params.DEVICE)
        target[:, col] = m(old_outputs[:, col]).to(params.DEVICE)
        loss = criterion(input=outputs, target=target)
    return loss


def mapFunction(labels, splits):
    #   Labels
    #  tensor([13, 31, 17, 94, 14, 35, 17, 94, 81, 14, 86, 13, 14, 35, 28, 14, 28, 86,
    #         81, 94,  3, 86, 28,  3, 35,  3, 17, 86, 31, 13, 13, 31, 35, 28, 17, 17,
    #         86, 81, 81, 31, 14, 13, 14, 14, 28, 17, 86, 13, 13,  3, 14, 13,  3, 17,
    #         86, 94, 28, 94, 86, 31, 17, 13, 35,  3,  3, 81, 28, 86, 35, 86, 86, 86,
    #         86, 31,  3, 94, 28, 94, 94, 14, 94,  3,  3, 17, 31, 13,  3, 94, 81, 81,
    #         81, 35, 31, 86, 17, 81, 94, 28,  3, 35, 94, 28, 17, 31, 17, 86, 28, 28,
    #         28, 17, 86, 13, 31, 31, 14,  3, 81, 31, 81, 17, 81,  3, 17, 81, 35, 28,
    #         17,  3], device='cuda:0')

    # Col
    #  [81 14  3 94 35 31 28 17 13 86]
    # Mapped labels  tensor([8, 5, 7, 3, 1, 4, 7, 3, 0, 1, 9, 8, 1, 4, 6, 1, 6, 9, 0, 3, 2, 9, 6, 2,
    #         4, 2, 7, 9, 5, 8, 8, 5, 4, 6, 7, 7, 9, 0, 0, 5, 1, 8, 1, 1, 6, 7, 9, 8,
    #         8, 2, 1, 8, 2, 7, 9, 3, 6, 3, 9, 5, 7, 8, 4, 2, 2, 0, 6, 9, 4, 9, 9, 9,
    #         9, 5, 2, 3, 6, 3, 3, 1, 3, 2, 2, 7, 5, 8, 2, 3, 0, 0, 0, 4, 5, 9, 7, 0,
    #         3, 6, 2, 4, 3, 6, 7, 5, 7, 9, 6, 6, 6, 7, 9, 8, 5, 5, 1, 2, 0, 5, 0, 7,
    #         0, 2, 7, 0, 4, 6, 7, 2], device='cuda:0')

    m_l = []
    l_splits = list(splits)
    for el in labels:
        m_l.append(l_splits.index(el))
    return torch.LongTensor(m_l).to(params.DEVICE)