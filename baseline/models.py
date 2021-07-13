import torch
import numpy as np

from owr.baseline import params
from owr.baseline import utils

# import params
# import utils


def compute_loss(outputs, old_outputs, onehot_labels, task, train_splits):
    criterion = torch.nn.BCEWithLogitsLoss()
    m = torch.nn.Sigmoid() #ultimo layer da aggiungere alla rete per la parte di feature extractor
    outputs, old_outputs, onehot_labels = outputs.to(params.DEVICE), old_outputs.to(params.DEVICE), \
                                          onehot_labels.to(params.DEVICE)
    classes = utils.get_classes(train_splits, task-1)
    if task == 0:
        loss = criterion(input=outputs, target=onehot_labels) #se sono nella prima task allora faccio solo classificazione con BCE
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

def trainLWF(task, train_loader, train_splits):
    print(f'task = {task} ')
    resNet = torch.load('resNet_task' + str(task) + '.pt').train(True) #alleno il modello precedentemente salvato e load tutti i tensors sulla CPU di default ma qui dovrebbe caricarli su GPU
    old_resNet = torch.load('resNet_task' + str(task) + '.pt').train(False) #valuto il modello appena allenato

    # Define the parameters for traininig:
    optimizer = torch.optim.SGD(resNet.parameters(), lr=params.LR, momentum=params.MOMENTUM,
                                weight_decay=params.WEIGHT_DECAY) #ottimizzo modello
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, params.STEP_SIZE,
                                                     gamma=params.GAMMA)  # allow to change the LR at predefined epochs
    # Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones. Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    # GAMMA: Multiplicative factor of learning rate decay.
    current_step = 0

    col = np.array(train_splits[int(task / 10)]).astype(int) #salvo in un vettore le classi apparteneti alla task
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
            # print(labels)
            mappedLabels = utils.map_splits(labels, col)
            # print(mappedLabels)
            onehot_labels = torch.eye(100)[labels].to(
                params.DEVICE)  # it creates the one-hot-encoding list for the labels; needed for BCELoss

            optimizer.zero_grad()  # Zero-ing the gradients

            # Forward pass to the network
            old_outputs = old_resNet(images)  # Yo = CNN(Xn, teta0, tetas)
            outputs = resNet(images)  # Yo_pred
            loss = compute_loss(outputs, old_outputs, onehot_labels, task, train_splits)

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
        mappedLabels = utils.map_splits(labels, col)
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
